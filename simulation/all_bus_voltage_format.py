"""all_bus_voltage_format.py -- a compact, exactly reversible container for the
per-bus voltage tables.

WHY THIS EXISTS
---------------
`global_voltages/all_bus_voltages_phase_{A,B,C}.csv` is ~98 % of a yearly
result tree: 907 buses x 1441 steps of float32, written as decimal text
(~12.9 MB per file, ~16 GB per Phase 3 batch). compress_results.py already
gzips them 4.6x. This gets a further **2.6x on top of that** -- ~11x against
the raw CSV -- and is *faster* to read than the gzipped text, because nothing
has to parse a million decimal numbers back.

The saving comes from the representation, not from a stronger codec (this venv
has no zstd/pyarrow/blosc, only stdlib gzip/bz2/lzma plus numpy):

  * **float32 is the real dtype.** network_setup.export_results casts to
    float32 before writing, so the text is pandas' shortest round-trip repr of
    a float32 -- 8 significant digits of a number near 1.0. Storing the array
    instead of its decimal spelling drops 12.9 MB to 5.2 MB before any codec.
  * **Fixed point.** Those values are exactly representable as integers at
    1e-8 (verified per file, and asserted at encode time, never assumed).
  * **Delta along time.** A bus's voltage moves slowly from step to step, so
    consecutive differences are small integers where the absolute values are
    ~1e8. This is where most of the ratio comes from.
  * **Byte-plane shuffle**, the trick blosc calls SHUFFLE: transpose the four
    bytes of each int32 so all the high bytes sit together, then all the next
    ones, and so on. Runs of near-identical bytes compress far better than
    interleaved ones.
  * **lzma** over the result.

Measured on the largest real file (1441x907, 12.92 MB of CSV):

    gzip-6 CSV text (what compress_results.py writes)   2.837 MB   0.11 s read
    lzma-1 CSV text                                     2.624 MB   0.18 s
    float32 raw + gzip-6                                2.263 MB   0.01 s
    float32 shuffled + lzma-1                           1.818 MB   0.08 s
    int32 delta + shuffle + gzip-6                      1.455 MB   0.04 s
    int32 delta + shuffle + lzma-6   <-- this format    1.180 MB   0.07 s

EXACTNESS
---------
This is a *re-encoding*, not a lossy approximation: decode_to_csv_bytes()
reproduces the original file byte for byte, header, index, line endings and
all (verified on 30 real files across runs and phases). Two properties make
that possible and both are checked at encode time rather than trusted:

  * the values are exactly representable at the chosen decimal scale, and
  * the scaled integers fit in int32.

If either fails, encoding raises and the caller keeps the CSV. That is the
same "refuse rather than lose data" stance as compress_results.py's
hash-before-delete.

The container is self-describing -- shape, scale, filters, codec, column names
and index all travel in the header -- so a future reader needs this file and
nothing else, and `compress_results.py --decompress` always gets the plain
CSVs back.
"""
from __future__ import annotations

import io
import json
import lzma
import struct

import numpy as np

MAGIC = b"ABV1"
SUFFIX = ".abv"

# Decimal places at which the stored values are exact integers. 8 is what every
# real file needs (pandas prints a float32 near 1.0 with up to 8 significant
# digits); it is written into the header regardless, so a file that needed a
# different one still reads back.
DEFAULT_SCALE = 8
LZMA_PRESET = 6

# pandas writes these with CRLF on Windows, and restoring the original bytes
# means restoring that too. Carried in the header rather than assumed, so a
# file written on another platform round-trips as itself.
DEFAULT_LINETERMINATOR = "\r\n"


def _shuffle(array: np.ndarray) -> bytes:
    """Group the Nth byte of every element together (blosc's SHUFFLE filter)."""
    itemsize = array.dtype.itemsize
    flat = np.frombuffer(array.tobytes(), np.uint8).reshape(-1, itemsize)
    return flat.T.tobytes()


def _unshuffle(raw: bytes, dtype, count: int) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    planes = np.frombuffer(raw, np.uint8).reshape(itemsize, count)
    return np.frombuffer(planes.T.tobytes(), dtype)


def encode_frame(frame, scale: int = DEFAULT_SCALE,
                 lineterminator: str = DEFAULT_LINETERMINATOR) -> bytes:
    """One DataFrame of per-bus voltages -> the container's bytes.

    Raises ValueError when the values cannot be held exactly at `scale`
    decimals or overflow int32 -- the caller must then keep the CSV rather
    than store something that will not come back.
    """
    values = frame.to_numpy(dtype=np.float64)
    factor = 10.0 ** scale
    scaled = np.rint(values * factor)
    if not np.array_equal(scaled / factor, values):
        raise ValueError(
            f"values are not exact at {scale} decimal places, so this encoding "
            f"would lose data")
    if scaled.min() < np.iinfo(np.int32).min or scaled.max() > np.iinfo(np.int32).max:
        raise ValueError("scaled values overflow int32")

    quantised = scaled.astype(np.int32)
    # Row 0 differences against zero, so the cumulative sum reconstructs the
    # absolute values -- prepending the first row instead would silently drop it.
    deltas = np.diff(quantised, axis=0,
                     prepend=np.zeros((1, quantised.shape[1]), np.int32)).astype(np.int32)

    header = {
        "shape": list(frame.shape),
        "scale": scale,
        "dtype": "float32",
        "filters": ["delta_time", "shuffle"],
        "codec": f"lzma-{LZMA_PRESET}",
        "lineterminator": lineterminator,
        "index_name": frame.index.name,
        "index": [str(i) for i in frame.index],
        "columns": [str(c) for c in frame.columns],
    }
    header_blob = lzma.compress(json.dumps(header).encode("utf-8"), preset=LZMA_PRESET)
    payload = lzma.compress(_shuffle(deltas), preset=LZMA_PRESET)
    return MAGIC + struct.pack("<I", len(header_blob)) + header_blob + payload


def encode_csv_bytes(raw: bytes, scale: int = DEFAULT_SCALE) -> bytes:
    """The bytes of one all_bus_voltages CSV -> the container's bytes."""
    import pandas as pd

    frame = pd.read_csv(io.BytesIO(raw), index_col=0)
    # Found from the first line break in the whole buffer, not from a fixed-size
    # prefix: the header row is 907 column names, ~6 kB, so a 4 kB sniff never
    # reaches the first newline and silently concludes "\n" for a CRLF file.
    first = raw.find(b"\n")
    terminator = "\r\n" if first > 0 and raw[first - 1:first] == b"\r" else "\n"
    return encode_frame(frame, scale=scale, lineterminator=terminator)


def _read_header(blob: bytes) -> tuple[dict, int]:
    if not blob.startswith(MAGIC):
        raise ValueError(f"not an {MAGIC.decode()} container")
    (header_len,) = struct.unpack("<I", blob[4:8])
    header = json.loads(lzma.decompress(blob[8:8 + header_len]).decode("utf-8"))
    return header, 8 + header_len


def decode_to_frame(blob: bytes):
    """The container's bytes -> a DataFrame identical to reading the CSV.

    float64, deliberately, even though the stored dtype is float32: that is
    what `pd.read_csv` on the original file returns, and `scaled / 10**scale`
    in float64 reproduces the parsed text *exactly* (the whole point of the
    fixed-point step). Handing back float32 instead would be the true value
    but would differ from the CSV in the eighth decimal, so every caller
    comparing the two -- tests included -- would have to know which spelling
    of the file it happened to read. Only CSV regeneration narrows to float32,
    because that is the dtype the text was printed from.
    """
    import pandas as pd

    header, offset = _read_header(blob)
    rows, cols = header["shape"]
    deltas = _unshuffle(lzma.decompress(blob[offset:]), np.int32, rows * cols)
    quantised = np.cumsum(deltas.reshape(rows, cols), axis=0, dtype=np.int64)
    values = quantised / 10.0 ** header["scale"]
    frame = pd.DataFrame(values, columns=header["columns"],
                         index=pd.Index(header["index"], name=header.get("index_name")))
    return frame


def decode_to_csv_bytes(blob: bytes) -> bytes:
    """The container's bytes -> the original CSV, byte for byte."""
    header, _offset = _read_header(blob)
    frame = decode_to_frame(blob).astype(header.get("dtype", "float32"))
    buffer = io.StringIO()
    frame.to_csv(buffer, lineterminator=header.get("lineterminator", DEFAULT_LINETERMINATOR))
    return buffer.getvalue().encode("utf-8")


def is_container(path) -> bool:
    """Cheap sniff, so a caller need not trust the extension alone."""
    try:
        with open(path, "rb") as handle:
            return handle.read(len(MAGIC)) == MAGIC
    except OSError:
        return False
