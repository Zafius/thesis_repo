"""compress_results.py -- shrink a finished sim_results/ tree in place.

WHAT IT COMPRESSES, AND WHY ONLY THAT
-------------------------------------
A Phase 3 (--yearly) result tree is ~16 GB, and ~98% of that is three files
per week folder:

    global_voltages/all_bus_voltages_phase_{A,B,C}.csv   ~12.6 MB each

which is the full per-bus voltage trajectory (907 buses x 1441 steps, float32
rendered as CSV text). Everything else in a week folder put together is under
1 MB: the losses CSV, voltage_quality.csv, transformer_loading.csv, the
per-inverter P/Q/V profiles, the two PDFs.

Those three files used to be *write-only*: network_setup.export_results wrote
them (see its log_all_bus_voltages branch) and nothing read them back. That is
no longer true -- the desktop app's Network Map plots any bus's trace straight
out of them -- so this script's output has a consumer now, and both spellings
it can produce are readable in place, without unpacking. Nothing else changed:
compare_control_strategies.py still takes its voltage numbers from
global_voltages/simulation_results_phase_*.csv (per-step min/max) and from
voltage_quality.csv (the exact all-bus violation figures, computed during the
run and persisted); compute_curtailment.py reads inverter_profiles/ only. They
are kept as raw archival evidence for the thesis, not as analysis input.

So this script zips exactly those files and leaves every file an analysis
script touches as plain CSV. That is the whole point of the narrow scope: no
consumer needs to learn about the archive format, no code path changes, and
the saving is still ~4.9x on 98% of the bytes -- about 12.5 GB back per Phase
3 tree.

DEFLATE level 6 is the deliberate choice over the alternatives measured on
this data (12.92 MB source file, this machine, 2026-07-27):

    gzip-6    2.63 MB   4.9x    19 MB/s     <-- same stream, chosen
    bz2-9     2.67 MB   4.8x    11 MB/s
    lzma-p1   2.41 MB   5.4x    10 MB/s
    lzma-p6   1.69 MB   7.6x     1 MB/s     (~36 h for the full backlog)

gzip -9 was also tried: 2.51 MB, i.e. 2% smaller for roughly double the CPU.
Not worth it. Parquet/zstd would beat all of these but need pyarrow, which
this venv does not have and which is not worth adding mid-thesis.

A one-entry .zip at level 6 compresses to the same bytes as gzip-6 above
(2.68 MB on this file -- it is the same DEFLATE stream, plus ~140 B of
container overhead) at the same speed. The move from .gz to .zip below is
about what can open the archive without installing anything, not about
compression: a .zip is what Windows Explorer, macOS Finder and every archive
tool extract natively; a lone .gz needs 7-Zip or similar on a stock Windows
machine. Archives already on disk as .gz (anything written before this
change) are still read transparently -- see _read_any and decompress_one.

--repack: A DENSER FORMAT, 2.6x SMALLER AGAIN
---------------------------------------------
Everything above compresses the *text*. `--repack` stores the numbers instead,
in the container described in all_bus_voltage_format.py -- float32 as exact
fixed-point integers, delta-encoded along time, byte-plane shuffled, then
lzma. On the same 12.92 MB file:

    zip-6 CSV text (default)            2.837 MB    4.6x    0.11 s to read
    .abv container (--repack)           1.180 MB   10.9x    0.07 s to read

so it is ~2.6x smaller than the .zip *and* quicker to read, because nothing
has to parse a million decimal numbers back. It is still exactly reversible:
`--decompress` restores the original CSVs byte for byte, verified per file
against a SHA-256 of the source before anything is deleted, and refused
outright if the values cannot be reproduced exactly.

The trade is portability. A .zip is readable by any tool on any machine with
no extra software; the container needs all_bus_voltage_format.py, which lives
in this repo beside the data it describes. For archival evidence that matters,
so zip remains the default and --repack is opt-in.

SAFETY
------
Each file is compressed to a .tmp sidecar, the .tmp is then read back and
decompressed, and its SHA-256 is compared against the SHA-256 of the original
before anything is deleted. Only on a byte-exact match is the .tmp renamed
into place and the original removed. Any failure (bad hash, full disk, locked
file) leaves the original exactly where it was and reports an error -- the
same "abort deletion to keep your files safe" stance as
sim_utils.compress_and_delete.

The operation is fully reversible: `--decompress` restores the original .csv
files, and a lone file can always be recovered with any zip tool -- including
Windows' own "Extract All", no download required.

Re-running on an already-compressed tree is a no-op, so it is safe to call
from a resumable plan that may retry a task.

USAGE
-----
    python compress_results.py sim_results/2026-07-22_18.18_p3_oslo_no1_base
    python compress_results.py --all                 # every eligible tree
    python compress_results.py --all --match p3_     # only Phase 3 trees
    python compress_results.py --all --dry-run       # report, change nothing
    python compress_results.py <root> --decompress   # put the CSVs back (.zip or legacy .gz)
    python compress_results.py <root> --repack       # the dense .abv container
    python compress_results.py --all --repack --dry-run

--all skips any tree modified in the last --min-age-minutes (default 15) so it
cannot touch a run that is still being written. An explicitly named root is
trusted and not age-checked -- that is the mode run_test_plan.py uses, where
the task's dependencies already guarantee the run finished.

Each processed tree gets a compression_report.json recording what was done and
how many bytes were saved; run_test_plan.py reads that back into its progress
file so TEST_PLAN_EXECUTION.md can show the saving.
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import all_bus_voltage_format as abv

SIM_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SIM_DIR / "sim_results"

# The only files this script touches. Relative glob, matched under each week
# folder in a result tree. See the module docstring for why the list is this
# short -- widening it means teaching the analysis scripts to open the archive.
COMPRESS_GLOBS = ("**/global_voltages/all_bus_voltages_phase_*.csv",)

ZIP_LEVEL = 6
REPORT_NAME = "compression_report.json"
CHUNK = 1 << 20  # 1 MiB streaming buffer -- never load a 13 MB CSV whole


def _sha256_stream(fh):
    h = hashlib.sha256()
    for chunk in iter(lambda: fh.read(CHUNK), b""):
        h.update(chunk)
    return h.hexdigest()


def _sha256_file(path):
    with open(path, "rb") as fh:
        return _sha256_stream(fh)


def compress_one(path_str, level=ZIP_LEVEL):
    """zip one CSV, verify the round-trip, then delete the original.

    Returns (status, before_bytes, after_bytes, message). status is one of
    "compressed", "skipped" (already done), "error". On "error" the original
    file is guaranteed to still be there.
    """
    src = Path(path_str)
    dst = src.with_name(src.name + ".zip")
    tmp = src.with_name(src.name + ".zip.tmp")

    if dst.exists() and not src.exists():
        return "skipped", dst.stat().st_size, dst.stat().st_size, "already compressed"
    if not src.exists():
        return "error", 0, 0, f"missing: {src}"

    before = src.stat().st_size
    try:
        digest_src = _sha256_file(src)

        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
            zf.write(src, arcname=src.name)

        # Read the archive back and hash the decompressed bytes. zf.testzip()
        # walks every member's CRC-32 first (catches a truncated/corrupt write);
        # the SHA-256 re-hash then catches anything a CRC-32 collision would
        # miss. The original is still on disk to fall back on through both.
        with zipfile.ZipFile(tmp, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                tmp.unlink(missing_ok=True)
                return "error", before, 0, f"verify failed (bad CRC in {bad}): {src}"
            with zf.open(src.name) as fo:
                digest_zip = _sha256_stream(fo)
        if digest_zip != digest_src:
            tmp.unlink(missing_ok=True)
            return "error", before, 0, f"verify failed (sha256 mismatch): {src}"

        after = tmp.stat().st_size
        os.replace(tmp, dst)   # atomic; dst may already exist from a half-done retry
        src.unlink()
        return "compressed", before, after, ""
    except Exception as exc:  # full disk, permissions, file locked by a viewer...
        tmp.unlink(missing_ok=True)
        return "error", before, 0, f"{type(exc).__name__}: {exc}"


def _read_any(path):
    """The plain CSV bytes behind a .csv, .csv.zip, .csv.gz or .abv path."""
    path = Path(path)
    if path.suffix == abv.SUFFIX:
        return abv.decode_to_csv_bytes(path.read_bytes())
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open(zf.namelist()[0]) as handle:
                return handle.read()
    if path.suffix == ".gz":  # legacy archives written before the .zip switch
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def repack_one(path_str):
    """Re-encode one all-bus voltage table into the dense .abv container.

    Accepts the .csv, the .csv.zip or a legacy .csv.gz -- all end up as
    <name>.abv, about 2.6x smaller than the zip and ~11x smaller than the CSV
    (see all_bus_voltage_format.py for the measured table and why).

    Same safety stance as compress_one: the container is written to a .tmp,
    decoded back, and its bytes compared against a SHA-256 of the original CSV
    *content* before anything is deleted. A file that cannot be reproduced
    exactly -- values not representable at the container's decimal scale, an
    int32 overflow, a short write -- leaves the source untouched and reports.
    """
    src = Path(path_str)
    stem = src.stem if src.suffix in (".zip", ".gz") else src.name
    dst = src.with_name(stem[:-4] + abv.SUFFIX if stem.endswith(".csv") else stem + abv.SUFFIX)
    tmp = dst.with_name(dst.name + ".tmp")

    if dst.exists() and not src.exists():
        return "skipped", dst.stat().st_size, dst.stat().st_size, "already repacked"
    if not src.exists():
        return "error", 0, 0, f"missing: {src}"

    before = src.stat().st_size
    try:
        original = _read_any(src)
        digest_src = hashlib.sha256(original).hexdigest()

        tmp.write_bytes(abv.encode_csv_bytes(original))
        restored = abv.decode_to_csv_bytes(tmp.read_bytes())
        if hashlib.sha256(restored).hexdigest() != digest_src:
            tmp.unlink(missing_ok=True)
            return "error", before, 0, f"verify failed (sha256 mismatch): {src}"

        after = tmp.stat().st_size
        os.replace(tmp, dst)
        src.unlink()
        return "compressed", before, after, ""
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return "error", before, 0, f"{type(exc).__name__}: {exc}"


def decompress_one(path_str):
    """Inverse of compress_one/repack_one: restore <name>.csv from
    <name>.csv.zip, a legacy <name>.csv.gz, or <name>.abv.

    The zip and gzip CRC-32 are both checked implicitly by reading the stream
    to the end (zipfile raises BadZipFile on a bad CRC the same way gzip
    raises on one), and the .abv container carries its own shape/scale
    header, so a corrupt archive raises before it is removed either way.
    """
    src = Path(path_str)                       # ...csv.zip, ...csv.gz or ...abv
    if src.suffix == abv.SUFFIX:
        dst = src.with_name(src.name[:-len(abv.SUFFIX)] + ".csv")
    else:
        dst = src.with_name(src.stem)          # ...csv
    tmp = src.with_name(dst.name + ".tmp")

    if dst.exists() and not src.exists():
        return "skipped", dst.stat().st_size, dst.stat().st_size, "already plain"
    if not src.exists():
        return "error", 0, 0, f"missing: {src}"

    before = src.stat().st_size
    try:
        if src.suffix == abv.SUFFIX:
            tmp.write_bytes(abv.decode_to_csv_bytes(src.read_bytes()))
        elif src.suffix == ".zip":
            with zipfile.ZipFile(src, "r") as zf:
                with zf.open(zf.namelist()[0]) as fi, open(tmp, "wb") as fo:
                    shutil.copyfileobj(fi, fo, CHUNK)
        else:  # legacy .gz
            with gzip.open(src, "rb") as fi, open(tmp, "wb") as fo:
                shutil.copyfileobj(fi, fo, CHUNK)
        after = tmp.stat().st_size
        os.replace(tmp, dst)
        src.unlink()
        return "compressed", before, after, ""
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return "error", before, 0, f"{type(exc).__name__}: {exc}"


def find_targets(root, decompress=False, mode="zip"):
    """Every file under `root` this script would act on, sorted for stable output.

    What counts as a target depends on the direction: compressing looks for
    plain .csv, repacking also picks up the .csv.zip (or a legacy .csv.gz)
    written by earlier runs (they are re-encoded, not left behind), and
    decompressing looks for any archive spelling.
    """
    root = Path(root)
    if decompress:
        patterns = [g + ext for g in COMPRESS_GLOBS for ext in (".zip", ".gz")]
        patterns += [g.replace(".csv", abv.SUFFIX) for g in COMPRESS_GLOBS]
    elif mode == "repack":
        # All three spellings: a tree may be half zipped (or, from before this
        # switch, half gzipped), and leaving those behind would defeat the
        # point of repacking it.
        patterns = (list(COMPRESS_GLOBS)
                    + [g + ext for g in COMPRESS_GLOBS for ext in (".zip", ".gz")])
    else:
        patterns = list(COMPRESS_GLOBS)
    found = []
    for pattern in patterns:
        found.extend(root.glob(pattern))
    return sorted(set(found))


def tree_mtime(root):
    """Newest mtime anywhere under root -- the staleness signal for --all."""
    newest = 0.0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(dirpath, name)))
            except OSError:
                pass
    return newest


def discover_roots(match=None, min_age_minutes=15, mode="zip"):
    """Eligible sim_results/<ts>/ trees for --all, newest-first by name.

    Skips trees with nothing to do and trees touched within min_age_minutes,
    so a run that is still writing can never be picked up. `mode` has to be
    passed through: an already-zipped tree has nothing to *compress* but
    plenty to *repack*, and judging that with the wrong mode silently hides
    every tree the caller actually asked about.
    """
    roots, skipped = [], []
    if not RESULTS_DIR.is_dir():
        return roots, skipped
    cutoff = time.time() - min_age_minutes * 60
    for entry in sorted(RESULTS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if match and match not in entry.name:
            continue
        if not find_targets(entry, mode=mode):
            continue
        if tree_mtime(entry) > cutoff:
            skipped.append(entry)
            continue
        roots.append(entry)
    return roots, skipped


def human(n_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024 or unit == "TB":
            return f"{n_bytes:.1f} {unit}" if unit != "B" else f"{n_bytes} B"
        n_bytes /= 1024.0


def process_tree(root, workers, level, decompress, dry_run, quiet=False, mode="zip"):
    """Compress (or decompress) one result tree. Returns a report dict."""
    root = Path(root)
    targets = find_targets(root, decompress=decompress, mode=mode)
    verb = "decompress" if decompress else ("repack" if mode == "repack" else "compress")

    if not targets:
        if not quiet:
            print(f"[{root.name}] nothing to {verb}")
        return {"root": str(root), "files": 0, "errors": [],
                "bytes_before": 0, "bytes_after": 0, "bytes_saved": 0}

    total_before = sum(p.stat().st_size for p in targets)
    if dry_run:
        # ~4.9x is the measured ratio on this data (see module docstring); only
        # used to preview the saving, never to decide anything.
        ratio = {"zip": 4.9, "repack": 2.6}[mode]
        est = total_before / ratio if not decompress else total_before * ratio
        print(f"[{root.name}] would {verb} {len(targets)} file(s), "
              f"{human(total_before)} -> ~{human(est)}")
        return {"root": str(root), "files": len(targets), "errors": [],
                "bytes_before": total_before, "bytes_after": int(est),
                "bytes_saved": int(total_before - est), "dry_run": True}

    if decompress:
        fn, args = decompress_one, [(str(p),) for p in targets]
    elif mode == "repack":
        fn, args = repack_one, [(str(p),) for p in targets]
    else:
        fn, args = compress_one, [(str(p), level) for p in targets]

    done = failed = 0
    before_sum = after_sum = 0
    errors = []
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, *a): a[0] for a in args}
        for fut in concurrent.futures.as_completed(futures):
            status, before, after, msg = fut.result()
            if status == "error":
                failed += 1
                errors.append(msg)
                print(f"  !! {msg}")
            else:
                done += 1
                before_sum += before
                after_sum += after
            if not quiet and (done + failed) % 100 == 0:
                print(f"  [{root.name}] {done + failed}/{len(targets)} ...")

    saved = before_sum - after_sum
    elapsed = time.perf_counter() - t0
    # Phrase the ratio in the direction that reads naturally for each mode:
    # compressing "saves" bytes, decompressing "restores" them.
    if decompress:
        ratio = (after_sum / before_sum) if before_sum else 0.0
        tail = f"({ratio:.1f}x expansion, {human(-saved)} restored)"
    else:
        ratio = (before_sum / after_sum) if after_sum else 0.0
        tail = f"({ratio:.1f}x, {human(saved)} saved)"
    print(f"[{root.name}] {verb}ed {done} file(s) in {elapsed:.0f}s: "
          f"{human(before_sum)} -> {human(after_sum)} {tail}"
          f"{f', {failed} FAILED' if failed else ''}")

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "action": verb,
        "zip_level": level,
        "patterns": list(COMPRESS_GLOBS),
        "files": done,
        "failed": failed,
        "errors": errors,
        "bytes_before": before_sum,
        "bytes_after": after_sum,
        "bytes_saved": saved,
        "seconds": round(elapsed, 1),
    }
    # Sidecar so the saving is auditable later, and so run_test_plan.py can
    # fold it into its progress file without parsing stdout.
    try:
        (root / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"  !! could not write {REPORT_NAME}: {exc}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Zip the write-only all-bus voltage CSVs in a finished "
                     "sim_results/ tree and delete the originals (verified "
                     "round-trip; reversible with --decompress).")
    parser.add_argument("roots", nargs="*",
                         help="sim_results/<timestamp> folder(s) to process. An explicitly "
                              "named root is trusted -- no age check -- so only pass one whose "
                              "run has finished.")
    parser.add_argument("--all", action="store_true",
                         help="Discover every eligible tree under sim_results/ instead of "
                              "naming them. Age-guarded (see --min-age-minutes).")
    parser.add_argument("--match", default=None,
                         help="With --all, only trees whose folder name contains this "
                              "substring (e.g. 'p3_' for Phase 3 only).")
    parser.add_argument("--min-age-minutes", type=int, default=15,
                         help="With --all, skip trees modified more recently than this, so a "
                              "still-running batch is never touched (default: 15).")
    parser.add_argument("--decompress", action="store_true",
                         help="Reverse: restore the .csv files from .csv.zip (or a legacy "
                              ".csv.gz) or .abv and delete the archives.")
    parser.add_argument("--repack", action="store_true",
                         help="Re-encode into the dense .abv container instead of zipping: "
                              "~2.6x smaller than the .zip and faster to read, still exactly "
                              "reversible. Picks up existing .csv.zip/.csv.gz as well as plain .csv.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would happen; change nothing.")
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1),
                         help="Parallel worker processes (default: min(6, cpu_count)). Lower it "
                              "when a simulation batch is running so the two don't fight for cores.")
    parser.add_argument("--level", type=int, default=ZIP_LEVEL, choices=range(1, 10),
                         metavar="1-9", help=f"zip DEFLATE compression level (default: {ZIP_LEVEL}).")
    args = parser.parse_args()

    if args.all:
        roots, skipped = discover_roots(args.match, args.min_age_minutes,
                                        mode="repack" if args.repack else "zip")
        for entry in skipped:
            print(f"[skip] {entry.name}: modified in the last {args.min_age_minutes} min "
                  f"-- looks like a run still in progress")
    else:
        roots = [Path(r) if Path(r).is_absolute() else SIM_DIR / r for r in args.roots]
        missing = [r for r in roots if not r.is_dir()]
        if missing:
            parser.error("not a directory: " + ", ".join(str(m) for m in missing))

    if not roots:
        print("Nothing to do." if args.all else "No roots given -- pass a folder or --all.")
        return 0

    grand_before = grand_after = 0
    failures = 0
    for root in roots:
        rep = process_tree(root, args.workers, args.level,
                            args.decompress, args.dry_run,
                            mode="repack" if args.repack else "zip")
        grand_before += rep["bytes_before"]
        grand_after += rep["bytes_after"]
        failures += rep.get("failed", 0)

    if len(roots) > 1:
        saved = grand_before - grand_after
        print(f"\nTOTAL across {len(roots)} tree(s): {human(grand_before)} -> "
              f"{human(grand_after)} ({human(saved)} saved)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
