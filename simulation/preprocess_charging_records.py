"""
preprocess_charging_records.py -- regenerate the EV charging-session profiles.

Rebuilds the per-session 15-minute profile CSVs under
filtered_apartments_2h/{night,morning,midday}/ from the raw dataset
filtered_apartments_2h/ChargingRecords.csv (Baek, Lee & Kim 2024,
https://doi.org/10.6084/m9.figshare.22495141). Replaces the untracked
legacy pipeline (unused/old_scripts/cvs_scripts/split_records.py +
add_24hour_entry.py + categorize_profiles.py) with one tracked script.

Model (thesis Ch. 4):
* The connection window is taken from the recorded timestamps
  (StartDay/StartTime -> EndDay/EndTime). The Duration column is NEVER read:
  it is corrupted for some rows (e.g. session 19758 records 14 min for a
  ~617-minute session).
* The vehicle charges at the nominal rate P_NOMINAL_KW from plug-in until the
  session's recorded energy (Demand, kWh) has been delivered, then remains
  plugged at P = 0 until departure. If the energy cannot be delivered at the
  nominal rate within the window, the whole window charges at the constant
  rate Demand/window instead (no idle tail).
* Each 15-minute slot k gets the slot-average power
  load_kw[k] = rate * overlap_minutes(k) / 15 (overlap rasterization), so the
  discretized profile conserves energy exactly: sum(load_kw) * 0.25 h == Demand.
  Grid-point-membership masks (the legacy approach) would silently drop
  charges shorter than one slot.
* plugged_in[k] = 1 iff the connection window overlaps slot k at all, which
  guarantees plugged_in >= (load_kw > 0) everywhere. Sessions spanning more
  than 24 h are plugged the entire (circular) day.
* Output format matches the legacy files: 97 rows, times 00:00..23:45 plus a
  24:00 row equal to the 00:00 row (so a full-day interpolation closes), with
  the new third column plugged_in.

Session set: the 9,510 apartment sessions (>= ~2 h) selected by the original
preprocessing. The filter is NOT re-derived -- it is not reproducible from
the current raw file -- so the ids live in the tracked manifest
apartment_2h_sessions.csv (session id N = raw row N-1, 0-based). If the
manifest is missing, it is bootstrapped by scanning the existing bucket
folders and written for future runs.

ChargerType check: the raw dataset labels each session's charger as slow (0)
or fast (1) (Baek et al. 2024, https://doi.org/10.1038/s41597-024-02942-9).
The P_NOMINAL_KW=7.2 model assumes Level-2/slow charging throughout, so every
session in the manifest is asserted to be ChargerType == 0 -- verified true
for all 9,510 sessions in the current manifest (the >= 2h duration filter
apparently already excludes fast-charge sessions on its own, since fast
charging rarely needs a 2h+ window to deliver typical apartment-charging
demand). If the manifest is ever regenerated with a different selection
rule, this assertion will catch any fast-charger session before it gets
silently modeled at the wrong rate.

Bucketing: each profile goes to the time window holding the largest share of
its (new) charging energy -- morning 06:00-12:59, midday 13:00-19:59, night
20:00-05:59 -- mirroring the legacy categorize_profiles.py. With the
front-loaded charging model this effectively follows the plug-in time.
"""

import os
import re
import glob

import numpy as np
import pandas as pd

P_NOMINAL_KW = 7.2       # nominal charging power (thesis Ch. 4)
SLOT_MIN = 15.0
N_SLOTS = 96
DAY_MIN = 1440.0
BUCKETS = ("morning", "midday", "night")  # legacy dict order (tie-break)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "filtered_apartments_2h")
RAW_CSV = os.path.join(DATA_DIR, "ChargingRecords.csv")
MANIFEST = os.path.join(SCRIPT_DIR, "apartment_2h_sessions.csv")

TIME_LABELS = [f"{(15 * k) // 60:02d}:{(15 * k) % 60:02d}" for k in range(N_SLOTS)]
SLOT_HOURS = np.array([(15 * k) // 60 for k in range(N_SLOTS)])
WINDOW_OF_HOUR = np.where(
    (SLOT_HOURS >= 6) & (SLOT_HOURS < 13), 0,          # morning
    np.where((SLOT_HOURS >= 13) & (SLOT_HOURS < 20), 1,  # midday
             2))                                          # night


def load_session_ids():
    """The 9,510 session ids, from the manifest (bootstrapped if missing)."""
    if os.path.exists(MANIFEST):
        ids = pd.read_csv(MANIFEST)["session_id"].astype(int).tolist()
        return sorted(ids)
    pattern = re.compile(r"ev_charging_profile_session_(\d+)\.csv$")
    ids = set()
    for bucket in BUCKETS:
        for path in glob.glob(os.path.join(DATA_DIR, bucket, "ev_charging_profile_session_*.csv")):
            ids.add(int(pattern.search(os.path.basename(path)).group(1)))
    if len(ids) != 9510:
        raise RuntimeError(
            f"Expected 9510 sessions in the bucket folders, found {len(ids)}; "
            "cannot bootstrap the manifest from an incomplete state.")
    ids = sorted(ids)
    pd.DataFrame({"session_id": ids}).to_csv(MANIFEST, index=False)
    print(f"Bootstrapped manifest {MANIFEST} ({len(ids)} sessions)")
    return ids


def minutes_of_day(hms):
    """'H:MM:SS' (possibly without leading zero) -> minutes since midnight."""
    h, m, s = (int(x) for x in str(hms).split(":"))
    return h * 60.0 + m + s / 60.0


def connection_window(row):
    """(start_min within day, duration_min) from timestamps; Duration unused."""
    start = minutes_of_day(row.StartTime)
    end = minutes_of_day(row.EndTime)
    day_offset = (pd.Timestamp(row.EndDay) - pd.Timestamp(row.StartDay)).days
    dur = day_offset * DAY_MIN + end - start
    if dur <= 0:
        raise ValueError(f"Non-positive connection window ({dur} min)")
    return start, dur


def rasterize(start_min, length_min):
    """Overlap minutes of the circular-day interval [start, start+length) with
    each 15-min slot. length >= 1440 covers the whole day."""
    overlap = np.zeros(N_SLOTS)
    if length_min >= DAY_MIN:
        overlap[:] = SLOT_MIN
        return overlap
    lo = np.arange(N_SLOTS) * SLOT_MIN
    hi = lo + SLOT_MIN
    a = start_min % DAY_MIN
    segments = ([(a, a + length_min)] if a + length_min <= DAY_MIN
                else [(a, DAY_MIN), (0.0, a + length_min - DAY_MIN)])
    for s, e in segments:
        overlap += np.clip(np.minimum(e, hi) - np.maximum(s, lo), 0.0, None)
    return overlap


def build_profile(start_min, dur_min, demand_kwh):
    """(load_kw[96], plugged_in[96]) for one session."""
    t_charge_min = demand_kwh / P_NOMINAL_KW * 60.0
    if t_charge_min <= dur_min:
        rate, charge_len = P_NOMINAL_KW, t_charge_min
    else:  # cannot deliver at nominal rate: constant rate, no idle tail
        rate, charge_len = demand_kwh / (dur_min / 60.0), dur_min
    load_kw = rate * rasterize(start_min, charge_len) / SLOT_MIN
    plugged = (rasterize(start_min, dur_min) > 0.0).astype(int)

    energy = load_kw.sum() * (SLOT_MIN / 60.0)
    if not np.isclose(energy, demand_kwh, rtol=1e-9, atol=1e-12):
        raise AssertionError(f"Energy not conserved: {energy} != {demand_kwh}")
    if np.any((load_kw > 0.0) & (plugged == 0)):
        raise AssertionError("Charging outside the plugged window")
    return load_kw, plugged, rate != P_NOMINAL_KW


def categorize(load_kw):
    """Bucket = window with the largest charging energy (legacy tie-break:
    morning, then midday, then night)."""
    sums = [load_kw[WINDOW_OF_HOUR == w].sum() for w in range(3)]
    return BUCKETS[int(np.argmax(sums))]


def main():
    session_ids = load_session_ids()
    raw = pd.read_csv(RAW_CSV)

    profiles = {}       # session_id -> (bucket, DataFrame)
    counts = {b: 0 for b in BUCKETS}
    n_fallback = n_wrap = n_multiday = 0
    total_energy = total_demand = 0.0

    for sid in session_ids:
        row = raw.iloc[sid - 1]  # session id N = raw row N-1
        if row.Location != "apartment":
            raise AssertionError(f"Session {sid} is not an apartment row")
        if not row.Demand > 0:
            raise AssertionError(f"Session {sid} has non-positive Demand")
        if int(row.ChargerType) != 0:
            raise AssertionError(
                f"Session {sid} is a fast charger (ChargerType={row.ChargerType}); "
                f"the {P_NOMINAL_KW} kW model assumes Level-2/slow charging only")
        start, dur = connection_window(row)
        load_kw, plugged, fallback = build_profile(start, dur, float(row.Demand))

        n_fallback += fallback
        n_wrap += (start + min(dur, DAY_MIN) > DAY_MIN)
        n_multiday += (dur >= DAY_MIN)
        total_energy += load_kw.sum() * (SLOT_MIN / 60.0)
        total_demand += float(row.Demand)

        bucket = categorize(load_kw)
        counts[bucket] += 1
        df = pd.DataFrame({
            "time": TIME_LABELS + ["24:00"],
            "load_kw": np.append(load_kw, load_kw[0]),       # 24:00 = 00:00 row
            "plugged_in": np.append(plugged, plugged[0]),
        })
        if df.isna().any().any():
            raise AssertionError(f"NaN in profile of session {sid}")
        profiles[sid] = (bucket, df)

    # Write phase: only after every session validated.
    #
    # The bucket folders are created rather than assumed. This script used to
    # only ever run against a checkout that already had them, but the desktop
    # app's handoff bundle ships the raw ChargingRecords.csv *instead* of the
    # 9,510 profiles (43 MB and 9,510 files saved) and rebuilds them on first
    # launch, so here they legitimately do not exist yet.
    for bucket in BUCKETS:
        os.makedirs(os.path.join(DATA_DIR, bucket), exist_ok=True)
        for path in glob.glob(os.path.join(DATA_DIR, bucket, "ev_charging_profile_session_*.csv")):
            os.remove(path)
    for sid, (bucket, df) in profiles.items():
        df.to_csv(os.path.join(DATA_DIR, bucket, f"ev_charging_profile_session_{sid}.csv"),
                  index=False)

    print(f"Wrote {len(profiles)} profiles: "
          + ", ".join(f"{b} {counts[b]}" for b in BUCKETS))
    print(f"Fallback-rate sessions (Demand undeliverable at {P_NOMINAL_KW} kW): {n_fallback}")
    print(f"Midnight-wrap sessions: {n_wrap}; sessions spanning >= 24 h: {n_multiday}")
    print(f"Energy check: sum(profiles) = {total_energy:.6f} kWh vs "
          f"sum(Demand) = {total_demand:.6f} kWh")


if __name__ == "__main__":
    main()
