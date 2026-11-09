"""
run_test_plan.py -- resumable, partially-parallel executor for ../thesis_test_plan.md.

Runs every task the plan calls for (gain tuning, single-day comparisons,
full-year robustness batches, post-processing) in the plan's priority order.
Progress is persisted to test_plan_progress.json after every task, so killing
this script (Ctrl+C, closed terminal, reboot, crash) and re-running it later
resumes exactly where it left off: a task is skipped only if its status is
"done" -- "pending"/"blocked"/"failed", and any "running" marker left behind
by a killed process, are all retried automatically. Gains files already on
disk (e.g. from manual runs before this script existed) are detected and
counted as done too.

Scope: the full 3x3 grid -- 3 locations (oslo_no1, athens, berlin_50hertz) x 3
penetration scenarios (base, medium, high) -- in every phase, 162 tasks total:

    Phase 1  18 tasks   9 combos x 2 optimizers (PID, droop)
    Phase 2  63 tasks   9 combos x (4 controllers + 1 compare + 1 curtail
                                    + 1 compress)
    Phase 3  81 tasks   9 combos x (4 controllers + 2 compares + 2 curtails
                                    + 1 compress, the pairs being max/min load)

Every run is followed by BOTH analysis steps, so no result folder is left
half-analysed: compare_control_strategies.py writes strategy_summary.csv + PNGs,
and compute_curtailment.py writes curtailment_summary.csv. The latter used to be
a manual pass that was easy to forget (the first 15 result sets had to be
backfilled by hand); it is a tracked task now.

Once BOTH have succeeded, a third step (compress_results.py) zips the raw
tree's write-only all-bus voltage CSVs and deletes the originals -- see the
"Disk footprint" section below.

Phase 3 formerly covered only the 3 base-scenario combinations, with an opt-in
--high-location flag for a single extra stress-test site. It now covers all nine
like the other phases, so that flag is gone; the medium/high yearly runs are the
bulk of the remaining work (each ~4h of wall clock per location/scenario).

TEST_PLAN_EXECUTION.md is regenerated alongside test_plan_progress.json after
every task: it explains the plan in prose (phases, scenarios, what data each
step collects) plus an auto-updated results index -- for each location x
scenario x controller, the exact sim_results/<timestamp>/ folder and
analysis_output/ folder the run and its compare step produced. Only the
section between the AUTO-GENERATED markers in that file is rewritten; edit
anything above/below those markers freely.

Usage (from simulation/, or via run_test_plan.bat):
    python run_test_plan.py                       # run/resume the plan
    python run_test_plan.py --status               # progress + time estimate only
    python run_test_plan.py --dry-run              # print the full task list
    python run_test_plan.py --workers 4            # cap Phase 1/2 concurrency (default 6)

-------------------------------------------------------------------------------
Parallelism: WHAT runs concurrently, and why it's safe
-------------------------------------------------------------------------------
sim_config.py's EV_CONTRIBUTION/PV_CONTRIBUTION are plain module constants
with no CLI override (every optimizer/yearly_tests.py subprocess reads them
at import time), so this script still rewrites those two lines in
sim_config.py immediately before each *cohort* of subprocess calls that needs
a specific scenario -- exactly as before, just once per cohort instead of
once per task. A cohort is every not-yet-done task for one (phase, scenario)
combination; this script always fully drains one cohort (waits for every
subprocess in it to finish) before touching sim_config.py again, so no two
concurrently-running subprocesses ever see different EV/PV values. Tasks
*within* a cohort differ only by --location/--controls/--timestamp -- all
real CLI flags, not shared mutable state -- so running several of them at
once is safe.

Given that, what actually runs in parallel (bounded by --workers, default 6):
  - Phase 1 (tune_pid/tune_droop): each optimizer subprocess is single-process
    (no internal multiprocessing -- differential_evolution here doesn't pass
    workers=), so up to --workers of them run at once across locations.
  - Phase 2 (day_run, one extreme day per load envelope): also single-process
    per controller subprocess, same treatment.
  - Phase 3 (yearly_run, --yearly): NEVER run concurrently with a sibling --
    always forced to 1 regardless of --workers. yearly_tests.py's --yearly
    path already spawns a ProcessPoolExecutor internally and saturates every
    core across the 53 ISO weeks (thesis_test_plan.md's own risk note: "don't
    run more than one --yearly batch at a time... they'll contend for cores
    and both slow down"). This is the one phase where "if possible" cashes
    out to "not possible" -- and the code says so via a printed note rather
    than silently ignoring --workers there.
  - compare (compare_control_strategies.py) and curtail
    (compute_curtailment.py): both read finished CSVs only and never touch
    sim_config.py's EV/PV, so they run in the same parallel pool as their
    phase. They sit in separate cohorts from each other (p2cmp_/p2curt_) for
    one reason: both write RESULTS_BASIS.md into the same output folder, so
    running them concurrently would be a write race on that one file.
    Neither depends on the other -- a failed compare does not block the
    curtailment metric, or vice versa.
  - compress (compress_results.py): its own cohort (p2zip_/p3zip_), last for
    the scenario, because unlike the two above it *deletes* source files and
    so must not overlap with anything still reading them. Each subprocess uses
    COMPRESS_WORKERS processes internally, sized so a full cohort stays well
    inside the core count.

-------------------------------------------------------------------------------
Disk footprint
-------------------------------------------------------------------------------
A Phase 3 result tree is ~16 GB and the nine of them would be ~145 GB, which
does not fit on this machine alongside everything else. ~98% of that bulk is
the three global_voltages/all_bus_voltages_phase_{A,B,C}.csv files per week
folder -- the full per-bus voltage trajectory, which network_setup.py writes
as raw archival evidence and which NO analysis script in this repo reads back
(compare_control_strategies.py uses simulation_results_phase_*.csv and
voltage_quality.csv; compute_curtailment.py uses inverter_profiles/).

So each (location, scenario) gets a tracked `compress` task that runs
compress_results.py over its sim_results/<timestamp>/ tree once compare AND
curtail have both succeeded: it zips those files (~4.9x) and deletes the
originals, taking a tree from ~16 GB to ~3.5 GB. Every other file stays plain
CSV, so no analysis script had to change.

Gating it on both analysis steps is the point -- "after the data is processed,
compress and delete the original". If either step failed, the compress task is
blocked and the raw CSVs stay untouched for a retry. The step is also
idempotent (re-running finds nothing to do) and reversible
(`python compress_results.py <root> --decompress` restores byte-identical
CSVs, verified by SHA-256 on the way in).

Phase 1/2 subprocesses are launched with OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS
pinned to 1 (parallel_env()) so N concurrent single-process jobs don't each
try to grab every BLAS thread and starve each other. yearly_run is exempted
(it's always alone, and wants its ProcessPoolExecutor to have the whole
machine).

Result-folder collision fix: network_setup.py's on-disk path for a run's
output (sim_results/<timestamp>/sim_results_combined/<Control>/
pv_<PV>_ev_<EV>_files_<max|min>/week_<N>) has NO location component --
only the outer timestamp folder separates one run from another. The
sequential version of this script never had two runs in flight at once, so a
same-minute timestamp collision was academic. Running several locations
concurrently makes it real, and it's not just cross-location: Phase 2 and
Phase 3 for the very same (location, scenario) would also collide, and
Phase 3's week_N for the argmax week would then silently overwrite Phase 2's
single-day result for that same week. resolve_timestamp() below fixes this
by suffixing every Phase 2/3 timestamp with its task's ts_key (e.g.
"p2_athens_base" / "p3_athens_base" -- unique per phase+location+scenario by
construction), guaranteeing distinct folders regardless of timing, and
persisting the chosen name so a resumed run's still-pending controllers land
in the same folder as their already-finished siblings.

-------------------------------------------------------------------------------
Control-type tracking
-------------------------------------------------------------------------------
Phase 2 and Phase 3 used to run all four controllers in one combined
`--controls 0 1 2 3` subprocess call, so their timing/status was tracked as
one blob per (location, scenario). Real per-controller timing varies hugely
(a Berlin full-year run measured 2026-07-20: PID 3026s, Droop 3111s,
Centralized 4957s, NoControl 1687s -- Centralized is ~3x NoControl), and
"which controller is it on right now" is exactly what you want to see in
--status during a long batch. So each controller now runs (and is timed and
tracked) as its own subprocess/task, e.g. p3_year_athens_base_Centralized.

-------------------------------------------------------------------------------
Fixes made while rewriting this for parallelism
-------------------------------------------------------------------------------
1. SCENARIOS previously read base=(0.40,0.05)/low=(0.20,0.02)/high=(0.60,0.15)
   -- none of which match ../thesis_test_plan.md's documented table (Base
   0.20/0.05, Medium 0.40/0.15, High 0.60/0.30). Fixed to match the doc
   exactly (and renamed "low" -> "medium" to match). reconcile_preexisting()
   now also *un-marks* a Phase 1 task "done" if its gains file no longer
   matches the (corrected) scenario tag on disk, so stale "done" rows left
   over from the old mapping self-heal into "pending" on the next run rather
   than silently letting Phase 2/3 run against the wrong (or default,
   untuned) gains.
2. The compare step's --root was missing the "/sim_results_combined" path
   segment that network_setup.py actually writes results under (a bug
   flagged in this file's own comments, never hit because no compare task
   had completed yet) -- fixed, and Phase 3 compare now correctly passes
   --scenario per load-extreme (max/min), since a yearly run's controllers
   always produce both scenario folders and compare_control_strategies.py
   can't auto-disambiguate two folders in one --root.
3. Subprocesses are launched with stdin=DEVNULL. yearly_tests.py prompts
   "[y/N] run the optimizer now?" if gains are missing at run time; under
   piped/background stdin it already falls back to defaults instead of
   hanging (see yearly_tests.py::_prompt_yes), but multiple concurrent
   subprocesses inheriting one shared console stdin could otherwise race for
   keystrokes. Dependencies already guarantee gains exist before Phase 2/3
   run, so this is a belt-and-suspenders fix, not a load-bearing one.
"""
import argparse
import heapq
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
SIM_CONFIG_PATH = SIM_DIR / "sim_config.py"
PROGRESS_PATH = SIM_DIR / "test_plan_progress.json"
LOG_PATH = SIM_DIR / "test_plan_run.log"
MANIFEST_PATH = SIM_DIR / "TEST_PLAN_EXECUTION.md"
MANIFEST_BEGIN = "<!-- BEGIN AUTO-GENERATED RESULTS INDEX (updated by run_test_plan.py -- do not edit below by hand) -->"
MANIFEST_END = "<!-- END AUTO-GENERATED RESULTS INDEX -->"

PYTHON = sys.executable

LOCATIONS = ["oslo_no1", "athens", "berlin_50hertz"]
SCENARIOS = {
    "base":   (0.20, 0.05),  # thesis_test_plan.md #3 -- lower bound
    "medium": (0.40, 0.15),  # mid-range, matches the sim_config.py default EV share at higher PV
    "high":   (0.60, 0.30),  # stress case
}
SCENARIO_ORDER = ["base", "medium", "high"]
NETWORK_SEED = 42

# Time-resolution multiplier (sim_config.STEP_MULT: 1 = 15-min steps, 15 = 1-min
# steps). Pinned explicitly here (not left to sim_config.py's own default) so
# Phase 1/2's resolution can't silently drift -- Phase 1 gains must be tuned
# at the same step Phase 2 runs at (thesis Ch. 3). Phase 3 uses the same 1-min
# step as Phase 1/2 (thesis_test_plan.md's runtime-cutting suggestion of
# dropping yearly runs to the coarser 15-min step was not taken).
STEP_SINGLE_DAY = 15
STEP_YEARLY = 15

CONTROL_IDS = ["0", "1", "2", "3"]
CONTROL_LABELS = {"0": "PID", "1": "Droop", "2": "Centralized", "3": "NoControl"}
LOAD_EXTREMES = ("max", "min")  # yearly_run covers both in one batch; day_run needs one per extreme

# Matches a full Phase-1 scenario cohort for the plan's 3 locations (3 x 2
# optimizers = 6 independent single-process tasks) -- override with --workers
# on a smaller/larger machine. Ignored (forced to 1) for Phase 3 -- see the
# module docstring.
DEFAULT_WORKERS = 6

# Worker processes *inside* one compress_results.py subprocess. A p2zip_/p3zip_
# cohort holds one task per location (3), and those 3 run concurrently in the
# --workers pool, so the real process count is 3 x this. Kept at 3 (=> 9) so
# compression stays comfortably inside the core count even if a cohort and
# something else overlap; zipping is the bottleneck, not the disk.
COMPRESS_WORKERS = 3

# Which archive format the compress task writes. "zip" leaves each table
# as a .csv.zip any tool anywhere can open, no extra software needed; "dense"
# re-encodes it into the .abv container (all_bus_voltage_format.py), ~2.6x
# smaller again and quicker to read, but only decodable next to this
# checkout. Portability wins by default for something kept as thesis
# evidence; --compress-format overrides it. Both are exactly reversible via
# --decompress.
COMPRESS_FORMAT = "zip"

# Seed timing estimates in seconds, used only until real observations exist
# (see estimate_seconds). day_run seeds are measured on this machine
# 2026-07-18, oslo_no1 single day/1-min step (No Control / Centralized only;
# PID/Droop interpolated). yearly_run seeds are measured 2026-07-20,
# berlin_50hertz full year at STEP=15 -- the same step Phase 3 now runs at --
# both self-correct once real same-(type, controller) runs complete (see
# estimate_seconds).
SEED_SECONDS = {
    ("tune_pid", None): 1200.0,
    ("tune_droop", None): 1200.0,
    ("day_run", "PID"): 250.0,
    ("day_run", "Droop"): 250.0,
    ("day_run", "Centralized"): 374.0,
    ("day_run", "NoControl"): 132.0,
    ("yearly_run", "PID"): 3026.0,
    ("yearly_run", "Droop"): 3111.0,
    ("yearly_run", "Centralized"): 4957.0,
    ("yearly_run", "NoControl"): 1687.0,
    ("compare", None): 15.0,
    # curtail measured 2026-07-26, athens: 0.6s for a single day, 5.5s for a
    # 53-week batch (one load extreme). Seeded to the slower case.
    ("curtail", None): 8.0,
    # compress measured 2026-07-27 on this machine at COMPRESS_WORKERS=3, while
    # a yearly batch was running: 324s for a full Phase 3 tree (1272 files /
    # 15.3 GB -> 2.9 GB), 3.5s for a Phase 2 tree (12 files). Seeded to the
    # Phase 3 case; self-corrects once real runs land.
    ("compress", None): 324.0,
}

STATE_LOCK = threading.RLock()  # guards progress dict + JSON/manifest writes + log file writes


def scenario_tag(ev, pv, seed=NETWORK_SEED):
    return f"ev{round(ev * 100)}_pv{round(pv * 100)}_seed{seed}"


def gains_path(kind, location, scenario):
    """Path to a tuned-gains CSV if it exists on disk, else None.

    Resolved by globbing locations/*/<location>/gains/ rather than looking up
    the location's country folder, so this needs no import of sim_config.
    """
    ev, pv = SCENARIOS[scenario]
    tag = scenario_tag(ev, pv)
    matches = list((SIM_DIR / "locations").glob(f"*/{location}/gains/{kind}_{tag}.csv"))
    return matches[0].relative_to(SIM_DIR).as_posix() if matches else None


def gains_exist(kind, location, scenario):
    return gains_path(kind, location, scenario) is not None


def set_scenario(ev, pv):
    """Rewrite sim_config.py's EV_CONTRIBUTION/PV_CONTRIBUTION lines in place.

    Only these two numeric literals are touched; every other line (including
    any other uncommitted edits already in the file) is left untouched.
    Called once per cohort (never while a cohort's subprocesses are still
    running) -- see the module docstring's parallelism section.
    """
    import re
    text = SIM_CONFIG_PATH.read_text(encoding="utf-8")
    new_text, n_ev = re.subn(r"(?m)^EV_CONTRIBUTION = [0-9.]+", f"EV_CONTRIBUTION = {ev}", text)
    new_text, n_pv = re.subn(r"(?m)^PV_CONTRIBUTION = [0-9.]+", f"PV_CONTRIBUTION = {pv}", new_text)
    if n_ev != 1 or n_pv != 1:
        raise RuntimeError(
            f"Expected exactly one EV_CONTRIBUTION and one PV_CONTRIBUTION line in "
            f"{SIM_CONFIG_PATH}, found {n_ev}/{n_pv}. Refusing to edit -- the file may "
            f"have been restructured; update set_scenario()'s regex to match.")
    if new_text != text:
        SIM_CONFIG_PATH.write_text(new_text, encoding="utf-8")


def parallel_env():
    """Env for a Phase 1/2 subprocess run alongside siblings in the same
    cohort: caps the common BLAS/OpenMP thread pools at 1 so N concurrent
    single-process optimizer/day-run subprocesses don't oversubscribe the
    machine fighting each other for the same matrix-math threads. Not used
    for yearly_run -- that task type always runs alone (forced serial) and
    already parallelizes internally via its own ProcessPoolExecutor, so it
    should keep unrestricted threading."""
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"
    return env


# ---------------------------------------------------------------------------
# Task list
# ---------------------------------------------------------------------------
def build_tasks():
    tasks = []

    # ---- Phase 1: tune gains (no compare stage) ----
    for scen in SCENARIO_ORDER:
        for loc in LOCATIONS:
            tasks.append(dict(id=f"p1_pid_{loc}_{scen}", phase=1, type="tune_pid", kind="run",
                               location=loc, scenario=scen, control=None, depends=[],
                               group=f"p1_{scen}",
                               desc=f"Phase 1: tune PID gains -- {loc} / {scen}"))
            tasks.append(dict(id=f"p1_droop_{loc}_{scen}", phase=1, type="tune_droop", kind="run",
                               location=loc, scenario=scen, control=None, depends=[],
                               group=f"p1_{scen}",
                               desc=f"Phase 1: tune droop gains -- {loc} / {scen}"))

    # ---- Phase 2: single extreme day per load envelope, one task per controller ----
    # The two envelopes are two different days of the year -- the peak-load day and
    # the lowest-load day -- so each needs its own run, unlike Phase 3 where one
    # yearly batch produces both. The "max" tasks keep their original unsuffixed ids
    # and result folders: they were run before the "min" envelope was added, and
    # renaming them would orphan the recorded progress and the analysis outputs the
    # thesis already cites.
    for scen in SCENARIO_ORDER:
        for loc in LOCATIONS:
            for extreme in LOAD_EXTREMES:
                sfx = "" if extreme == "max" else f"_{extreme}"
                ts_key = f"p2{sfx}_{loc}_{scen}"
                day = "worst-case" if extreme == "max" else "lightest-load"
                run_ids = []
                for c in CONTROL_IDS:
                    label = CONTROL_LABELS[c]
                    tid = f"p2_day_{loc}_{scen}_{label}{sfx}"
                    run_ids.append(tid)
                    tasks.append(dict(id=tid, phase=2, type="day_run", kind="run",
                                       location=loc, scenario=scen, control=c, extreme=extreme,
                                       depends=[f"p1_pid_{loc}_{scen}", f"p1_droop_{loc}_{scen}"],
                                       group=f"p2run{sfx}_{scen}", ts_key=ts_key,
                                       desc=f"Phase 2: {day} day ({extreme} load), {label} -- {loc} / {scen}"))
                tasks.append(dict(id=f"p2_cmp_{loc}_{scen}{sfx}", phase=2, type="compare", kind="cmp",
                                   location=loc, scenario=scen, control=None, depends=run_ids,
                                   group=f"p2cmp{sfx}_{scen}", ts_key=ts_key,
                                   out_dir=f"phase2_{loc}_{scen}{sfx}", extreme=extreme,
                                   desc=f"Phase 2: summarize day-run ({extreme} load) -- {loc} / {scen}"))
                # Curtailment depends on the runs, not on the compare task -- it is an
                # independent post-processing pass over the same raw CSVs, so a failed
                # compare should not block it. It gets its own cohort purely so the two
                # never run concurrently: both write RESULTS_BASIS.md into out_dir.
                tasks.append(dict(id=f"p2_curt_{loc}_{scen}{sfx}", phase=2, type="curtail", kind="cmp",
                                   location=loc, scenario=scen, control=None, depends=list(run_ids),
                                   group=f"p2curt{sfx}_{scen}", ts_key=ts_key,
                                   out_dir=f"phase2_{loc}_{scen}{sfx}", extreme=extreme,
                                   desc=f"Phase 2: curtailment metric ({extreme} load) -- {loc} / {scen}"))
                # Shrink the raw tree only once BOTH analysis steps have succeeded --
                # this one deletes source CSVs, so unlike compare/curtail it depends
                # on them, not just on the runs. See "Disk footprint" in the docstring.
                tasks.append(dict(id=f"p2_zip_{loc}_{scen}{sfx}", phase=2, type="compress", kind="cmp",
                                   location=loc, scenario=scen, control=None,
                                   depends=run_ids + [f"p2_cmp_{loc}_{scen}{sfx}",
                                                      f"p2_curt_{loc}_{scen}{sfx}"],
                                   group=f"p2zip{sfx}_{scen}", ts_key=ts_key,
                                   desc=f"Phase 2: compress raw tree ({extreme} load) -- {loc} / {scen}"))

    # ---- Phase 3: full 53-week yearly, one task per controller, forced serial ----
    def add_phase3(loc, scen):
        ts_key = f"p3_{loc}_{scen}"
        run_ids = []
        for c in CONTROL_IDS:
            label = CONTROL_LABELS[c]
            tid = f"p3_year_{loc}_{scen}_{label}"
            run_ids.append(tid)
            tasks.append(dict(id=tid, phase=3, type="yearly_run", kind="run",
                               location=loc, scenario=scen, control=c,
                               depends=[f"p1_pid_{loc}_{scen}", f"p1_droop_{loc}_{scen}"],
                               group=f"p3run_{scen}", ts_key=ts_key,
                               desc=f"Phase 3: full 53-week yearly, {label} -- {loc} / {scen}"))
        for extreme in LOAD_EXTREMES:
            tasks.append(dict(id=f"p3_cmp_{loc}_{scen}_{extreme}", phase=3, type="compare", kind="cmp",
                               location=loc, scenario=scen, control=None, depends=list(run_ids),
                               group=f"p3cmp_{scen}", ts_key=ts_key,
                               out_dir=f"phase3_{loc}_{scen}_{extreme}", extreme=extreme,
                               desc=f"Phase 3: summarize yearly run ({extreme} load) -- {loc} / {scen}"))
        for extreme in LOAD_EXTREMES:
            tasks.append(dict(id=f"p3_curt_{loc}_{scen}_{extreme}", phase=3, type="curtail", kind="cmp",
                               location=loc, scenario=scen, control=None, depends=list(run_ids),
                               group=f"p3curt_{scen}", ts_key=ts_key,
                               out_dir=f"phase3_{loc}_{scen}_{extreme}", extreme=extreme,
                               desc=f"Phase 3: curtailment metric ({extreme} load) -- {loc} / {scen}"))
        # One compress per (location, scenario): the tree holds both load
        # extremes, so it waits on all four runs and on both extremes' compare
        # and curtail. This is where the ~12.5 GB/tree saving comes from.
        tasks.append(dict(id=f"p3_zip_{loc}_{scen}", phase=3, type="compress", kind="cmp",
                           location=loc, scenario=scen, control=None,
                           depends=run_ids
                                   + [f"p3_cmp_{loc}_{scen}_{e}" for e in LOAD_EXTREMES]
                                   + [f"p3_curt_{loc}_{scen}_{e}" for e in LOAD_EXTREMES],
                           group=f"p3zip_{scen}", ts_key=ts_key,
                           desc=f"Phase 3: compress raw tree -- {loc} / {scen}"))

    # Full 3x3: every location at every scenario. Scenario-major ordering means
    # base finishes everywhere before medium starts anywhere, so a run cut short
    # still leaves a complete result set at the lower penetrations rather than
    # one location done at all three and the others untouched.
    for scen in SCENARIO_ORDER:
        for loc in LOCATIONS:
            add_phase3(loc, scen)

    return tasks


def group_tasks(tasks):
    """Group the flat task list (in build_tasks' emission order) into
    (phase, scenario, stage) cohorts, preserving the order each group first
    appears in. Each cohort is run to completion -- in parallel where safe --
    before the next one starts; see the module docstring."""
    groups, order = {}, []
    for t in tasks:
        g = t["group"]
        if g not in groups:
            groups[g] = []
            order.append(g)
        groups[g].append(t)
    return [(g, groups[g]) for g in order]


# ---------------------------------------------------------------------------
# Progress persistence
# ---------------------------------------------------------------------------
def load_progress():
    if PROGRESS_PATH.exists():
        data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    else:
        data = {"created": datetime.now().isoformat(timespec="seconds")}
    data.setdefault("tasks", {})
    data.setdefault("result_timestamps", {})
    # "high_location" appears in progress files written before Phase 3 covered
    # all nine location/scenario combinations. It is no longer read; left in
    # place rather than deleted so an older file round-trips unchanged.
    return data


def save_progress(progress):
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def persist(tasks, progress):
    """Save progress.json and refresh the results-index table in the .md
    manifest together, so the two never drift out of sync. Safe to call from
    any worker thread (STATE_LOCK is an RLock, so a caller that already holds
    it -- e.g. execute_and_record -- can call this directly)."""
    with STATE_LOCK:
        save_progress(progress)
        update_manifest(tasks, progress)


def resolve_timestamp(task, progress):
    """Shared sim_results/<timestamp> folder for every controller in a Phase
    2/3 (location, scenario) group -- assigned once, on first use, and
    persisted so a resumed run's still-pending controllers land in the same
    folder as their already-finished siblings. See the module docstring for
    why the location+scenario suffix (not just a bare timestamp) is required
    once locations run concurrently."""
    key = task["ts_key"]
    with STATE_LOCK:
        ts = progress["result_timestamps"].get(key)
        if ts is None:
            # key itself (e.g. "p2_oslo_no1_base" / "p3_oslo_no1_base") is
            # unique per (phase, location, scenario) by construction -- using
            # it directly (not just location+scenario) is what keeps Phase 2
            # and Phase 3 runs for the same location+scenario from folding
            # into the same folder (Phase 3's week_N for the argmax week
            # would silently overwrite Phase 2's single-day result otherwise).
            ts = f"{time.strftime('%Y-%m-%d_%H.%M')}_{key}"
            progress["result_timestamps"][key] = ts
            save_progress(progress)
    return ts


# ---------------------------------------------------------------------------
# Results-index manifest (TEST_PLAN_EXECUTION.md)
# ---------------------------------------------------------------------------
STATUS_MARK = {"done": "done", "running": "running", "blocked": "blocked",
               "failed": "**FAILED**", "pending": "pending"}


def _cell(value):
    """Markdown table cell: code-span a path, em-dash a missing value."""
    if not value:
        return "—"
    return f"`{value}`"


def _phase1_table(progress):
    lines = ["| Location | Scenario | PID gains file | Droop gains file |",
             "|---|---|---|---|"]
    for scen in SCENARIO_ORDER:
        for loc in LOCATIONS:
            pid_rec = progress["tasks"].get(f"p1_pid_{loc}_{scen}", {})
            droop_rec = progress["tasks"].get(f"p1_droop_{loc}_{scen}", {})
            pid_cell = _cell(gains_path("pid", loc, scen))
            droop_cell = _cell(gains_path("droop", loc, scen))
            if pid_rec.get("status") != "done":
                pid_cell += f" ({STATUS_MARK.get(pid_rec.get('status', 'pending'), 'pending')})"
            if droop_rec.get("status") != "done":
                droop_cell += f" ({STATUS_MARK.get(droop_rec.get('status', 'pending'), 'pending')})"
            lines.append(f"| {loc} | {scen} | {pid_cell} | {droop_cell} |")
    return "\n".join(lines)


def _phase2_table(progress):
    lines = ["| Location | Scenario | Load | Control | Status | Duration | sim_results/ folder |",
             "|---|---|---|---|---|---|---|"]
    for scen in SCENARIO_ORDER:
        for loc in LOCATIONS:
            for extreme in LOAD_EXTREMES:
                sfx = "" if extreme == "max" else f"_{extreme}"
                ts = progress["result_timestamps"].get(f"p2{sfx}_{loc}_{scen}")
                folder_cell = _cell(f"sim_results/{ts}" if ts else None)
                for label in CONTROL_LABELS.values():
                    rec = progress["tasks"].get(f"p2_day_{loc}_{scen}_{label}{sfx}", {})
                    status = STATUS_MARK.get(rec.get("status", "pending"), "pending")
                    secs = rec.get("seconds")
                    duration = format_hms(secs) if secs else "—"
                    lines.append(f"| {loc} | {scen} | {extreme} | {label} | {status} | "
                                  f"{duration} | {folder_cell} |")
    lines += ["", "Phase 2 post-processing (per location/scenario/load-extreme) -- `compare` writes "
                  "`strategy_summary.csv` + PNGs, `curtail` writes `curtailment_summary.csv`, "
                  "`compress` zips the raw tree's all-bus voltage CSVs:", "",
              "| Location | Scenario | Load | compare | curtail | compress | Saved | analysis_output/ folder |",
              "|---|---|---|---|---|---|---|---|"]
    for scen in SCENARIO_ORDER:
        for loc in LOCATIONS:
            for extreme in LOAD_EXTREMES:
                sfx = "" if extreme == "max" else f"_{extreme}"
                cmp_rec = progress["tasks"].get(f"p2_cmp_{loc}_{scen}{sfx}", {})
                curt_rec = progress["tasks"].get(f"p2_curt_{loc}_{scen}{sfx}", {})
                zip_rec = progress["tasks"].get(f"p2_zip_{loc}_{scen}{sfx}", {})
                cmp_status = STATUS_MARK.get(cmp_rec.get("status", "pending"), "pending")
                curt_status = STATUS_MARK.get(curt_rec.get("status", "pending"), "pending")
                zip_status = STATUS_MARK.get(zip_rec.get("status", "pending"), "pending")
                done = cmp_rec.get("status") == "done" or curt_rec.get("status") == "done"
                cell = _cell(f"analysis_output/phase2_{loc}_{scen}{sfx}") if done else "—"
                lines.append(f"| {loc} | {scen} | {extreme} | {cmp_status} | {curt_status} | "
                              f"{zip_status} | {format_bytes(zip_rec.get('bytes_saved'))} | {cell} |")
    return "\n".join(lines)


def _phase3_table(progress):
    rows = [(loc, scen) for scen in SCENARIO_ORDER for loc in LOCATIONS]

    lines = ["| Location | Scenario | Control | Status | Duration | sim_results/ folder |",
             "|---|---|---|---|---|---|"]
    for loc, scen in rows:
        ts = progress["result_timestamps"].get(f"p3_{loc}_{scen}")
        folder_cell = _cell(f"sim_results/{ts}" if ts else None)
        for label in CONTROL_LABELS.values():
            rec = progress["tasks"].get(f"p3_year_{loc}_{scen}_{label}", {})
            status = STATUS_MARK.get(rec.get("status", "pending"), "pending")
            secs = rec.get("seconds")
            duration = format_hms(secs) if secs else "—"
            lines.append(f"| {loc} | {scen} | {label} | {status} | {duration} | {folder_cell} |")

    lines += ["", "Phase 3 post-processing (per location/scenario/load-extreme) -- `compare` writes "
                  "`strategy_summary.csv` + PNGs, `curtail` writes `curtailment_summary.csv`:", "",
              "| Location | Scenario | Extreme | compare | curtail | analysis_output/ folder |",
              "|---|---|---|---|---|---|"]
    for loc, scen in rows:
        for extreme in LOAD_EXTREMES:
            cmp_rec = progress["tasks"].get(f"p3_cmp_{loc}_{scen}_{extreme}", {})
            curt_rec = progress["tasks"].get(f"p3_curt_{loc}_{scen}_{extreme}", {})
            cmp_status = STATUS_MARK.get(cmp_rec.get("status", "pending"), "pending")
            curt_status = STATUS_MARK.get(curt_rec.get("status", "pending"), "pending")
            out_dir = f"phase3_{loc}_{scen}_{extreme}"
            done = cmp_rec.get("status") == "done" or curt_rec.get("status") == "done"
            cell = _cell(f"analysis_output/{out_dir}") if done else "—"
            lines.append(f"| {loc} | {scen} | {extreme} | {cmp_status} | {curt_status} | {cell} |")

    # Compression is per (location, scenario) -- one raw tree covers both load
    # extremes -- so it gets its own table rather than a column that would
    # repeat itself down the max/min pairs above.
    lines += ["", "Phase 3 raw-tree compression (per location/scenario, runs once "
                  "`compare` and `curtail` have both succeeded for both extremes):", "",
              "| Location | Scenario | compress | Saved | Files | sim_results/ folder |",
              "|---|---|---|---|---|---|"]
    for loc, scen in rows:
        rec = progress["tasks"].get(f"p3_zip_{loc}_{scen}", {})
        status = STATUS_MARK.get(rec.get("status", "pending"), "pending")
        ts = progress["result_timestamps"].get(f"p3_{loc}_{scen}")
        n_files = rec.get("files_compressed")
        lines.append(f"| {loc} | {scen} | {status} | {format_bytes(rec.get('bytes_saved'))} | "
                      f"{n_files if n_files else '—'} | {_cell(f'sim_results/{ts}' if ts else None)} |")
    if not rows:
        return "_No Phase 3 runs scheduled yet._"
    return "\n".join(lines)


def render_manifest_tables(tasks, progress):
    counts = {}
    for task in tasks:
        st = progress["tasks"].get(task["id"], {}).get("status", "pending")
        counts[st] = counts.get(st, 0) + 1
    remaining = total_estimate(tasks, progress)

    parts = [
        f"_Last updated {datetime.now():%Y-%m-%d %H:%M} -- "
        f"{counts.get('done', 0)}/{len(tasks)} tasks done, "
        f"~{format_hms(remaining)} of sequential-equivalent work remaining "
        f"(parallel cohorts finish faster than this sum)._",
        "",
        "### Phase 1 -- tuned gains",
        _phase1_table(progress),
        "",
        "### Phase 2 -- one extreme day per load envelope, per controller",
        _phase2_table(progress),
        "",
        "### Phase 3 -- full 53-week yearly, per controller",
        _phase3_table(progress),
    ]
    return "\n".join(parts)


def update_manifest(tasks, progress):
    body = f"{MANIFEST_BEGIN}\n\n{render_manifest_tables(tasks, progress)}\n\n{MANIFEST_END}\n"
    if MANIFEST_PATH.exists():
        text = MANIFEST_PATH.read_text(encoding="utf-8")
        if MANIFEST_BEGIN in text and MANIFEST_END in text:
            pre = text.split(MANIFEST_BEGIN)[0]
            post = text.split(MANIFEST_END)[1]
            new_text = pre + body + post
        else:
            new_text = text.rstrip("\n") + "\n\n" + body
    else:
        new_text = "# Test plan execution log\n\n" + body
    MANIFEST_PATH.write_text(new_text, encoding="utf-8")


def sync_task_metadata(tasks, progress):
    for task in tasks:
        rec = progress["tasks"].setdefault(task["id"], {"status": "pending"})
        rec["type"] = task["type"]
        rec["phase"] = task["phase"]
        rec["location"] = task["location"]
        rec["scenario"] = task["scenario"]
        rec["control_label"] = CONTROL_LABELS.get(task["control"]) if task["control"] else None
        rec["desc"] = task["desc"]
        if "out_dir" in task:
            rec["out_dir"] = task["out_dir"]
        if "extreme" in task:
            rec["extreme"] = task["extreme"]


def reconcile_preexisting(tasks, progress):
    """Keep Phase 1 status in sync with what's actually on disk, in both
    directions: mark a task done if a matching gains file exists (tuned
    manually, or by an older run of this script), and un-mark it done if no
    matching file exists (e.g. stale "done" rows left over from before
    SCENARIOS' EV/PV values were corrected to match thesis_test_plan.md --
    see the module docstring). "Done" should always mean "the expected
    output is actually on disk," not "some process once said so.\""""
    changed = False
    for task in tasks:
        if task["type"] not in ("tune_pid", "tune_droop"):
            continue
        rec = progress["tasks"][task["id"]]
        kind = "pid" if task["type"] == "tune_pid" else "droop"
        exists = gains_exist(kind, task["location"], task["scenario"])
        if exists and rec["status"] != "done":
            rec["status"] = "done"
            rec["note"] = "pre-existing gains file detected on disk"
            rec.pop("blocked_on", None)
            changed = True
        elif not exists and rec["status"] == "done":
            rec["status"] = "pending"
            rec["note"] = "gains file no longer found on disk for this scenario tag -- will re-tune"
            rec.pop("seconds", None)
            changed = True
    return changed


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_cmd(args, task_id, log_fh, env=None):
    banner = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] START {task_id}: {' '.join(args)}"
    print(f">>> [{task_id}] {' '.join(args)}")
    with STATE_LOCK:
        log_fh.write(banner + "\n")
        log_fh.flush()
    t0 = time.perf_counter()
    proc = subprocess.run(args, cwd=SIM_DIR, env=env, stdin=subprocess.DEVNULL)
    elapsed = time.perf_counter() - t0
    with STATE_LOCK:
        log_fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] END   {task_id}: "
                      f"exit={proc.returncode} elapsed={elapsed:.1f}s\n")
        log_fh.flush()
    return proc.returncode, elapsed


def execute_task(task, progress, log_fh):
    t = task["type"]
    loc, scen = task["location"], task["scenario"]
    env = None if t == "yearly_run" else parallel_env()

    if t == "tune_pid":
        rc, secs = run_cmd([PYTHON, "pid_optimizer_approx.py", "--location", loc,
                             "--step", str(STEP_SINGLE_DAY)], task["id"], log_fh, env)
        return rc, secs, {}

    if t == "tune_droop":
        rc, secs = run_cmd([PYTHON, "droop_optimizer_approx.py", "--location", loc,
                             "--step", str(STEP_SINGLE_DAY)], task["id"], log_fh, env)
        return rc, secs, {}

    if t == "day_run":
        ts = resolve_timestamp(task, progress)
        rc, secs = run_cmd([PYTHON, "yearly_tests.py", "--controls", task["control"],
                             "--location", loc, "--step", str(STEP_SINGLE_DAY),
                             "--day-extreme", task.get("extreme", "max"),
                             "--timestamp", ts], task["id"], log_fh, env)
        return rc, secs, {"result_folder": ts}

    if t == "yearly_run":
        ts = resolve_timestamp(task, progress)
        rc, secs = run_cmd([PYTHON, "yearly_tests.py", "--yearly", "--controls", task["control"],
                             "--location", loc, "--step", str(STEP_YEARLY),
                             "--timestamp", ts], task["id"], log_fh, env)
        return rc, secs, {"result_folder": ts}

    if t == "compare":
        ts = progress["result_timestamps"].get(task["ts_key"])
        if not ts:
            print(f"!! {task['id']}: no result folder recorded for {task['ts_key']}; skipping compare.")
            return 1, 0.0, {}
        ev, pv = SCENARIOS[scen]
        scen_folder = f"pv_{round(pv * 100)}_ev_{round(ev * 100)}_files_{task['extreme']}"
        out_dir = f"analysis_output/{task['out_dir']}"
        root = f"sim_results/{ts}/sim_results_combined"
        rc, secs = run_cmd([PYTHON, "compare_control_strategies.py",
                             "--root", root, "--scenario", scen_folder,
                             "--output-dir", out_dir], task["id"], log_fh, env)
        return rc, secs, {}

    if t == "curtail":
        ts = progress["result_timestamps"].get(task["ts_key"])
        if not ts:
            print(f"!! {task['id']}: no result folder recorded for {task['ts_key']}; skipping curtailment.")
            return 1, 0.0, {}
        ev, pv = SCENARIOS[scen]
        scen_folder = f"pv_{round(pv * 100)}_ev_{round(ev * 100)}_files_{task['extreme']}"
        out_dir = f"analysis_output/{task['out_dir']}"
        root = f"sim_results/{ts}/sim_results_combined"
        # --location is load-bearing here: without it PV curtailment is silently
        # reported as 0 (the PV schedule can only be reconstructed from that
        # location's cached PVGIS table).
        rc, secs = run_cmd([PYTHON, "compute_curtailment.py",
                             "--root", root, "--location", loc,
                             "--scenario", scen_folder,
                             "--output-dir", out_dir], task["id"], log_fh, env)
        return rc, secs, {}

    if t == "compress":
        ts = progress["result_timestamps"].get(task["ts_key"])
        if not ts:
            print(f"!! {task['id']}: no result folder recorded for {task['ts_key']}; skipping compress.")
            return 1, 0.0, {}
        root = SIM_DIR / "sim_results" / ts
        if not root.is_dir():
            print(f"!! {task['id']}: {root} does not exist; nothing to compress.")
            return 1, 0.0, {}
        # The folder is named explicitly (not --all), so compress_results.py
        # skips its "is this run still being written?" age guard -- correct
        # here, because this task's dependencies already prove it finished.
        compress_cmd = [PYTHON, "compress_results.py", str(root),
                        "--workers", str(COMPRESS_WORKERS)]
        if COMPRESS_FORMAT == "dense":
            compress_cmd.append("--repack")
        rc, secs = run_cmd(compress_cmd, task["id"], log_fh, env)
        extra = {}
        report = root / "compression_report.json"
        if rc == 0 and report.exists():
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
                extra["files_compressed"] = data.get("files")
                extra["bytes_saved"] = data.get("bytes_saved")
            except (OSError, ValueError):
                pass  # the saving is cosmetic; never fail the task over it
        return rc, secs, extra

    raise ValueError(f"unknown task type: {t}")


def dep_blocking(task, progress):
    for dep_id in task["depends"]:
        status = progress["tasks"].get(dep_id, {}).get("status")
        if status != "done":
            return dep_id, status
    return None, None


def execute_and_record(task, tasks, progress, log_fh):
    tid = task["id"]
    with STATE_LOCK:
        rec = progress["tasks"][tid]
        rec["status"] = "running"
        rec["started"] = datetime.now().isoformat(timespec="seconds")
        rec.pop("blocked_on", None)
        persist(tasks, progress)
    print(f"-> {task['desc']}")

    rc, secs, extra = execute_task(task, progress, log_fh)

    with STATE_LOCK:
        rec.update(extra)
        rec["finished"] = datetime.now().isoformat(timespec="seconds")
        rec["seconds"] = secs
        if rc == 0:
            rec["status"] = "done"
            rec.pop("last_error", None)
            print(f"-- done: {tid} ({format_hms(secs)})")
        else:
            rec["status"] = "failed"
            rec["last_error"] = f"exit code {rc}"
            print(f"!! {tid} FAILED (exit {rc}) after {format_hms(secs)} -- "
                  f"continuing with remaining independent tasks.")
        persist(tasks, progress)


def run_cohort(group_name, group_list, tasks, progress, log_fh, max_workers):
    eligible = []
    for task in group_list:
        rec = progress["tasks"][task["id"]]
        if rec["status"] == "done":
            continue
        blocked_on, blocked_status = dep_blocking(task, progress)
        if blocked_on:
            with STATE_LOCK:
                if rec["status"] != "blocked" or rec.get("blocked_on") != blocked_on:
                    rec["status"] = "blocked"
                    rec["blocked_on"] = blocked_on
                    persist(tasks, progress)
            print(f"[skip] {task['id']}: blocked on {blocked_on} (status={blocked_status})")
            continue
        eligible.append(task)

    if not eligible:
        return

    forced_serial = group_name.startswith("p3run_")
    workers = 1 if forced_serial else max(1, min(max_workers, len(eligible)))
    note = (" (forced serial -- yearly_tests.py --yearly already uses every "
            "core internally; see module docstring)" if forced_serial else "")
    print(f"\n{'=' * 70}\ncohort {group_name}: {len(eligible)} task(s), {workers} worker(s){note}\n{'=' * 70}")

    if workers == 1:
        for task in eligible:
            execute_and_record(task, tasks, progress, log_fh)
    else:
        # Note on Ctrl+C here: interrupting the main thread while it's
        # waiting in as_completed() below will propagate up through this
        # function once every currently-running subprocess.run() in the pool
        # returns (ThreadPoolExecutor.__exit__ joins its threads before
        # letting the exception through). On Windows, Ctrl+C is delivered to
        # every child process sharing this console too, so those subprocess
        # calls typically exit on their own within moments -- this isn't an
        # immediate abort, but it isn't a hang either. Any task still marked
        # "running" when the process dies is retried automatically on the
        # next invocation (only "done" is ever skipped).
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(execute_and_record, task, tasks, progress, log_fh) for task in eligible]
            for fut in as_completed(futures):
                fut.result()


def run_plan(tasks, progress, log_fh, max_workers):
    for group_name, group_list in group_tasks(tasks):
        if group_list[0]["kind"] == "run":
            set_scenario(*SCENARIOS[group_list[0]["scenario"]])
        run_cohort(group_name, group_list, tasks, progress, log_fh, max_workers)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_hms(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def format_bytes(n_bytes):
    """Human-readable byte count for the manifest's "Saved" columns; em-dash
    when a compress task hasn't reported one yet."""
    if not n_bytes:
        return "—"
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0


def estimate_seconds(progress, type_, control_label):
    samples = [r["seconds"] for r in progress["tasks"].values()
               if r.get("type") == type_ and r.get("control_label") == control_label
               and r.get("status") == "done" and r.get("seconds")]
    if samples:
        return sum(samples) / len(samples)
    return SEED_SECONDS.get((type_, control_label), 60.0)


def total_estimate(tasks, progress):
    return sum(estimate_seconds(progress, task["type"], CONTROL_LABELS.get(task["control"]))
               for task in tasks
               if progress["tasks"].get(task["id"], {}).get("status") != "done")


# ---------------------------------------------------------------------------
# Wall-clock estimation
#
# total_estimate() above sums per-task effort, which over-states the time left
# whenever a cohort runs several tasks concurrently. The three helpers below
# model the schedule this script actually executes -- sequential cohorts, each
# internally parallel up to --workers, except p3run_ cohorts which are forced
# serial -- to give a real finish time. Previously a separate read-only
# companion script (test_plan_status.py); folded in here so the task/cohort/
# timing definitions have exactly one owner.
# ---------------------------------------------------------------------------
def remaining_seconds(rec, task, progress, now):
    """Estimated seconds still needed for one not-done task. A task already
    running is credited for the time it has spent so far."""
    est = estimate_seconds(progress, task["type"], CONTROL_LABELS.get(task["control"]))
    if rec.get("status") == "running" and rec.get("started"):
        elapsed = (now - datetime.fromisoformat(rec["started"])).total_seconds()
        return max(est - elapsed, 0.0)
    return est


def cohort_wall_seconds(durations, workers):
    """Approximate makespan of scheduling `durations` onto `workers` identical
    lanes (longest-processing-time list scheduling), matching how run_cohort
    dispatches a cohort's tasks across its ThreadPoolExecutor."""
    if not durations:
        return 0.0
    lanes = [0.0] * min(workers, len(durations))
    heapq.heapify(lanes)
    for d in sorted(durations, reverse=True):
        least = heapq.heappop(lanes)
        heapq.heappush(lanes, least + d)
    return max(lanes)


def estimate_wall_clock(tasks, progress, workers, now):
    """Sum of each cohort's makespan, in the order run_plan() visits cohorts
    (cohorts never overlap -- the next starts only once the previous is done)."""
    total = 0.0
    for group_name, group_list in group_tasks(tasks):
        durations = []
        for task in group_list:
            rec = progress["tasks"].get(task["id"], {"status": "pending"})
            if rec.get("status") == "done":
                continue
            durations.append(remaining_seconds(rec, task, progress, now))
        cohort_workers = 1 if group_name.startswith("p3run_") else workers
        total += cohort_wall_seconds(durations, cohort_workers)
    return total


def print_status(tasks, progress, workers=DEFAULT_WORKERS):
    counts = {}
    for task in tasks:
        st = progress["tasks"].get(task["id"], {}).get("status", "pending")
        counts[st] = counts.get(st, 0) + 1
    print(f"Tasks: {len(tasks)} total | done={counts.get('done', 0)} "
          f"pending={counts.get('pending', 0)} blocked={counts.get('blocked', 0)} "
          f"running={counts.get('running', 0)} failed={counts.get('failed', 0)}")
    print(f"Remaining sequential-equivalent effort: ~{format_hms(total_estimate(tasks, progress))} "
          f"(parallel Phase 1/2 cohorts finish faster than this sum; Phase 3 does not)")

    now = datetime.now()
    wall_remaining = estimate_wall_clock(tasks, progress, workers, now)
    if wall_remaining > 0:
        print(f"Estimated wall-clock time left (--workers {workers}, Phase 3 forced serial): "
              f"~{format_hms(wall_remaining)}")
        print(f"Projected finish: {now + timedelta(seconds=wall_remaining):%Y-%m-%d %H:%M}")

    running = [(t, progress["tasks"][t["id"]]) for t in tasks
               if progress["tasks"].get(t["id"], {}).get("status") == "running"
               and progress["tasks"][t["id"]].get("started")]
    if running:
        print("\nCurrently running:")
        for task, rec in running:
            elapsed = (now - datetime.fromisoformat(rec["started"])).total_seconds()
            est = estimate_seconds(progress, task["type"], CONTROL_LABELS.get(task["control"]))
            print(f"  {task['id']:<32s} elapsed {format_hms(elapsed)} / est {format_hms(est)} "
                  f"(~{format_hms(max(est - elapsed, 0.0))} left)")

    by_phase = {}
    for task in tasks:
        by_phase.setdefault(task["phase"], []).append(task)
    marker = {"done": "[x]", "running": "[~]", "blocked": "[!]", "failed": "[X]", "pending": "[ ]"}
    for phase in sorted(by_phase):
        print(f"\nPhase {phase}:")
        for task in by_phase[phase]:
            rec = progress["tasks"].get(task["id"], {})
            st = rec.get("status", "pending")
            secs = rec.get("seconds")
            extra = f"  ({format_hms(secs)})" if secs else ""
            print(f"  {marker.get(st, '[ ]')} {task['id']:<30s} {task['desc']}{extra}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Resumable, partially-parallel executor for ../thesis_test_plan.md. "
                     "Re-running this script (with no flags) resumes from wherever it last stopped.")
    parser.add_argument("--status", action="store_true",
                         help="Print progress, a wall-clock estimate with projected finish time, and any "
                              "currently-running tasks, then exit without running anything. Reads the "
                              "progress file only -- safe to run while a batch is in flight. The estimate "
                              "assumes the --workers value given here, so pass the same one the batch uses.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the full ordered task list with current status, then exit.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help="Max concurrent subprocesses for Phase 1 (gain tuning) and Phase 2 "
                              "(single-day comparison) cohorts -- both are single-process tasks, "
                              "so several locations/controllers can run at once safely. Phase 3 "
                              "(--yearly) always forces this to 1: it already saturates every core "
                              f"internally, so a second concurrent batch would only contend for "
                              f"cores and slow both down (default: {DEFAULT_WORKERS}).")
    parser.add_argument("--yes", action="store_true",
                         help="Skip the start confirmation prompt (for unattended/scheduled runs).")
    parser.add_argument("--compress-format", choices=("zip", "dense"), default=COMPRESS_FORMAT,
                         help="Archive format for the compress tasks. zip (default) writes "
                              ".csv.zip, readable by any tool anywhere with no extra software. "
                              "dense writes the .abv container: ~2.6x smaller again and quicker "
                              "to read, but only decodable next to this checkout. Both are "
                              "exactly reversible.")
    args = parser.parse_args()
    globals()["COMPRESS_FORMAT"] = args.compress_format
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    progress = load_progress()
    tasks = build_tasks()
    sync_task_metadata(tasks, progress)
    reconcile_preexisting(tasks, progress)

    # --status/--dry-run are strictly READ-ONLY: they reconcile in memory, print,
    # and exit without persisting. This matters because the usual reason to run
    # --status is that a batch is already in flight -- and that batch owns the
    # progress file, rewriting it after every task. A second process persisting
    # its own snapshot would race the runner and could clobber a just-finished
    # task's record (and needlessly rewrite TEST_PLAN_EXECUTION.md mid-run).
    if args.dry_run or args.status:
        print_status(tasks, progress, args.workers)
        return

    persist(tasks, progress)
    print_status(tasks, progress, args.workers)
    print(f"\nParallel workers for Phase 1/2 cohorts: {args.workers} "
          f"(Phase 3 always runs one controller at a time -- see --help).")
    if not args.yes:
        try:
            input("\nPress Enter to start/resume, Ctrl+C to cancel... ")
        except KeyboardInterrupt:
            print("\nCancelled -- no tasks were run.")
            return

    run_start = time.perf_counter()
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as log_fh:
            log_fh.write(f"\n\n===== run started {datetime.now().isoformat(timespec='seconds')} "
                          f"(workers={args.workers}) =====\n")
            run_plan(tasks, progress, log_fh, args.workers)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress was saved after each completed task -- re-run this "
              "script (or run_test_plan.bat) to resume. Tasks still 'running' when this exited "
              "are retried automatically (only 'done' tasks are ever skipped).")
    finally:
        # Always leave sim_config.py on the documented base scenario so other
        # tools reading it directly see the canonical default, not whatever
        # scenario the last cohort happened to run under.
        set_scenario(*SCENARIOS["base"])
        print(f"\nSession wall-clock: {format_hms(time.perf_counter() - run_start)}")

    print()
    print_status(tasks, progress)


if __name__ == "__main__":
    main()
