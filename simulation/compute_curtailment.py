"""
compute_curtailment.py -- post-processing metric: per-inverter, per-timestep
ACTIVE-POWER CURTAILMENT (kW), the charging/PV power an inverter gave up so it
could stay within its current rating while supplying reactive power to fight
over/under-voltage.

    curtailment[inverter, t] = max(|P_scheduled| - |P_actual|, 0)   [kW]

P_actual is the post-curtailment power the run already logged
(inverter_profiles/global_p_phase_{A,B,C}.csv, kW). P_scheduled is the
pre-curtailment schedule, which is NOT saved -- but it is exactly
reconstructable, because the run saves which charging session each EV inverter
drew from (ev_charging_session_ids.csv) and the session curves are fixed input
data. We replay the same interpolation network_setup.build_day_inputs() uses:

    scheduled_ev_kw = -interp(session['load_kw'] onto the run's step grid)

This matches No Control's logged P bit-for-bit (No Control never curtails, so
its actual P *is* the schedule) -- verified to 0.0000 kW, which is also the
built-in correctness check this script runs when a NoControl folder is present.

PV inverters (present in global_p but absent from ev_charging_session_ids.csv)
are curtailed far more rarely; their schedule is reconstructed from the
location's cached PV table (sim_config.pv_source_path) when --location is
given, otherwise PV columns are reported as 0 curtailment with a note.

Outputs, per week folder:
  inverter_profiles/active_power_curtailment_phase_{A,B,C}.csv  (inverter x step, kW)
and, per strategy/scenario, printed to the console / optional --summary-csv:
  total_scheduled_kwh -- energy the inverters were meant to provide,
  total_curtailed_kwh -- energy they gave up (not provided), and
  curtailed_pct       -- 100 * curtailed / scheduled, the share of the intended
                         charging/PV energy sacrificed to voltage support.

Usage:
    python compute_curtailment.py --root sim_results/<ts>/sim_results_combined
    python compute_curtailment.py --root <...> --location berlin_50hertz   # also do PV

    # one load extreme into a specific folder -- how run_test_plan.py's curtail
    # step invokes it, alongside the matching compare step
    python compute_curtailment.py --root <...> --location athens \
        --scenario pv_5_ev_20_files_max --output-dir analysis_output/phase3_athens_base_max

run_test_plan.py runs this automatically as a `curtail` task after each phase's
compare task, one per (location, scenario[, load extreme]) -- so every result
folder gets a curtailment_summary.csv without a manual pass.
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
EV_DATA_DIR = THIS_DIR / "filtered_apartments_2h"

STRATEGY_DIRS = ["NoControl", "PIDcontrol", "Droopcontrol", "CentralizedControl"]
WEEK_RE = re.compile(r"week_(\d+)$")

# Dropped next to the summary CSVs so anyone reading them knows the time basis.
RESULTS_BASIS_NOTE = """\
# Results basis: 53 representative DAYS, not a 365-day year

Every yearly run here simulates ONE representative day per ISO week (see the
`n_weeks` column, = 53). So all values are aggregated over 53 days, not a full
calendar year. Scale ONLY the extensive totals below when presenting an annual
figure -- the intensive metrics are already period-independent.

Annual scale factor: ~7 (one day stands for its 7-day week; 365/53 = 6.9).

## Extensive TOTALS -- represent 53 days; multiply by ~7 for an annual estimate
  strategy_summary.csv : total_active_losses_kwh, total_excursion_puh,
                         total_reactive_effort_kvar, total_q_variation_kvar,
                         estimated_additional_inverter_losses_kwh, sign_flips_total
  curtailment_summary.csv : total_scheduled_kwh, total_curtailed_kwh

## Intensive metrics -- period-independent; do NOT scale
  percentages : voltage_violation_pct, curtailed_pct, transformer_*_pct
  peaks/maxes : peak_instantaneous_loss_kw, peak_instantaneous_q_kvar,
                vuf_max_pct, transformer_loading_max_pct, transformer_peak_s_kva
  means/rates : vuf_mean_pct, transformer_loading_mean_pct, mean_seconds_per_step,
                sign_flips_mean_per_inverter_per_day, rms_voltage_deviation_pu
  ratios      : reactive_fairness_gini, reactive_fairness_cv

## _max vs _min folders
Two SEPARATE representative-day sets (each week's peak-load day vs its min-load
day) -- bounding cases, do not add them together.
"""
# Scenario folder name yearly_tests.py writes, e.g. pv_15_ev_40_files_max.
SCENARIO_RE = re.compile(r"pv_(\d+)_ev_(\d+)_files_(max|min)$")


def analysis_dir_for(scenario_name: str, location: str, analysis_root: Path) -> Path | None:
    """Map a run scenario folder (pv_<PV>_ev_<EV>_files_<max|min>) to the
    matching analysis_output/<location>_ev<EV>_pv<PV>_<max|min>/ folder, so the
    curtailment summary lands next to that scenario's strategy_summary.csv.

    Note this is the older per-scenario naming (still used by the legacy
    berlin_50hertz_ev40_pv15_* result folders). run_test_plan.py's compare step
    instead names its output phase<N>_<location>_<scenario>[_<extreme>], so
    pass --analysis-root and relocate the file when pairing with those."""
    m = SCENARIO_RE.match(scenario_name)
    if not m:
        return None
    pv, ev, extreme = m.group(1), m.group(2), m.group(3)
    return analysis_root / f"{location}_ev{ev}_pv{pv}_{extreme}"


def reconstruct_scheduled_ev(week_dir: Path, new_steps: int) -> dict[str, np.ndarray]:
    """{'inverter_<id>': scheduled P (kW, neg=charging)} for every EV inverter,
    rebuilt from ev_charging_session_ids.csv + its session curves, interpolated
    onto the run's `new_steps` grid exactly as build_day_inputs() does."""
    ids_path = week_dir / "ev_charging_session_ids.csv"
    if not ids_path.exists():
        return {}
    try:
        ids = pd.read_csv(ids_path)
    except pd.errors.EmptyDataError:
        # A run with no EV chargers at all (a PV-only scenario). Older runs
        # wrote a header-less empty file here; newer ones write a header with
        # no rows. Both mean the same thing: nothing to reconstruct.
        return {}
    scheduled = {}
    for _, r in ids.iterrows():
        session_path = EV_DATA_DIR / r["bucket"] / r["session_file"]
        if not session_path.exists():
            raise FileNotFoundError(
                f"EV session file for inverter {r['inverter_id']} not found: "
                f"{session_path}. The metric reconstructs the schedule from the "
                f"original filtered_apartments_2h/ data; it must be present.")
        load_kw = np.nan_to_num(pd.read_csv(session_path)["load_kw"].values, nan=0.0)
        sched = -1.0 * np.interp(np.linspace(0, 1, new_steps),
                                 np.linspace(0, 1, len(load_kw)), load_kw)
        scheduled[f"inverter_{int(r['inverter_id'])}"] = sched
    return scheduled


def reconstruct_scheduled_pv(week_index: int, new_steps: int, pv_file: Path) -> np.ndarray:
    """Scheduled PV generation (kW, pos) on the run's step grid, from the
    location's cached weekly PV table -- the same curve build_day_inputs()
    injects for every PV inverter that week (they all share one profile)."""
    pv_master = pd.read_csv(pv_file, index_col=0)
    pv_kw_raw = np.nan_to_num(pv_master[f"week_{week_index + 1}"].values / 1e3, nan=0.0)  # W->kW
    return np.interp(np.linspace(0, 1, new_steps),
                     np.linspace(0, 1, len(pv_kw_raw)), pv_kw_raw)


def compute_week(week_dir: Path, pv_file: Path | None, validate: bool) -> float:
    """Write active_power_curtailment_phase_*.csv for one strategy-week and
    return the total curtailed energy (kWh)."""
    inv_dir = week_dir / "inverter_profiles"
    m = WEEK_RE.search(week_dir.name)
    week_index = int(m.group(1)) if m else 0

    actual = {}
    for ph in ("A", "B", "C"):
        p = inv_dir / f"global_p_phase_{ph}.csv"
        if p.exists():
            actual[ph] = pd.read_csv(p, index_col=0)
    if not actual:
        return 0.0
    new_steps = len(next(iter(actual.values())))
    index = next(iter(actual.values())).index

    scheduled_ev = reconstruct_scheduled_ev(week_dir, new_steps)
    if validate and scheduled_ev:  # NoControl: schedule must equal logged actual
        for ph, df in actual.items():
            for col in df.columns:
                if col in scheduled_ev:
                    d = float(np.abs(scheduled_ev[col] - df[col].values).max())
                    assert d < 1e-6, f"reconstruction mismatch {col} ({d} kW) -- schedule replay is wrong"

    # dt in hours for the kWh integral -- the MEDIAN spacing of the actual
    # timestamps, exactly compare_control_strategies._dt_hours()'s convention, so
    # curtailed energy is on the same scale as that module's loss kWh (and robust
    # to the last row wrapping 24:00 -> 00:00, which the median ignores). Equals
    # 0.25/step_mult h (1 min at step_mult=15, 15 min at step_mult=1).
    _diffs = pd.to_timedelta(pd.Index(index).astype(str)).to_series().diff().dropna()
    step_h = float(_diffs.median().total_seconds() / 3600.0) if len(_diffs) else 0.0
    curtailed_kwh = 0.0   # energy the inverters gave up (not provided)
    scheduled_kwh = 0.0   # energy they were meant to provide (the denominator)
    for ph, act_df in actual.items():
        curt = pd.DataFrame(index=index)
        for col in act_df.columns:
            if col in scheduled_ev:
                sched = scheduled_ev[col]
            elif pv_file is not None:
                sched = reconstruct_scheduled_pv(week_index, new_steps, pv_file)
            else:
                sched = None  # unknown schedule (PV without --location): skip from both
            if sched is None:
                curt[col] = 0.0
            else:
                curt[col] = np.clip(np.abs(sched) - np.abs(act_df[col].values), 0.0, None)
                scheduled_kwh += float(np.abs(sched).sum() * step_h)
        curt.to_csv(inv_dir / f"active_power_curtailment_phase_{ph}.csv")
        curtailed_kwh += float(curt.to_numpy().sum() * step_h)
    return curtailed_kwh, scheduled_kwh


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="sim_results/<timestamp>/sim_results_combined folder")
    ap.add_argument("--location", default=None,
                    help="Location key: reconstructs PV schedules "
                         "(sim_config.pv_source_path) and names the analysis_output "
                         "sub-folders the per-scenario summaries are saved into. "
                         "Omit to report PV curtailment as 0 and skip the analysis save.")
    ap.add_argument("--analysis-root", type=Path, default=THIS_DIR / "analysis_output",
                    help="Root of the analysis_output tree; each scenario's "
                         "curtailment_summary.csv is written next to that scenario's "
                         "strategy_summary.csv (default: simulation/analysis_output). "
                         "Ignored when --output-dir is given.")
    ap.add_argument("--scenario", default=None,
                    help="Process only this scenario folder (e.g. "
                         "pv_5_ev_20_files_max), instead of every scenario folder "
                         "found under --root. A yearly run holds both a max-load and "
                         "a min-load folder, so this is how one load extreme is "
                         "singled out -- same flag as compare_control_strategies.py.")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Write curtailment_summary.csv straight into this folder "
                         "instead of deriving analysis_output/<location>_ev*_pv*_<extreme>/ "
                         "from the scenario name. Mirrors compare_control_strategies.py "
                         "--output-dir, so both post-processing steps can be pointed at "
                         "the same phase<N>_<location>_<scenario> folder. Requires the "
                         "run to resolve to a single scenario (see --scenario).")
    ap.add_argument("--summary-csv", type=Path, default=None,
                    help="Optional extra path for one combined table across all scenarios.")
    args = ap.parse_args()

    pv_file = None
    if args.location:
        import sim_config as cfg
        pv_file = Path(cfg.pv_source_path(args.location))

    rows = []
    for strat in STRATEGY_DIRS:
        strat_dir = args.root / strat
        if not strat_dir.is_dir():
            continue
        validate = strat == "NoControl"
        for scen_dir in sorted(p for p in strat_dir.iterdir() if p.is_dir()):
            if args.scenario and scen_dir.name != args.scenario:
                continue
            week_dirs = sorted((p for p in scen_dir.iterdir() if WEEK_RE.search(p.name)),
                               key=lambda p: int(WEEK_RE.search(p.name).group(1)))
            strat_curt = 0.0
            strat_sched = 0.0
            for wd in week_dirs:
                c, s = compute_week(wd, pv_file, validate)
                strat_curt += c
                strat_sched += s
            pct = (100.0 * strat_curt / strat_sched) if strat_sched > 0 else 0.0
            rows.append({"strategy": strat, "scenario": scen_dir.name,
                         "n_weeks": len(week_dirs),
                         "total_scheduled_kwh": round(strat_sched, 3),
                         "total_curtailed_kwh": round(strat_curt, 3),
                         "curtailed_pct": round(pct, 4)})
            print(f"{strat:20s} {scen_dir.name:24s} weeks={len(week_dirs):3d} "
                  f"curtailed {strat_curt:9.3f} kWh = {pct:6.3f}% of {strat_sched:9.1f} kWh scheduled")

    if not rows:
        if args.scenario:
            print(f"[error] no scenario folder named {args.scenario!r} under {args.root}.")
            return 1
        print(f"[error] no strategy/scenario/week folders found under {args.root}.")
        return 1
    df = pd.DataFrame(rows)

    # Explicit single-folder save: lets run_test_plan.py's curtail step drop the
    # summary into the same phase<N>_<location>_<scenario> folder its compare step
    # writes strategy_summary.csv to, rather than the older per-scenario naming
    # analysis_dir_for() derives.
    if args.output_dir:
        scenarios = list(df["scenario"].unique())
        if len(scenarios) > 1:
            print(f"[error] --output-dir needs a single scenario, but --root holds "
                  f"{scenarios}. Narrow it with --scenario.")
            return 1
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "curtailment_summary.csv"
        df.drop(columns=["scenario"]).to_csv(out_path, index=False)
        (args.output_dir / "RESULTS_BASIS.md").write_text(RESULTS_BASIS_NOTE, encoding="utf-8")
        print(f"Wrote {out_path}")
    # Otherwise save each scenario's summary next to its strategy_summary.csv
    # under analysis_output/ (needs --location to name the folders).
    elif args.location:
        for scenario_name, sub in df.groupby("scenario"):
            out_dir = analysis_dir_for(scenario_name, args.location, args.analysis_root)
            if out_dir is None:
                print(f"[warn] scenario {scenario_name!r} not recognized; skipping analysis save.")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "curtailment_summary.csv"
            sub.drop(columns=["scenario"]).to_csv(out_path, index=False)
            (out_dir / "RESULTS_BASIS.md").write_text(RESULTS_BASIS_NOTE, encoding="utf-8")
            print(f"Wrote {out_path}")
    else:
        print("[note] --location not given; skipped saving into analysis_output "
              "(folder names need the location prefix).")

    if args.summary_csv:
        df.to_csv(args.summary_csv, index=False)
        print(f"Wrote {args.summary_csv}")

    return 0


if __name__ == "__main__":
    # Exit code matters: run_test_plan.py marks its curtail task failed on
    # non-zero, so a missing scenario folder must not look like success.
    raise SystemExit(main())
