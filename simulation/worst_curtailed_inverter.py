"""
worst_curtailed_inverter.py -- find the inverter(s) hit hardest by active-power
curtailment (charging/PV power given up so the inverter could supply Q to fight
over/under-voltage).

Per inverter, across all weeks of a run, it accumulates:
  scheduled_kwh -- energy it was meant to provide (reconstructed from the saved
                   ev_charging_session_ids.csv, exactly as compute_curtailment.py),
  curtailed_kwh -- energy it gave up = sum(max(|P_scheduled|-|P_actual|,0)) * dt,
  curtailed_pct -- 100 * curtailed / scheduled (its OWN share, so a small charger
                   that loses most of its session ranks worse than a big one that
                   loses a little).

Two "worst" views because they answer different questions:
  * by kWh -- who lost the most energy in absolute terms (grid-impact view).
  * by %  -- who was most affected relative to what they were promised (fairness
             view); this is usually the more meaningful "worst inverter".

Usage:
    python worst_curtailed_inverter.py --root sim_results/<ts>/sim_results_combined
    python worst_curtailed_inverter.py --root <...> --strategy CentralizedControl --top 10
    python worst_curtailed_inverter.py --root <...> --location berlin_50hertz   # also PV + save CSV
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compute_curtailment import (STRATEGY_DIRS, WEEK_RE, analysis_dir_for,
                                  reconstruct_scheduled_ev, reconstruct_scheduled_pv)


def _step_h(index) -> float:
    diffs = pd.to_timedelta(pd.Index(index).astype(str)).to_series().diff().dropna()
    return float(diffs.median().total_seconds() / 3600.0) if len(diffs) else 0.0


def per_inverter_curtailment(scen_dir: Path, pv_file: Path | None) -> pd.DataFrame:
    """One row per inverter: phase, bucket, scheduled_kwh, curtailed_kwh,
    curtailed_pct, aggregated over every week folder in this strategy/scenario."""
    week_dirs = sorted((p for p in scen_dir.iterdir() if WEEK_RE.search(p.name)),
                       key=lambda p: int(WEEK_RE.search(p.name).group(1)))
    sched_kwh: dict[str, float] = {}
    curt_kwh: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for wd in week_dirs:
        inv_dir = wd / "inverter_profiles"
        actual = {ph: pd.read_csv(inv_dir / f"global_p_phase_{ph}.csv", index_col=0)
                  for ph in ("A", "B", "C") if (inv_dir / f"global_p_phase_{ph}.csv").exists()}
        if not actual:
            continue
        first = next(iter(actual.values()))
        new_steps, step_h = len(first), _step_h(first.index)
        m = WEEK_RE.search(wd.name)
        week_index = int(m.group(1)) if m else 0

        scheduled_ev = reconstruct_scheduled_ev(wd, new_steps)
        # inverter -> phase / bucket, for the report (constant across weeks)
        ids_path = wd / "ev_charging_session_ids.csv"
        if ids_path.exists():
            for _, r in pd.read_csv(ids_path).iterrows():
                meta.setdefault(f"inverter_{int(r['inverter_id'])}",
                                {"phase": r["phase"], "bucket": r["bucket"]})

        for ph, act_df in actual.items():
            for col in act_df.columns:
                if col in scheduled_ev:
                    sched = scheduled_ev[col]
                elif pv_file is not None:
                    sched = reconstruct_scheduled_pv(week_index, new_steps, pv_file)
                    meta.setdefault(col, {"phase": ph.lower(), "bucket": "pv"})
                else:
                    continue  # unknown schedule (PV without --location)
                curt = np.clip(np.abs(sched) - np.abs(act_df[col].values), 0.0, None)
                sched_kwh[col] = sched_kwh.get(col, 0.0) + float(np.abs(sched).sum() * step_h)
                curt_kwh[col] = curt_kwh.get(col, 0.0) + float(curt.sum() * step_h)

    rows = []
    for inv, s in sched_kwh.items():
        c = curt_kwh.get(inv, 0.0)
        rows.append({"inverter": inv,
                     "phase": meta.get(inv, {}).get("phase", "?"),
                     "bucket": meta.get(inv, {}).get("bucket", "?"),
                     "scheduled_kwh": round(s, 3),
                     "curtailed_kwh": round(c, 3),
                     "curtailed_pct": round(100.0 * c / s, 4) if s > 0 else 0.0})
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str, top: int):
    if df.empty or df["curtailed_kwh"].sum() == 0:
        print(f"\n=== {label}: no curtailment ===")
        return
    print(f"\n=== {label} ===")
    for key, title in (("curtailed_kwh", "by absolute energy lost (kWh)"),
                       ("curtailed_pct", "by % of own scheduled charging")):
        top_df = df.sort_values(key, ascending=False).head(top)
        print(f"  worst {title}:")
        for _, r in top_df.iterrows():
            print(f"    {r['inverter']:14s} phase {r['phase']} {r['bucket']:>8s} | "
                  f"{r['curtailed_kwh']:9.3f} kWh lost of {r['scheduled_kwh']:9.3f} "
                  f"scheduled ({r['curtailed_pct']:6.3f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="sim_results/<timestamp>/sim_results_combined folder")
    ap.add_argument("--strategy", default=None, choices=STRATEGY_DIRS,
                    help="Only this control strategy (default: all that curtail).")
    ap.add_argument("--top", type=int, default=5, help="How many worst inverters to list (default 5).")
    ap.add_argument("--location", default=None,
                    help="Location key: also reconstruct PV schedules and save a "
                         "curtailment_by_inverter.csv into analysis_output.")
    ap.add_argument("--analysis-root", type=Path, default=Path(__file__).resolve().parent / "analysis_output")
    args = ap.parse_args()

    pv_file = None
    if args.location:
        import sim_config as cfg
        pv_file = Path(cfg.pv_source_path(args.location))

    # One per-inverter CSV PER control type (run all strategies by default so
    # every control type's file is written; --strategy restricts to one).
    strategies = [args.strategy] if args.strategy else STRATEGY_DIRS
    for strat in strategies:
        strat_dir = args.root / strat
        if not strat_dir.is_dir():
            continue
        for scen_dir in sorted(p for p in strat_dir.iterdir() if p.is_dir()):
            df = per_inverter_curtailment(scen_dir, pv_file)
            report(df, f"{strat} / {scen_dir.name}", args.top)
            if args.location and not df.empty:
                out_dir = analysis_dir_for(scen_dir.name, args.location, args.analysis_root)
                if out_dir is not None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"curtailment_by_inverter_{strat}.csv"
                    df.sort_values("curtailed_kwh", ascending=False).to_csv(out_path, index=False)
                    print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
