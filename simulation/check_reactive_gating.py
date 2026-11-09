"""
check_reactive_gating.py -- verify every EV inverter only supplies reactive
power while its plugged-in gate (Q_MODE_PLUGGED_ONLY) is true, by comparing
each week's persisted Q trace against the actual plugged mask the controller
was given.

The EV-to-session assignment -- and therefore the plugged-in mask -- is
reseeded identically on every build_day_inputs() call within a run (same
NETWORK_SEED, same ev_sgen_indices, same file listing on disk), regardless of
week index or weekly_mult (see network_setup.build_day_inputs). So one
reconstruction per inverter, at the run's time resolution, covers every week.

Usage:
    python check_reactive_gating.py --root sim_results/<timestamp>/sim_results_combined \
        --scenario pv_5_ev_40_files_max
"""
import argparse
from pathlib import Path

import numpy as np
import pandapower.networks as nw
import pandas as pd

import network_setup as ns
import sim_config as cfg

# Numerical noise floor: anything above this while unplugged is a genuine
# gating leak, not floating-point residue from the load-flow solve.
Q_TOLERANCE_KVAR = 0.01

CONTROLLED_STRATEGIES = ["PIDcontrol", "Droopcontrol", "CentralizedControl"]


def reconstruct_plugged_masks(step_mult: float) -> dict:
    """{sgen_idx: (phase, plugged_bool_array)}. Identical for every week of a
    run (see module docstring), so this is built once, not per week."""
    net = nw.ieee_european_lv_asymmetric()
    net, ev_sgen_indices, ev_phases, pv_sgen_indices, *_ = ns.setup_simulation(
        net, seed=cfg.NETWORK_SEED, contribution=cfg.EV_CONTRIBUTION,
        pv_contribution=cfg.PV_CONTRIBUTION)
    _, sgen_profiles, _, _, _ = ns.build_day_inputs(
        0, 1.0, step_mult, ev_sgen_indices, pv_sgen_indices, seed=cfg.NETWORK_SEED)

    masks = {}
    for i, sgen_idx in enumerate(ev_sgen_indices):
        plugged = sgen_profiles[f"plugged_{sgen_idx}"].to_numpy() > 0.5
        masks[sgen_idx] = (ev_phases[i], plugged)
    return masks


def check_week(week_dir: Path, masks: dict) -> list[dict]:
    """Violation records (one per inverter with any leaked step) for a
    single week_N folder."""
    violations = []
    q_by_phase = {}
    for phase in ("A", "B", "C"):
        path = week_dir / "inverter_profiles" / f"global_q_phase_{phase}.csv"
        if path.exists():
            q_by_phase[phase] = pd.read_csv(path, index_col=0)

    for sgen_idx, (phase, plugged) in masks.items():
        df = q_by_phase.get(phase.upper())
        col = f"inverter_{sgen_idx}"
        if df is None or col not in df.columns:
            continue  # inverter never active on this phase/week -- nothing to check
        q = df[col].to_numpy()
        n = min(len(q), len(plugged))
        leak_mask = (~plugged[:n]) & (np.abs(q[:n]) > Q_TOLERANCE_KVAR)
        if leak_mask.any():
            violations.append({
                "week_dir": str(week_dir),
                "inverter_id": sgen_idx,
                "phase": phase,
                "n_leaked_steps": int(leak_mask.sum()),
                "n_unplugged_steps": int((~plugged[:n]).sum()),
                "max_leaked_q_kvar": float(np.abs(q[:n][leak_mask]).max()),
            })
    return violations


def discover_scenario_dir(control_dir: Path, scenario: str | None) -> Path:
    if scenario is not None:
        return control_dir / scenario
    children = sorted(p for p in control_dir.iterdir() if p.is_dir())
    if len(children) != 1:
        names = ", ".join(p.name for p in children)
        raise ValueError(f"{control_dir} has {len(children)} scenarios ({names}); "
                        "pass --scenario.")
    return children[0]


def main():
    parser = argparse.ArgumentParser(
        description="Check that EV inverters only supply Q while their EV is plugged in.")
    parser.add_argument("--root", type=Path, required=True,
                       help="Path to a sim_results_combined/ folder.")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--step", type=float, default=cfg.STEP_MULT,
                       help="Time-resolution multiplier the run was generated at "
                            f"(default from sim_config.py: {cfg.STEP_MULT})")
    parser.add_argument("--output", type=Path, default=Path("gating_violations.csv"))
    args = parser.parse_args()

    print(f"Reconstructing plugged masks at step={args.step} ...")
    masks = reconstruct_plugged_masks(args.step)
    print(f"{len(masks)} EV inverters.")

    all_violations = []
    for strategy in CONTROLLED_STRATEGIES:
        control_dir = args.root / strategy
        if not control_dir.is_dir():
            print(f"[skip] {control_dir} not found")
            continue
        scenario_dir = discover_scenario_dir(control_dir, args.scenario)
        week_dirs = sorted(scenario_dir.glob("week_*"),
                          key=lambda p: int(p.name.split("_")[1]))
        print(f"Checking {strategy} ({len(week_dirs)} weeks) ...")
        for week_dir in week_dirs:
            for v in check_week(week_dir, masks):
                v["strategy"] = strategy
                all_violations.append(v)

    if not all_violations:
        print("\nNo gating violations found: every inverter's Q was zero "
             "whenever its plugged mask was False.")
        return

    df = pd.DataFrame(all_violations)
    summary = (df.groupby(["strategy", "inverter_id", "phase"])
              .agg(weeks_affected=("week_dir", "nunique"),
                   total_leaked_steps=("n_leaked_steps", "sum"),
                   max_leaked_q_kvar=("max_leaked_q_kvar", "max"))
              .reset_index()
              .sort_values("max_leaked_q_kvar", ascending=False))
    print(f"\n{len(summary)} (strategy, inverter, phase) combinations show leaked Q:\n")
    print(summary.to_string(index=False))
    summary.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
