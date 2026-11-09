"""
hosting_capacity_sweep.py -- Hosting Capacity Gain: the increase in the
maximum allowable EV penetration percentage (before voltage violations
exceed an acceptable threshold, default 5%) that each control strategy
achieves over No Control.

For each requested strategy, binary-searches EV_CONTRIBUTION (the fraction of
network loads fitted with an EV charger) for the largest value whose single
worst-case-day voltage-violation percentage stays at or below
--violation-threshold-pct, using the exact same all-LV-bus violation figure
network_setup.py already computes per run (see
compare_control_strategies.voltage_violation_percentage) -- just read
in-memory from the OutputWriter instead of round-tripping through
voltage_quality.csv.

METHODOLOGY CAVEATS (read before citing these numbers)
--------------------------------------------------------------------------
- PID/Droop gains are NOT retuned at every penetration level tested -- the
  gains already tuned for the current location+scenario (sim_config's
  EV_CONTRIBUTION / PV_CONTRIBUTION / NETWORK_SEED, or --pid-gains) are
  reused across the whole sweep. Retuning at every point would multiply
  runtime by the DE optimizer's own cost per point (each a multi-minute
  search). Reusing means PID/Droop's hosting capacity is a lower-bound
  estimate at penetrations far from the scenario they were tuned on -- a
  controller retuned at that penetration might do better. The Centralized
  controller is unaffected: it re-estimates its sensitivity matrix online
  every run, independent of EV_CONTRIBUTION.
- Each EV% point is evaluated on the single worst-case day (the week with
  the highest load multiplier), not the full 53-week yearly batch --
  consistent with yearly_tests.py's own fast default, but a strategy's
  hosting capacity could differ on a different week.
- The bisection search assumes voltage violations are monotonically
  non-decreasing in EV_CONTRIBUTION (more EV load -> equal or worse voltage
  drop). This holds under No Control and is expected to hold for the
  controllers once they saturate, but isn't proven for every possible
  gain/topology combination -- a strategy with a sharply non-monotonic
  violation curve could be mis-estimated.
- Every evaluated point runs a full single-day simulation and writes its
  usual per-run CSVs under sim_results/hosting_capacity_<timestamp>_.../ --
  expect one small results folder per bisection step, not just per strategy.

Usage:
    python hosting_capacity_sweep.py --controls 0 1 2
    python hosting_capacity_sweep.py --controls 1 2 --location athens --ev-max 0.8
    python hosting_capacity_sweep.py --controls 0 1 2 --violation-threshold-pct 2
"""
import os
import copy
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandapower.networks as nw

import sim_config as cfg
import network_setup as ns
from compare_control_strategies import STRATEGY_COLORS, STRATEGY_LABELS

CONTROL_NAMES = {0: 'PIDcontrol', 1: 'Droopcontrol', 2: 'CentralizedControl', 3: 'NoControl'}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# One evaluation: run a single worst-case day at a given EV penetration and
# control strategy, return the exact all-LV-bus voltage-violation percentage.
# ---------------------------------------------------------------------------
def violation_pct_for_run(net_template, ev_contribution, pv_contribution, control_type,
                          timestamp, step_mult, q_mode, pid_gains, droop_gains_file, pv_file,
                          network_seed, worst_idx, worst_multiplier):
    net = copy.deepcopy(net_template)
    (net, ev_sgen_indices, ev_phases, pv_sgen_indices, pv_phases, dyn_sgen_indices,
     dyn_sgen_indices_a, dyn_sgen_indices_b, dyn_sgen_indices_c) = ns.setup_simulation(
        net, seed=network_seed, contribution=ev_contribution, pv_contribution=pv_contribution)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, ow = ns.simulation_with_dynamic_gen(
            net, ev_sgen_indices, ev_phases, pv_sgen_indices, pv_phases, dyn_sgen_indices,
            ev_contribution, pv_contribution, dyn_sgen_indices_a, dyn_sgen_indices_b, dyn_sgen_indices_c,
            weekly_mult=worst_multiplier, step_mult=step_mult, week=worst_idx, min_max_indx='max',
            individualcontrol=control_type, timestamp=timestamp, seed=network_seed,
            pid_gains=pid_gains, q_mode=q_mode, pv_file=pv_file, droop_gains_file=droop_gains_file,
            return_ow=True,
            # The exact violation count doesn't depend on this flag (it's
            # always computed over every LV bus); skipping the all-bus CSV
            # export is purely a speed/disk-space win across a sweep.
            log_all_bus_voltages=False)

    n_lv_buses = int((net.bus.vn_kv < 1.0).sum())
    # 'res_bus_3ph.vm_a_pu' always exists: the vm_a_pu_min/vm_a_pu_max eval_name
    # columns inside it are registered unconditionally (network_setup.py), so its
    # row count is the number of simulated steps regardless of log_all_bus_voltages.
    n_steps = len(ow.output['res_bus_3ph.vm_a_pu'])
    total_measurements = n_steps * 3 * n_lv_buses
    if total_measurements == 0:
        return float('nan')
    return 100.0 * ow.total_violations / total_measurements


# ---------------------------------------------------------------------------
# Bisection search for the hosting capacity of one strategy
# ---------------------------------------------------------------------------
def bisect_hosting_capacity(evaluate, ev_min, ev_max, threshold_pct, tol):
    """evaluate(ev_contribution) -> violation_pct. Binary search for the
    largest feasible EV_CONTRIBUTION (violation_pct <= threshold_pct) in
    [ev_min, ev_max], assuming violations are monotonically non-decreasing in
    ev_contribution (see the module docstring's caveat).

    Returns (hosting_capacity, violation_pct_at_capacity, censored), where
    censored is 'low' if even ev_min is already infeasible, 'high' if even
    ev_max is still feasible (both mean the true value lies outside the
    tested range), else None.
    """
    viol_lo = evaluate(ev_min)
    if viol_lo > threshold_pct:
        return ev_min, viol_lo, 'low'

    viol_hi = evaluate(ev_max)
    if viol_hi <= threshold_pct:
        return ev_max, viol_hi, 'high'

    lo, hi = ev_min, ev_max
    viol_at_lo = viol_lo
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        viol_mid = evaluate(mid)
        if viol_mid <= threshold_pct:
            lo, viol_at_lo = mid, viol_mid
        else:
            hi = mid
    return lo, viol_at_lo, None


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_hosting_capacity_bar(summary: pd.DataFrame, controls_to_run: list, output_path: Path) -> None:
    """Bar chart: hosting capacity (max feasible EV penetration %) per
    strategy, with the No Control baseline as a dashed reference line."""
    labels = [STRATEGY_LABELS[CONTROL_NAMES[c]] for c in controls_to_run]
    ordered = summary.reindex(labels)
    colors = [STRATEGY_COLORS[CONTROL_NAMES[c]] for c in controls_to_run]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = sns.barplot(x=ordered.index, y=ordered["hosting_capacity_ev_pct"],
                      hue=ordered.index, palette=colors, legend=False, ax=ax)
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(f"{height:,.1f}%", (bar.get_x() + bar.get_width() / 2, height),
                   ha="center", va="bottom", fontsize=10)

    if 3 in controls_to_run:
        baseline = summary.loc[STRATEGY_LABELS["NoControl"], "hosting_capacity_ev_pct"]
        ax.axhline(baseline, color="#898781", linestyle="--", linewidth=1,
                  label="No Control baseline")
        ax.legend(loc="best", frameon=True)

    ax.set_xlabel("Control strategy")
    ax.set_ylabel("Hosting capacity [max EV penetration, %]")
    ax.set_title("EV hosting capacity by control strategy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Hosting Capacity Gain: binary-search the maximum EV "
                    "penetration each control strategy tolerates (voltage "
                    "violations <= --violation-threshold-pct) and compare "
                    "against the No Control baseline. See the module "
                    "docstring for the methodology caveats (gains are reused "
                    "across the sweep, not retuned per point).")
    parser.add_argument('--controls', type=int, nargs='+', default=[0, 1, 2], choices=[0, 1, 2, 3],
                        metavar='N',
                        help='Control types to test against the No Control baseline: '
                             '0=PID, 1=Droop, 2=Centralized (No Control, 3, is always '
                             'evaluated as the baseline whether or not it is listed here) '
                             '(default: 0 1 2)')
    parser.add_argument('--ev-min', type=float, default=0.05,
                        help='Lower bound of the EV_CONTRIBUTION search range (default: 0.05)')
    parser.add_argument('--ev-max', type=float, default=1.0,
                        help='Upper bound of the EV_CONTRIBUTION search range (default: 1.0)')
    parser.add_argument('--tol', type=float, default=0.01,
                        help='Bisection tolerance on EV_CONTRIBUTION, e.g. 0.01 resolves '
                             'hosting capacity to the nearest 1 percentage point (default: 0.01)')
    parser.add_argument('--violation-threshold-pct', type=float,
                        default=cfg.HOSTING_CAPACITY_VIOLATION_THRESHOLD_PCT,
                        help='Voltage-violation percentage above which an EV penetration '
                             'level is considered infeasible (default from sim_config.py: '
                             f'{cfg.HOSTING_CAPACITY_VIOLATION_THRESHOLD_PCT})')
    parser.add_argument('--pv-contribution', type=float, default=cfg.PV_CONTRIBUTION,
                        help='PV units as a fraction of the number of loads, held fixed '
                             f'across the sweep (default from sim_config.py: {cfg.PV_CONTRIBUTION})')
    parser.add_argument('--step', type=float, default=cfg.STEP_MULT,
                        help='Time-resolution multiplier. Should match the resolution any '
                             'tuned PID/Droop gains were produced at '
                             f'(default from sim_config.py: {cfg.STEP_MULT})')
    parser.add_argument('--q-mode', type=int, default=cfg.Q_MODE, choices=[0, 1],
                        help='0 = supply Q at all times; 1 = supply Q only while an EV is '
                             f'plugged in (default from sim_config.py: {cfg.Q_MODE})')
    parser.add_argument('--location', default=cfg.LOCATION, choices=sorted(cfg.LOCATIONS),
                        help=f'Simulation location (default from sim_config.py: {cfg.LOCATION})')
    parser.add_argument('--pid-gains', type=float, nargs='+', default=None, metavar='G',
                        help='PID params: Kp Ki Kd [q_decay [q_rate [hysteresis]]]. Default: '
                             'the location+scenario tuned gains under locations/<key>/gains/ '
                             'if present, otherwise built-in values.')
    parser.add_argument('--output-dir', type=Path, default=Path('analysis_output'),
                        help='Where to write the summary CSV and bar chart (default: analysis_output)')
    args = parser.parse_args()

    location = cfg.ensure_location(args.location)
    pv_file = cfg.pv_source_path(location)
    droop_gains_file = cfg.droop_gains_path(location)

    df_load = pd.read_csv(cfg.region_load_path(location))
    max_mult = df_load['Peak[pu]'].to_numpy()
    worst_idx = int(np.argmax(max_mult))
    worst_multiplier = float(max_mult[worst_idx])

    PID_PARAM_ORDER = ['Kp', 'Ki', 'Kd', 'q_decay', 'q_rate', 'hysteresis']
    if args.pid_gains is None:
        pid_gains_csv = cfg.pid_gains_path(location)
        if os.path.exists(pid_gains_csv):
            row = pd.read_csv(pid_gains_csv).iloc[0]
            pid_gains = [float(row[c]) for c in PID_PARAM_ORDER if c in row.index]
            print(f"PID gains loaded from {location}/{os.path.basename(pid_gains_csv)}: {pid_gains}")
        else:
            pid_gains = [0.05, 0.02, 0.0, 0.95, 0.00079, 0.004]
            print(f"No tuned PID gains for '{location}'; using built-in defaults {pid_gains}.")
    else:
        pid_gains = args.pid_gains

    net_template = nw.ieee_european_lv_asymmetric()
    timestamp = f"hosting_capacity_{time.strftime('%Y-%m-%d_%H.%M')}"

    # EV_CONTRIBUTION is rounded down to a charger COUNT (setup_simulation:
    # int(len(loads) * contribution)); below 1/len(loads) that count is 0, no
    # EV sgens are created at all, and the network build hits a pre-existing
    # AttributeError elsewhere (net.asymmetric_sgen has no 'phi' column when
    # it holds only PV generators). Fail clearly here instead of surfacing
    # that error mid-bisection.
    n_loads = len(net_template.asymmetric_load)
    min_feasible_ev = 1.0 / n_loads
    if args.ev_min < min_feasible_ev:
        parser.error(f"--ev-min {args.ev_min} places 0 EV chargers on this "
                     f"{n_loads}-load network (int({n_loads} * {args.ev_min}) = 0); "
                     f"use --ev-min >= {min_feasible_ev:.4f} (1/{n_loads}).")

    print("=" * 70)
    print(f"Location '{location}' worst-case day index: {worst_idx} "
         f"(multiplier {worst_multiplier:.3f})")
    print(f"EV_CONTRIBUTION search range [{args.ev_min:.2f}, {args.ev_max:.2f}], "
         f"tol={args.tol:.3f}, violation threshold={args.violation_threshold_pct:.1f}%")
    print("Gains are reused across the whole sweep, not retuned per EV level "
         "(see the module docstring's methodology caveats).")
    print("=" * 70)

    # No Control (3) is always evaluated first as the baseline the gain is
    # measured against, whether or not it was explicitly requested.
    controls_to_run = list(dict.fromkeys([3] + args.controls))

    results = {}
    for control_type in controls_to_run:
        name = CONTROL_NAMES[control_type]
        print(f"\n--- {name} ---")

        def evaluate(ev, control_type=control_type, name=name):
            run_timestamp = f"{timestamp}_{name}"
            viol_pct = violation_pct_for_run(
                net_template, ev, args.pv_contribution, control_type, run_timestamp,
                args.step, args.q_mode, pid_gains, droop_gains_file, pv_file,
                cfg.NETWORK_SEED, worst_idx, worst_multiplier)
            status = 'OK' if viol_pct <= args.violation_threshold_pct else 'FAIL'
            print(f"  EV={ev * 100:5.1f}% -> violations={viol_pct:5.2f}% ({status})")
            return viol_pct

        hc, viol_at_hc, censored = bisect_hosting_capacity(
            evaluate, args.ev_min, args.ev_max, args.violation_threshold_pct, args.tol)

        if censored == 'low':
            print(f"  WARNING: even EV_CONTRIBUTION={args.ev_min:.2f} exceeds the violation "
                 f"threshold; hosting capacity < {args.ev_min * 100:.1f}% "
                 f"(reported as the lower bound).")
        elif censored == 'high':
            print(f"  WARNING: EV_CONTRIBUTION={args.ev_max:.2f} is still feasible; hosting "
                 f"capacity >= {args.ev_max * 100:.1f}% (reported as the upper bound) -- rerun "
                 f"with a larger --ev-max to pin down the true value.")
        print(f"  Hosting capacity: {hc * 100:.2f}% EV penetration "
             f"(violations {viol_at_hc:.2f}% at that point)")

        results[control_type] = {
            'hosting_capacity_ev_pct': hc * 100.0,
            'violation_pct_at_capacity': viol_at_hc,
            'censored': censored or '',
        }

    baseline_hc = results[3]['hosting_capacity_ev_pct']
    rows = []
    for control_type in controls_to_run:
        r = results[control_type]
        rows.append({
            'strategy': STRATEGY_LABELS[CONTROL_NAMES[control_type]],
            'hosting_capacity_ev_pct': r['hosting_capacity_ev_pct'],
            'violation_pct_at_capacity': r['violation_pct_at_capacity'],
            'hosting_capacity_gain_pct_points': r['hosting_capacity_ev_pct'] - baseline_hc,
            'censored': r['censored'],
        })
    summary = pd.DataFrame(rows).set_index('strategy')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "hosting_capacity_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nWrote {summary_path}")
    print(summary)

    bar_path = args.output_dir / "hosting_capacity_bar.png"
    plot_hosting_capacity_bar(summary, controls_to_run, bar_path)
    print(f"Wrote {bar_path}")


if __name__ == "__main__":
    main()
