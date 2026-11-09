# pid_optimizer_approx.py
"""
Fast PID gain optimizer for a pandapower network.

This script optimizes the proportional (Kp), integral (Ki), derivative (Kd) gains
of a VectorizedAsymmetricPIDController together with its decay coefficient
(q_decay), its per-step slew limit (q_rate) and its deadband hysteresis, on an
exact replica of the worst-case day simulated by yearly_tests.py (same network
seed, EV/PV placement, charging profiles, and step resolution).

Instead of running a full Newton-Raphson load flow at every optimization step,
the script:
1. Performs a single uncontrolled baseline simulation to capture the voltage
   trajectories at every inverter bus.
2. Computes all-bus 3-phase sensitivity anchors (dV/dQ) via perturbation every
   sim_config.SENS_REESTIMATE_HOURS of the day (that instant's household
   loads, EV charging power and PV injection applied), held constant between
   anchors (network_setup.calculate_time_varying_sensitivity) -- the same
   routine, cadence and hold the centralized controller uses online.
3. Evaluates candidate PID parameters using a fast discrete-time surrogate
   model that replays the whole day against the anchor active at each
   instant: V(t) = V_base(t) + S[t] * Q(t-1). The per-step Q update
   is custom_controller_example.pid_q_update — the exact same function the
   real VectorizedAsymmetricPIDController executes in the simulation.
   Candidates are scored by surrogate_objective.score(), a weighted sum of
   three normalized terms (voltage-band violation, reactive-power effort,
   control chatter) shared verbatim with droop_optimizer_approx.py.
   Numerically unstable trials are aborted.
4. Searches with Differential Evolution, then refines with Nelder-Mead.
5. Saves the optimal gains to the location+scenario file
   locations/<key>/gains/pid_<scenario>.csv (sim_config.pid_gains_path),
   which yearly_tests.py --location picks up automatically as its --pid-gains
   default for the same scenario (EV/PV share and placement seed).

See surrogate_objective.py for why chatter is measured on the second difference
of Q and why the term normalization is what stops the search from selecting a
bang-bang relay.
"""

import os
import time
import copy
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.optimize as opt
import pandapower.networks as nw

# Import baseline settings and setup from the thesis repository files
import sim_config as cfg
import network_setup as ns
import custom_controller_example as cce
import surrogate_objective as so

# ---------------------------------------------------------------------------
# Configuration -- every default comes from sim_config.py, the single source
# shared with yearly_tests.py and droop_optimizer_approx.py, so the gains are
# tuned at the same time resolution and on the same network the yearly runs
# use. Only the command line (--step, in practice via run.bat) overrides it.
# ---------------------------------------------------------------------------
SEED = cfg.SEED                    # RNG seed for the Differential Evolution search
NETWORK_SEED = cfg.NETWORK_SEED    # network/profile seed used by yearly_tests.py workers
EV_CONTRIBUTION = cfg.EV_CONTRIBUTION
PV_CONTRIBUTION = cfg.PV_CONTRIBUTION
CONTROL_TYPE = 0  # 0 = PID Control

# Voltage band and objective come from surrogate_objective.py, shared with
# droop_optimizer_approx.py so both controllers are tuned against one yardstick.
V_LOW, V_HIGH = so.V_LOW, so.V_HIGH

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Discrete-Time Surrogate Simulation
# ---------------------------------------------------------------------------
def simulate_surrogate(
    params: np.ndarray,
    V_base_all: np.ndarray, sens: dict,
    v_low: float = V_LOW, v_high: float = V_HIGH,
    plugged: np.ndarray = None, q_mode: int = cce.Q_MODE_ALWAYS,
    p_sched: np.ndarray = None, max_curtail_frac: float = cce.MAX_P_CURTAIL_FRAC
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the discrete-time PID controllers step-by-step against the
    zero-order-hold sensitivity anchors in `sens` (the dict returned by
    network_setup.calculate_time_varying_sensitivity) -- the same anchors,
    cadence and hold the real centralized controller uses online.

    params : (Kp, Ki, Kd, q_decay, q_rate, hysteresis)
    V_base_all : (n_steps, 3*n_buses) uncontrolled all-bus voltage trajectory.
    plugged : (n_steps, n_inv) bool, the plugged_in profile columns in the
        same inverter order as sens['v_rows']. With
        q_mode == cce.Q_MODE_PLUGGED_ONLY the surrogate applies the identical
        availability gate the real controller runs: Q and the integral are
        zeroed while no vehicle is plugged in, so the gains are tuned under
        the availability they will actually have.

    Each controller is fed back with its own bus/phase voltage (rows
    sens['v_rows']); the returned V_est covers ALL bus-phases so the tuning
    objective penalizes the whole network, exactly like the centralized one.

    The Q update is cce.pid_q_update — the identical function the real
    VectorizedAsymmetricPIDController runs, so the surrogate and the
    simulation share one control law by construction. The Q clip is the same
    voltage-dependent ceiling S_max(V) = V * I_max (cce.s_max_from_current) the
    real controller applies.

    p_sched : (n_steps, n_inv) scheduled active power per inverter (MW; the
        sgen ev_<id> profile). Only needed when max_curtail_frac < 1.0: then the
        applied Q is pulled below the law output via cce.apply_current_limit
        wherever curtailing P by max_curtail_frac cannot free enough current --
        the same bounded-curtailment law the real controller applies, so the
        surrogate tracks the ACTUALLY applied Q rather than assuming reactive
        priority. At max_curtail_frac=1.0 the cap never binds and the replay is
        identical to before (Q depends on V alone).
    """
    Kp, Ki, Kd, q_decay, q_rate, hysteresis = params
    S_full, anchor_of_step, v_rows = sens['S_full'], sens['anchor_of_step'], sens['v_rows']

    N_steps = V_base_all.shape[0]
    n_inv = len(v_rows)
    V_est = np.zeros_like(V_base_all)
    Q_PID = np.zeros((N_steps, n_inv))

    gate = (q_mode == cce.Q_MODE_PLUGGED_ONLY)
    integral = np.zeros(n_inv)
    q_prev = np.zeros(n_inv)

    # At t=0, the initial voltage is equal to the base voltage (since Q[-1] = 0)
    V_est[0] = V_base_all[0]
    prev_v = V_base_all[0, v_rows].copy()

    for t in range(N_steps):
        # Feedback: each inverter's own connection bus/phase voltage
        v_ctrl = V_est[t, v_rows]
        # Voltage-dependent apparent-power ceiling, per inverter (32 A limit)
        s_limit = cce.s_max_from_current(v_ctrl)
        q_out, integral, prev_v = cce.pid_q_update(
            v_ctrl, integral, prev_v, q_prev,
            Kp, Ki, Kd, q_decay, s_limit, v_low=v_low, v_high=v_high,
            q_rate=q_rate, hysteresis=hysteresis)
        # Plugged-only gate, identical to the real controller's step 4:
        # unplugged inverters output no Q and their integral is reset.
        if gate:
            unplugged = ~plugged[t]
            q_out[unplugged] = 0.0
            integral[unplugged] = 0.0
        # The PID law state (q_prev for the slew reference, integral) evolves on
        # the unclipped law output, exactly like the real controller, which sets
        # q_prev = q_output before applying the current-limit/curtailment cap.
        q_prev = q_out
        # Bounded curtailment cap: where P cannot be curtailed enough (only
        # under max_curtail_frac < 1.0), reduce the APPLIED Q so the network and
        # the effort/chatter terms below see what the inverter can actually
        # supply. cce.apply_current_limit is the same law the real controller runs.
        if p_sched is not None and max_curtail_frac < 1.0:
            _, q_out = cce.apply_current_limit(p_sched[t], q_out, s_limit, max_curtail_frac)
        Q_PID[t] = q_out

        # All-bus voltage prediction for the next step, with the anchor
        # matrix active at that instant (zero-order hold)
        if t < N_steps - 1:
            V_est[t+1] = V_base_all[t+1] + S_full[anchor_of_step[t+1]] @ q_out

            # Abort only on numerical blow-up. Q is clipped to +/- S_max(V) every
            # step, so the closed loop cannot diverge in amplitude; oscillatory
            # gains are handled by the chatter term in the objective.
            if not np.all(np.isfinite(V_est[t+1])):
                return None, None

    return V_est, Q_PID


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
# Search-space encoding. Kp and Ki span more than a decade, so they are searched
# in log10 space — otherwise Differential Evolution spends nearly all of its
# population in the top half of the range and barely resolves the small-gain
# region where the loop is actually stable.
#
# Ki is bounded well below the old (0.01, 1.0). The inverter saturates on the
# first step outside the deadband as soon as Ki * e >= S_max(V) ~ 7.7e-3 MVar,
# because the integral starts at zero and its first increment is exactly e.
# That threshold is Ki ~ 0.79 for a 0.01 p.u. excursion and Ki ~ 0.40 for a
# 0.02 p.u. one. Past it the loop stops being a PID and becomes a relay, which
# then limit-cycles against the one-step feedback delay -- the old tuned
# Ki = 0.899 did exactly this. Capping Ki at 0.10 keeps Ki * e an order of
# magnitude under s_limit across the excursions this feeder actually sees, so
# the loop stays in its linear region where the gains mean something.
PARAM_NAMES = ['Kp', 'Ki', 'Kd', 'q_decay', 'q_rate', 'hysteresis']

BOUNDS = [
    (-3.00, -0.50),   # log10(Kp)      -> 0.005 .. 0.158
    (-3.00, -0.50),   # log10(Ki)      -> 0.002 .. 0.100
    ( 0.00,  0.02),   # Kd             (derivative-on-measurement: safe but small)
    ( 0.80,  1),      # q_decay        (0.80 = fast bleed, 1.00 = no decay)
    ( 0.02,  1.00),   # q_rate / s_limit  (fraction of full range per step)
    ( 0.00,  0.008),  # hysteresis     (p.u. shrink of the integral-decay band)
]


def decode(x: np.ndarray, s_nominal: float) -> np.ndarray:
    """Map a search vector to physical PID parameters."""
    return np.array([
        10.0 ** x[0],        # Kp
        10.0 ** x[1],        # Ki
        x[2],                # Kd
        x[3],                # q_decay
        x[4] * s_nominal,    # q_rate, MVar per step (fraction of nominal S_max)
        x[5],                # hysteresis
    ])


# The objective itself (violation + effort + chatter, each normalized) lives in
# surrogate_objective.py and is shared verbatim with droop_optimizer_approx.py.

# ---------------------------------------------------------------------------
# Optimization Routine
# ---------------------------------------------------------------------------
# Differential-Evolution budget (kept in one place so the CMA-ES benchmark can
# be given a matching function-evaluation ceiling for a fair comparison).
DE_MAXITER = 60
DE_POPSIZE = 20


def _pid_run_de(objective):
    """Differential Evolution global search. Returns (x, objective, seconds)."""
    print("\nGlobal search (Differential Evolution)...")
    t0 = time.perf_counter()
    de = opt.differential_evolution(
        objective,
        BOUNDS,
        seed=SEED,
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=1e-8,
        strategy='best1bin',
        mutation=(0.3, 1.0),   # dithered: wide early exploration, fine late
        recombination=0.9,
        init='sobol',          # lower-discrepancy than latinhypercube
        updating='deferred',
        polish=False,          # replaced by the Nelder-Mead stage below
    )
    elapsed = time.perf_counter() - t0
    print(f"  DE best objective: {de.fun:.4f}  ({de.nit} generations)")
    return de.x, float(de.fun), elapsed


def _pid_run_cma(objective):
    """
    CMA-ES global search over the same box as DE. Returns (x, objective,
    seconds). Started from the box midpoint with per-coordinate initial step
    sizes proportional to each bound's range (CMA_stds): the six parameters
    are searched on wildly different scales (a 1.5-decade log10(Kp) span next
    to a 0.02-wide Kd), so an isotropic sigma0 would be far too coarse on the
    narrow axes and far too fine on the wide ones. With CMA_stds = per-axis
    range and sigma0 = 0.25, the initial std on each axis is a quarter of that
    axis's width, the value cma recommends.
    """
    import cma
    los = np.array([b[0] for b in BOUNDS])
    his = np.array([b[1] for b in BOUNDS])
    ranges = his - los
    x0 = 0.5 * (los + his)
    # Match DE's evaluation ceiling so the wall-clock/quality comparison is fair.
    max_evals = DE_POPSIZE * len(BOUNDS) * DE_MAXITER
    print("\nGlobal search (CMA-ES)...")
    t0 = time.perf_counter()
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.25, {
        'bounds': [los.tolist(), his.tolist()],
        'CMA_stds': ranges.tolist(),
        'seed': SEED,
        'maxfevals': max_evals,
        'tolfun': 1e-8,
        'verbose': -9,
    })
    es.optimize(objective)
    elapsed = time.perf_counter() - t0
    print(f"  CMA-ES best objective: {es.result.fbest:.4f}  "
          f"({es.result.iterations} generations, {es.result.evaluations} evals)")
    return np.array(es.result.xbest), float(es.result.fbest), elapsed


def optimise_pid(
    V_base_all: np.ndarray, sens: dict, s_nominal: float,
    plugged: np.ndarray = None, q_mode: int = cce.Q_MODE_ALWAYS,
    optimizer: str = 'de',
    p_sched: np.ndarray = None, max_curtail_frac: float = cce.MAX_P_CURTAIL_FRAC
) -> Tuple[np.ndarray, dict]:
    """
    Two-stage search: a global stage for coverage, then Nelder-Mead for local
    refinement.

    The global stage is selectable with `optimizer`:
      'de'   -- Differential Evolution (default; unchanged behaviour)
      'cma'  -- CMA-ES (self-adapting step size/covariance)
      'both' -- run both and keep the better refined result, printing a
                DE-vs-CMA-ES comparison table

    The refinement stage matters. The objective is piecewise-defined — deadband
    masks, np.clip on the output, the slew limiter — so it has kinks everywhere.
    DE's built-in polish uses L-BFGS-B, which estimates gradients by finite
    differences and is unreliable on exactly this kind of surface. Nelder-Mead
    is derivative-free and does not care about the kinks. Whichever global
    method(s) run go through the SAME Nelder-Mead polish, so the comparison is
    apples-to-apples.

    s_nominal is S_max at nominal voltage (V = 1.0 p.u.); it is the fixed
    reference used to scale q_rate (decode) and normalize the objective terms.
    The per-step Q clip inside simulate_surrogate is the voltage-dependent
    S_max(V), matching the real controller.

    plugged/q_mode are forwarded to simulate_surrogate so the gains are tuned
    under the same Q-availability gate the real run uses.

    Returns (physical_params, terms_of_best).
    """
    def surrogate_objective(x: np.ndarray) -> float:
        V_est, Q_PID = simulate_surrogate(
            decode(x, s_nominal), V_base_all=V_base_all, sens=sens,
            plugged=plugged, q_mode=q_mode,
            p_sched=p_sched, max_curtail_frac=max_curtail_frac)
        return so.score(V_est, Q_PID, s_nominal)   # None -> FAILURE_PENALTY

    def refine(x_global, pre):
        # Nelder-Mead polish; keep it only if it actually improves on the
        # global stage's own best (matches the original nm.fun < de.fun guard).
        nm = opt.minimize(
            surrogate_objective, x_global,
            method='Nelder-Mead',
            bounds=BOUNDS,
            options={'xatol': 1e-5, 'fatol': 1e-8, 'maxiter': 4000},
        )
        return (nm.x, float(nm.fun)) if nm.fun < pre else (x_global, pre)

    runners = {'de': _pid_run_de, 'cma': _pid_run_cma}
    methods = ['de', 'cma'] if optimizer == 'both' else [optimizer]

    results = {}   # label -> dict(pre, x_post, post, seconds)
    for m in methods:
        x_pre, pre, seconds = runners[m](surrogate_objective)
        print("Local refinement (Nelder-Mead)...")
        x_post, post = refine(x_pre, pre)
        print(f"  refined objective: {post:.4f}")
        results[m] = dict(pre=pre, x_post=x_post, post=post, seconds=seconds)

    if optimizer == 'both':
        so.print_comparison_table(results)

    best_label = min(results, key=lambda m: results[m]['post'])
    best_x = results[best_label]['x_post']
    if optimizer == 'both':
        print(f"  winner: {LABELS[best_label]} "
              f"(refined objective {results[best_label]['post']:.4f})")

    params = decode(best_x, s_nominal)
    V_est, Q_PID = simulate_surrogate(params, V_base_all, sens,
                                      plugged=plugged, q_mode=q_mode,
                                      p_sched=p_sched, max_curtail_frac=max_curtail_frac)
    terms = so.objective_terms(V_est, Q_PID, s_nominal)

    # A parameter pinned to a bound means the box, not the physics, is choosing.
    for name, value, (lo, hi) in zip(PARAM_NAMES, best_x, BOUNDS):
        span = hi - lo
        if value <= lo + 0.01 * span or value >= hi - 0.01 * span:
            print(f"  WARNING: {name} sits at a search bound ({value:.4g}); widen it and re-run.")

    return params, terms


# LABELS and the DE-vs-CMA comparison table are shared with
# droop_optimizer_approx.py -- see surrogate_objective.py.
LABELS = so.LABELS

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Tune the PID gains on the worst-case day. The time "
                    "resolution defaults to sim_config.STEP_MULT so the gains "
                    "match the resolution the yearly runs use.")
    parser.add_argument('--step', type=float, default=cfg.STEP_MULT,
                        help='Time-resolution multiplier: 1 = 15-minute steps, '
                             '3 = 5-minute steps, 15 = 1-minute steps '
                             f'(default from sim_config.py: {cfg.STEP_MULT})')
    parser.add_argument('--q-mode', type=int, default=cfg.Q_MODE, choices=[0, 1],
                        help='0 = supply Q at all times; 1 = supply Q only '
                             'while an EV is plugged in. Applied both in the '
                             'surrogate tuning and the verification runs '
                             f'(default from sim_config.py: {cfg.Q_MODE})')
    parser.add_argument('--optimizer', choices=['de', 'cma', 'both'], default='de',
                        help="Global-search stage: 'de' = Differential Evolution "
                             "(default), 'cma' = CMA-ES, 'both' = run both and keep "
                             'the better refined result (prints a comparison table). '
                             'Both share the same Nelder-Mead polish.')
    parser.add_argument('--max-curtail-frac', type=float, default=cfg.MAX_P_CURTAIL_FRAC,
                        metavar='F',
                        help='Max fraction [0-1] of active power an inverter may '
                             'curtail for Q before Q is reduced instead. Must match '
                             'the value the yearly run uses, so the gains are tuned '
                             'against the Q the inverters can actually apply '
                             f'(default from sim_config.py: {cfg.MAX_P_CURTAIL_FRAC})')
    parser.add_argument('--location', default=cfg.LOCATION,
                        choices=sorted(cfg.LOCATIONS),
                        help='Location to tune for: its ENTSO-E load zone sets '
                             'the worst-case day and its PVGIS curve the PV '
                             'injection. Gains are saved to '
                             'locations/<key>/gains/pid_<scenario>.csv '
                             f'(default from sim_config.py: {cfg.LOCATION})')
    args = parser.parse_args()
    step_mult = args.step
    q_mode = args.q_mode
    optimizer = args.optimizer
    max_curtail_frac = args.max_curtail_frac
    location = cfg.ensure_location(args.location)
    pv_file = cfg.pv_source_path(location)
    # Sensitivity anchors every cfg.SENS_REESTIMATE_HOURS of simulated time
    # with zero-order hold in between -- the same cadence the centralized
    # controller re-estimates its S_Q at.
    sens_anchor_every = cfg.sens_anchor_steps(step_mult)

    # 1. Identify worst-case day from this location's peak-load multiplier curve
    df_load = pd.read_csv(cfg.region_load_path(location))
    mult = np.array(df_load['Peak[pu]'].values)
    worst_idx = int(np.argmax(mult))
    worst_multiplier = float(mult[worst_idx])
    print("=" * 70)
    print(f"Location '{location}' worst-case day index: {worst_idx} "
          f"with multiplier: {worst_multiplier:.3f}")
    print("=" * 70)

    # 2. Build the network structure (same seed as the yearly_tests.py workers)
    net_template = nw.ieee_european_lv_asymmetric()
    (net_template, ev_sgen_indices, ev_phases,
     pv_sgen_indices, pv_phases,
     dyn_sgen_indices,
     dyn_sgen_indices_a, dyn_sgen_indices_b, dyn_sgen_indices_c
    ) = ns.setup_simulation(net_template, seed=NETWORK_SEED,
                            contribution=EV_CONTRIBUTION,
                            pv_contribution=PV_CONTRIBUTION)

    # Compile the list of all active inverters, their phases, and target buses
    inverters = dyn_sgen_indices_a + dyn_sgen_indices_b + dyn_sgen_indices_c
    phases = ['a']*len(dyn_sgen_indices_a) + ['b']*len(dyn_sgen_indices_b) + ['c']*len(dyn_sgen_indices_c)

    target_bus_id_a = [net_template.asymmetric_sgen.at[idx, 'bus'] for idx in dyn_sgen_indices_a]
    target_bus_id_b = [net_template.asymmetric_sgen.at[idx, 'bus'] for idx in dyn_sgen_indices_b]
    target_bus_id_c = [net_template.asymmetric_sgen.at[idx, 'bus'] for idx in dyn_sgen_indices_c]
    target_buses = target_bus_id_a + target_bus_id_b + target_bus_id_c

    # 3. Compute baseline profiles (uncontrolled run)
    print("Running uncontrolled baseline simulation to collect voltage profiles...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total_loss_zero, ow = ns.simulation_with_dynamic_gen(
            copy.deepcopy(net_template),
            ev_sgen_indices, ev_phases,
            pv_sgen_indices, pv_phases,
            dyn_sgen_indices,
            EV_CONTRIBUTION, PV_CONTRIBUTION,
            dyn_sgen_indices_a, dyn_sgen_indices_b, dyn_sgen_indices_c,
            weekly_mult=worst_multiplier,
            step_mult=step_mult,
            week=worst_idx,
            min_max_indx='max',
            individualcontrol=3,  # 3 = No Control (uncontrolled baseline)
            timestamp='baseline_zero',
            seed=NETWORK_SEED,
            return_ow=True,
            log_all_bus_voltages=True,
            pv_file=pv_file
        )

    # Extract the uncontrolled all-bus voltage trajectory (the tuning
    # objective penalizes the whole network, like the centralized controller)
    V_base_all = ns.extract_all_bus_v_base(ow, net_template)


    # Inverter limit: apparent-power ceiling at nominal voltage for the shared
    # 32 A current limit, S_max(1.0 p.u.) = V_ln_nominal * I_max. Used only as
    # the fixed reference for q_rate scaling and objective normalization; the
    # per-step Q clip is the voltage-dependent S_max(V) inside the surrogate.
    s_nominal = cce.s_max_from_current(1.0)

    # 4. Compute the zero-order-hold sensitivity anchors with the day's inputs
    household_loads, sgen_profiles, _, _, _ = ns.build_day_inputs(
        worst_idx, worst_multiplier, step_mult, ev_sgen_indices, pv_sgen_indices,
        seed=NETWORK_SEED, pv_file=pv_file)
    sens = ns.calculate_time_varying_sensitivity(
        net_template, inverters, phases, target_buses,
        household_loads, sgen_profiles,
        ev_sgen_indices, ev_phases, pv_sgen_indices, pv_phases,
        anchor_every=sens_anchor_every)

    # Plugged-in masks in the surrogate's inverter order (a, then b, then c --
    # NOT sgen_profiles column order), for the Q-availability gate.
    plugged = sgen_profiles[[f"plugged_{i}" for i in inverters]].to_numpy(dtype=bool)

    # Scheduled active power per inverter (MW), same inverter order, so the
    # surrogate can apply the bounded-curtailment cap when max_curtail_frac < 1.
    p_sched = sgen_profiles[[f"ev_{i}" for i in inverters]].to_numpy(dtype=float)

    # 5. Run optimization
    start_time = time.perf_counter()
    params, terms = optimise_pid(V_base_all, sens, s_nominal,
                                 plugged=plugged, q_mode=q_mode,
                                 optimizer=optimizer,
                                 p_sched=p_sched, max_curtail_frac=max_curtail_frac)
    elapsed = time.perf_counter() - start_time

    print(f"\nOptimization completed in {elapsed:.1f}s")
    print("Optimal parameters:")
    for name, value in zip(PARAM_NAMES, params):
        print(f"  {name:<11s} = {value:.6f}")
    print(so.format_breakdown(terms))

    # Save the optimal gains to this location's folder; yearly_tests.py --location
    # picks the matching file up as its --pid-gains default.
    opt_gains = params.tolist()
    gains_out_path = cfg.pid_gains_path(location)
    os.makedirs(os.path.dirname(gains_out_path), exist_ok=True)
    pd.DataFrame([dict(zip(PARAM_NAMES, opt_gains))]).to_csv(gains_out_path, index=False)
    print(f"Optimal PID gains saved to: {gains_out_path}")

    # 6. Final verification with full Newton-Raphson simulations
    print("\nRunning verification simulation with the previous default PID gains...")
    # The gains the old optimizer converged on. Ki=0.899 saturates the inverter
    # on the first step outside the deadband, so this run is the relay/limit-cycle
    # baseline the new objective is meant to beat.
    original_gains = [0.14024313, 0.89883566, 0.03838071, 0.95265426]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total_loss_original = ns.simulation_with_dynamic_gen(
            copy.deepcopy(net_template),
            ev_sgen_indices, ev_phases,
            pv_sgen_indices, pv_phases,
            dyn_sgen_indices,
            EV_CONTRIBUTION, PV_CONTRIBUTION,
            dyn_sgen_indices_a, dyn_sgen_indices_b, dyn_sgen_indices_c,
            weekly_mult=worst_multiplier,
            step_mult=step_mult,
            week=worst_idx,
            min_max_indx='max',
            individualcontrol=CONTROL_TYPE,
            timestamp='baseline_original',
            seed=NETWORK_SEED,
            pid_gains=original_gains,
            return_ow=False,
            q_mode=q_mode,
            pv_file=pv_file,
            max_curtail_frac=max_curtail_frac
        )

    print("Running verification simulation with optimal optimized PID gains...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total_loss_opt = ns.simulation_with_dynamic_gen(
            copy.deepcopy(net_template),
            ev_sgen_indices, ev_phases,
            pv_sgen_indices, pv_phases,
            dyn_sgen_indices,
            EV_CONTRIBUTION, PV_CONTRIBUTION,
            dyn_sgen_indices_a, dyn_sgen_indices_b, dyn_sgen_indices_c,
            weekly_mult=worst_multiplier,
            step_mult=step_mult,
            week=worst_idx,
            min_max_indx='max',
            individualcontrol=CONTROL_TYPE,
            timestamp='optimized_final',
            seed=NETWORK_SEED,
            pid_gains=opt_gains,
            return_ow=False,
            q_mode=q_mode,
            pv_file=pv_file,
            max_curtail_frac=max_curtail_frac
        )

    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS (Worst-case Day Line Losses)")
    print("=" * 70)
    print(f"Uncontrolled base case losses               : {total_loss_zero:.4f} kWh")
    print(f"Previous default PID gains case losses      : {total_loss_original:.4f} kWh")
    print(f"Optimized PID gains case losses             : {total_loss_opt:.4f} kWh")
    improvement_over_zero = (total_loss_zero - total_loss_opt) / total_loss_zero * 100
    improvement_over_original = (total_loss_original - total_loss_opt) / total_loss_original * 100
    print(f"Improvement over Uncontrolled               : {improvement_over_zero:+.2f}%")
    print(f"Improvement over previous PID gains         : {improvement_over_original:+.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
