"""
Fast Droop gain optimizer for a pandapower network.

This script optimizes the positive (droopgain_pos) and negative (droopgain_neg)
droop gains of an AsymmetricDynamicDroopController, on an exact replica of the
worst-case day simulated by yearly_tests.py (same network seed, EV/PV placement,
charging profiles, and step resolution).

To accelerate the process, the script:
1. Performs a single uncontrolled baseline simulation to collect the voltage
   trajectories at every inverter bus.
2. Computes all-bus 3-phase sensitivity anchors (dV/dQ) via perturbation every
   sim_config.SENS_REESTIMATE_HOURS of the day (that instant's household
   loads, EV charging power and PV injection applied), held constant between
   anchors (network_setup.calculate_time_varying_sensitivity) -- the same
   routine, cadence and hold the centralized controller uses online.
3. Simulates candidate droop gains closed-loop in milliseconds, replaying
   the whole day against the anchor active at each instant:
   V(t) = V_base(t) + S[t] * Q(t-1). The per-step Q update is
   custom_controller_example.droop_q_update — the exact same function the real
   AsymmetricDynamicDroopController executes in the simulation. Candidates are
   scored by surrogate_objective.score(), a weighted sum of three normalized
   terms (voltage-band violation, reactive-power effort, control chatter)
   shared verbatim with pid_optimizer_approx.py. Numerically unstable trials
   are aborted.
4. Searches with Differential Evolution, then refines with Nelder-Mead.
5. Saves the optimized gains to the location+scenario file
   locations/<key>/gains/droop_<scenario>.csv (sim_config.droop_gains_path,
   read by network_setup.py for every Droop run at that location/scenario) and
   the verification voltage/Q profiles alongside as droop_profiles_<scenario>.csv.

Why the shared objective matters here
-------------------------------------
Droop is a memoryless algebraic law, so unlike the PID loop it cannot wind up or
latch into bang-bang behaviour. It can still oscillate: the loop closes around
one step of feedback delay (V(t-1) -> Q(t) -> V(t)), so a droop gain g is
destabilizing once g * rho(S_QQ) exceeds 1, where rho is the spectral radius of
the sensitivity matrix.

Both the old crossing-count term and the new chatter term do detect that gross
instability. The difference is shape, not sensitivity. Measured on a 10-inverter
replica with rho = 1.73 (stability limit g = 0.577):

    g      chatter      old crossing penalty
    0.10     0.002          0
    0.30     0.016          0
    0.50     0.038          0
    0.80  81121            6.9e7

The crossing count is flat at zero right up to the bifurcation and then jumps by
eight orders of magnitude, so Differential Evolution gets no gradient telling it
which stable gain is smoother -- only a cliff. Chatter rises monotonically across
the whole stable region, so the search can rank 0.1 against 0.5 on smoothness and
trade that against band compliance. The crossing count's 1e-4 p.u. mask also hides
any limit cycle whose per-step voltage swing falls below it, and its unnormalized
1e5 scale is not comparable to the effort and violation penalties it was summed
with. Both objectives reject g = 0.8; only the new one can shape the choice below
it.
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

# Import settings and setup from the thesis repository files
import sim_config as cfg
import network_setup as ns
import custom_controller_example as cce
import surrogate_objective as so

# ---------------------------------------------------------------------------
# Configuration -- every default comes from sim_config.py, the single source
# shared with yearly_tests.py and pid_optimizer_approx.py, so the gains are
# tuned at the same time resolution and on the same network the yearly runs
# use. Only the command line (--step, in practice via run.bat) overrides it.
# ---------------------------------------------------------------------------
SEED = cfg.SEED                    # RNG seed for the Differential Evolution search
NETWORK_SEED = cfg.NETWORK_SEED    # network/profile seed used by yearly_tests.py workers
EV_CONTRIBUTION = cfg.EV_CONTRIBUTION
PV_CONTRIBUTION = cfg.PV_CONTRIBUTION
CONTROL_TYPE = 1  # 1 = Droop Control

# Voltage band and objective come from surrogate_objective.py, shared with
# pid_optimizer_approx.py so both controllers are tuned against one yardstick.
V_LOW, V_HIGH = so.V_LOW, so.V_HIGH

# Each inverter gets its own (droopgain_pos, droopgain_neg) pair, searched
# jointly. The parameter vector x is laid out as
# [pos_inv_0, pos_inv_1, ..., pos_inv_{n-1}, neg_inv_0, neg_inv_1, ..., neg_inv_{n-1}].

# Droop gains in MVar per p.u. of voltage deviation. Unlike the PID gains these
# span less than a decade of useful range, so they are searched linearly.
GAIN_BOUNDS = (0.0, 5.0)   # per-inverter bound, both pos (overvoltage absorption)
                           # and neg (undervoltage injection)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Closed-Loop Surrogate Droop Simulation
# ---------------------------------------------------------------------------
def simulate_surrogate_droop(
    droopgain_pos: np.ndarray, droopgain_neg: np.ndarray,
    V_base_all: np.ndarray, sens: dict,
    v_low: float = V_LOW, v_high: float = V_HIGH,
    plugged: np.ndarray = None, q_mode: int = cce.Q_MODE_ALWAYS,
    p_sched: np.ndarray = None, max_curtail_frac: float = cce.MAX_P_CURTAIL_FRAC
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the discrete-time Droop controllers step-by-step against the
    zero-order-hold sensitivity anchors in `sens` (the dict returned by
    network_setup.calculate_time_varying_sensitivity) -- the same anchors,
    cadence and hold the real centralized controller uses online.

    droopgain_pos/droopgain_neg are per-inverter arrays, shape (n_inv,) --
    each inverter is tuned independently. cce.droop_q_update is elementwise,
    so passing arrays here scores exactly the individual gains each inverter
    would use in the real simulation.

    V_base_all : (n_steps, 3*n_buses) uncontrolled all-bus voltage trajectory.
    Each controller is fed back with its own bus/phase voltage (rows
    sens['v_rows']); the returned V_est covers ALL bus-phases so the tuning
    objective penalizes the whole network, exactly like the centralized one.

    plugged : (n_steps, n_inv) bool, the plugged_in profile columns in the
    same inverter order as sens['v_rows']. With
    q_mode == cce.Q_MODE_PLUGGED_ONLY the surrogate applies the identical
    availability gate the real controller runs: Q is zeroed while no vehicle
    is plugged in (droop is memoryless, so that is the whole gate).

    The Q update is cce.droop_q_update — the identical function the real
    AsymmetricDynamicDroopController runs, so the surrogate and the simulation
    share one control law by construction. The Q clip is the same
    voltage-dependent ceiling S_max(V) = V * I_max (cce.s_max_from_current) the
    real controller applies.

    p_sched : (n_steps, n_inv) scheduled active power per inverter (MW). Only
        used when max_curtail_frac < 1.0: the applied Q is then pulled below the
        droop law wherever curtailing P by max_curtail_frac cannot free enough
        current (cce.apply_current_limit, the same bounded-curtailment law the
        real controller runs). At max_curtail_frac=1.0 the replay is identical.
    """
    S_full, anchor_of_step, v_rows = sens['S_full'], sens['anchor_of_step'], sens['v_rows']
    N_steps = V_base_all.shape[0]
    n_inv = len(v_rows)
    V_est = np.zeros_like(V_base_all)
    Q_droop = np.zeros((N_steps, n_inv))

    gate = (q_mode == cce.Q_MODE_PLUGGED_ONLY)
    V_est[0] = V_base_all[0]

    for t in range(N_steps):
        # Feedback: each inverter's own connection bus/phase voltage
        v_ctrl = V_est[t, v_rows]
        # Voltage-dependent apparent-power ceiling, per inverter (32 A limit)
        s_limit = cce.s_max_from_current(v_ctrl)
        q_out = cce.droop_q_update(
            v_ctrl, droopgain_pos, droopgain_neg, s_limit,
            v_low=v_low, v_high=v_high)
        # Plugged-only gate, identical to the real controller's step 4
        if gate:
            q_out = np.where(plugged[t], q_out, 0.0)
        # Bounded curtailment cap (only bites at max_curtail_frac < 1.0): reduce
        # the applied Q where P cannot be curtailed enough. Droop is memoryless,
        # so there is no state to keep -- the applied Q is the whole story.
        if p_sched is not None and max_curtail_frac < 1.0:
            _, q_out = cce.apply_current_limit(p_sched[t], q_out, s_limit, max_curtail_frac)
        Q_droop[t] = q_out

        # All-bus voltage prediction for the next step, with the anchor
        # matrix active at that instant (zero-order hold)
        if t < N_steps - 1:
            V_est[t+1] = V_base_all[t+1] + S_full[anchor_of_step[t+1]] @ q_out

            # Abort only on numerical blow-up. Q is clipped to +/- S_max(V) every
            # step, so the closed loop cannot diverge in amplitude; oscillatory
            # gains are handled by the chatter term in the objective.
            if not np.all(np.isfinite(V_est[t+1])):
                return None, None

    return V_est, Q_droop


def simulate_surrogate_droop_batch(
    droopgain_pos: np.ndarray, droopgain_neg: np.ndarray,
    V_base_all: np.ndarray, sens: dict,
    v_low: float = V_LOW, v_high: float = V_HIGH,
    plugged: np.ndarray = None, q_mode: int = cce.Q_MODE_ALWAYS,
    p_sched: np.ndarray = None, max_curtail_frac: float = cce.MAX_P_CURTAIL_FRAC
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batched counterpart of simulate_surrogate_droop: replays the whole day for
    an entire population of candidate gains at once instead of one candidate
    at a time. The per-step Python for-loop is the actual bottleneck of
    the optimizer (each iteration only does a few tiny numpy ops), so this
    pays that loop's interpreter overhead once per Differential Evolution
    generation instead of once per population member -- see optimise_droop's
    use of scipy's `vectorized=True`.

    droopgain_pos/droopgain_neg: shape (n_candidates, n_inv) -- one row per
    candidate. Every candidate's trajectory is independent (this is not a
    coupled multi-candidate simulation); they are only computed together so
    the underlying numpy calls operate on one bigger array instead of being
    invoked separately per candidate. The per-step voltage update becomes a
    single (n_candidates, n_inv) @ (n_inv, 3n) matrix product whose row c is
    exactly the single-candidate result, because rows of a matmul never mix.

    The all-bus voltage trajectory of a whole population (steps x candidates
    x 3n bus-phases) would be gigabytes, so it is never stored: the tuning
    objective's raw violation sum (surrogate_objective.violation_core) is
    accumulated step by step and returned instead, alongside the Q
    trajectories and a per-candidate validity mask (a diverging candidate's
    NaN values stay confined to its own rows; it is masked at scoring).

    Returns (Q_droop, violation_sums, valid_mask):
      Q_droop        : (n_steps, n_candidates, n_inv)
      violation_sums : (n_candidates,) raw violation_core sums over all steps
                       and all bus-phases
      valid_mask     : (n_candidates,) False where a trajectory went non-finite
    """
    anchor_of_step, v_rows = sens['anchor_of_step'], sens['v_rows']
    N_steps = V_base_all.shape[0]
    n_inv = len(v_rows)
    n_candidates = droopgain_pos.shape[0]

    # float32 for the arrays that drive the per-step matmul and violation_core
    # -- this is a search proxy that only ranks candidates (the verification
    # run in main() replays the winner through full double-precision
    # pandapower), so float32's ~1e-7 relative error is negligible next to the
    # 1e-3 p.u. objective scale (surrogate_objective.V_SCALE) while roughly
    # doubling throughput on both, the two costs that dominate this loop.
    # Copies, not in-place casts, so the float64 sens/V_base_all the caller
    # (and simulate_surrogate_droop's single-candidate Nelder-Mead refinement)
    # still holds are untouched. s_max_from_current/droop_q_update keep
    # computing in float64 internally (V_LN_NOMINAL_KV is a numpy float64
    # scalar, which forces promotion) -- cheap on these (n_candidates, n_inv)
    # intermediates regardless, so q_out is cast back down right after.
    V_base_all = V_base_all.astype(np.float32)
    S_full = sens['S_full'].astype(np.float32)
    droopgain_pos = droopgain_pos.astype(np.float32)
    droopgain_neg = droopgain_neg.astype(np.float32)

    Q_droop = np.zeros((N_steps, n_candidates, n_inv), dtype=np.float32)
    violation_sums = np.zeros(n_candidates)
    valid = np.ones(n_candidates, dtype=bool)

    # Current-step all-bus voltages, one row per candidate; every candidate
    # starts on the uncontrolled base trajectory (Q[-1] = 0).
    v_all = np.broadcast_to(V_base_all[0], (n_candidates, V_base_all.shape[1]))

    gate = (q_mode == cce.Q_MODE_PLUGGED_ONLY)

    for t in range(N_steps):
        violation_sums += so.violation_core(v_all, v_low=v_low, v_high=v_high, axis=1)

        # Feedback: each inverter's own connection bus/phase voltage
        v_ctrl = v_all[:, v_rows]
        s_limit = cce.s_max_from_current(v_ctrl)
        q_out = cce.droop_q_update(
            v_ctrl, droopgain_pos, droopgain_neg, s_limit,
            v_low=v_low, v_high=v_high).astype(np.float32, copy=False)
        # Plugged-only gate: plugged[t] row-broadcasts over the candidate axis
        if gate:
            q_out = q_out * plugged[t]
        # Bounded curtailment cap (only at max_curtail_frac < 1.0). p_sched[t]
        # is (n_inv,) and broadcasts over the candidate axis of q_out/s_limit;
        # apply_current_limit works in float64, so cast the result back to
        # float32 to keep the per-step matmul on the fast path.
        if p_sched is not None and max_curtail_frac < 1.0:
            _, q_out = cce.apply_current_limit(p_sched[t], q_out, s_limit, max_curtail_frac)
            q_out = q_out.astype(np.float32, copy=False)
        Q_droop[t] = q_out

        if t < N_steps - 1:
            v_all = V_base_all[t+1] + q_out @ S_full[anchor_of_step[t+1]].T
            valid &= np.all(np.isfinite(v_all), axis=1)

    return Q_droop, violation_sums, valid

# ---------------------------------------------------------------------------
# Optimization Routine
# ---------------------------------------------------------------------------
# Differential-Evolution budget, in one place so the CMA-ES benchmark can be
# given a matching function-evaluation ceiling.
DE_MAXITER = 60
DE_POPSIZE = 20

# LABELS and the DE-vs-CMA comparison table are shared with
# pid_optimizer_approx.py -- see surrogate_objective.py.
LABELS = so.LABELS


def _droop_seeded_init_pop(n_inv, inverter_ids, rng, prev_gains_file=None):
    """
    DE's structured initial population. A uniform draw over (0, 5)^(2*n_inv)
    essentially never contains an "all gains small" candidate, yet the stable
    region of this landscape lives there (a gain g destabilizes once
    g * rho(S) > 1, see the module docstring) -- an unseeded population is
    all-oscillating and DE_MAXITER generations cannot descend out of it. Seed
    a few structured members: a geometric ramp of uniform small gains, the
    pure zero candidate, and the previously tuned gains if a CSV exists;
    best1bin then anchors recombination on the best of these immediately.

    Returns (init_pop, prev_gains) where prev_gains is the clipped previous-CSV
    vector (or None) -- reused as CMA-ES's starting mean.
    """
    lo, hi = GAIN_BOUNDS
    n_pop = DE_POPSIZE * 2 * n_inv
    init_pop = rng.uniform(lo, hi, size=(n_pop, 2 * n_inv))
    ramp = np.geomspace(0.02, 1.0, 16)
    for r, g in enumerate(ramp):
        init_pop[r, :] = g
    init_pop[16, :] = 0.0
    prev_gains = None
    prev_csv = prev_gains_file
    if prev_csv and os.path.exists(prev_csv):
        df_prev = pd.read_csv(prev_csv).set_index('id')
        pos_col = 'gain_pos' if 'gain_pos' in df_prev.columns else 'mean_positive'
        neg_col = 'gain_neg' if 'gain_neg' in df_prev.columns else 'mean_negative'
        if all(i in df_prev.index for i in inverter_ids):
            prev_gains = np.clip(np.concatenate([
                df_prev.loc[inverter_ids, pos_col].values,
                df_prev.loc[inverter_ids, neg_col].values]), lo, hi)
            init_pop[17] = prev_gains
    return init_pop, prev_gains


def optimise_droop(
    V_base_all: np.ndarray, sens: dict, s_nominal: float,
    inverter_ids: list,
    plugged: np.ndarray = None, q_mode: int = cce.Q_MODE_ALWAYS,
    optimizer: str = 'de', prev_gains_file: str = None,
    p_sched: np.ndarray = None, max_curtail_frac: float = cce.MAX_P_CURTAIL_FRAC
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Two-stage search: a global stage for coverage, then Nelder-Mead for local
    refinement.

    The global stage is selectable with `optimizer`:
      'de'   -- Differential Evolution (default; unchanged behaviour)
      'cma'  -- CMA-ES (self-adapting step size/covariance)
      'both' -- run both and keep the better refined result, printing a
                DE-vs-CMA-ES comparison table

    Both global methods drive the SAME vectorized batched surrogate
    (simulate_surrogate_droop_batch) once per generation, and both go through
    the SAME single-candidate Nelder-Mead polish, so the comparison is
    apples-to-apples.

    Every inverter gets its own (pos, neg) pair, so the search space is
    2 * n_inv dimensional: x = [pos_0, ..., pos_{n-1}, neg_0, ..., neg_{n-1}].

    s_nominal is S_max at nominal voltage (V = 1.0 p.u.), the fixed reference
    used to normalize the objective terms; the per-step Q clip inside the
    surrogate is the voltage-dependent S_max(V), matching the real controller.

    plugged/q_mode are forwarded to the surrogates so the gains are tuned
    under the same Q-availability gate the real run uses.

    Returns (pos_gains, neg_gains, terms_of_best), each gains array shape (n_inv,).
    """
    n_inv = len(inverter_ids)
    bounds = [GAIN_BOUNDS] * (2 * n_inv)
    lo, hi = GAIN_BOUNDS
    # Mean-normalization denominator of the violation term: every step of the
    # day times every bus-phase of the network (see objective_terms_batch).
    n_v_entries = V_base_all.size

    def surrogate_objective(x: np.ndarray) -> float:
        # Single-candidate objective -- used by the Nelder-Mead refinement
        # stage below, which evaluates one point at a time.
        V_est, Q_droop = simulate_surrogate_droop(
            x[:n_inv], x[n_inv:], V_base_all=V_base_all, sens=sens,
            plugged=plugged, q_mode=q_mode,
            p_sched=p_sched, max_curtail_frac=max_curtail_frac)
        return so.score(V_est, Q_droop, s_nominal)   # None -> FAILURE_PENALTY

    def score_population(X: np.ndarray) -> np.ndarray:
        # Whole-population objective: X has shape (S, 2*n_inv), one row per
        # candidate. Runs the batched surrogate once for the whole population.
        pos_batch = X[:, :n_inv]   # (S, n_inv)
        neg_batch = X[:, n_inv:]   # (S, n_inv)
        Q_droop, violation_sums, valid = simulate_surrogate_droop_batch(
            pos_batch, neg_batch, V_base_all=V_base_all, sens=sens,
            plugged=plugged, q_mode=q_mode,
            p_sched=p_sched, max_curtail_frac=max_curtail_frac)
        valid &= np.all(np.isfinite(Q_droop), axis=(0, 2))
        return so.score_batch(violation_sums, n_v_entries, Q_droop, s_nominal,
                              valid_mask=valid)

    rng = np.random.default_rng(SEED)
    init_pop, prev_gains = _droop_seeded_init_pop(n_inv, inverter_ids, rng, prev_gains_file=prev_gains_file)

    def run_de():
        # scipy hands the vectorized callback x with shape (2*n_inv, S) --
        # one column per candidate -- and expects an (S,) array back.
        print("\nGlobal search (Differential Evolution, vectorized)...")
        t0 = time.perf_counter()
        de = opt.differential_evolution(
            lambda x: score_population(x.T),
            bounds,
            seed=SEED,
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            # 1e-8 (the previous value) requires the population's objective
            # values to agree to ~8 digits, which this landscape's FAILURE_PENALTY
            # trials and oscillatory region essentially never reach -- it always
            # burns the full DE_MAXITER budget. 1e-6 still demands the population
            # be almost fully collapsed (a much stricter bar than scipy's own
            # default of 1e-2), so it should only cut generations that were
            # already redundant; Nelder-Mead refines whatever DE returns either
            # way. Watch the printed "(N generations)" line after re-running --
            # if quality regresses on your tuning runs, revert to 1e-8.
            tol=1e-6,
            strategy='best1bin',
            mutation=(0.3, 1.0),   # dithered: wide early exploration, fine late
            recombination=0.9,
            init=init_pop,         # seeded population (see above)
            updating='deferred',
            polish=False,          # replaced by the Nelder-Mead stage below
            vectorized=True,       # evaluate the whole population in one call
        )
        elapsed = time.perf_counter() - t0
        print(f"  DE best objective: {de.fun:.4f}  ({de.nit} generations)")
        return de.x, float(de.fun), elapsed

    def run_cma():
        # CMA-ES ask/tell, feeding each generation's whole asked population
        # through the same batched surrogate (one call per generation, mirroring
        # DE's vectorized=True). The bounds are uniform (0, 5), so an isotropic
        # step size is appropriate; sigma0 = 1/4 of the box width. The starting
        # mean reuses the same structured seed DE relies on -- the previous
        # tuned gains if available, otherwise a small uniform gain -- so CMA-ES
        # begins in the stable region instead of the all-oscillating interior.
        import cma
        x0 = prev_gains if prev_gains is not None else np.full(2 * n_inv, 0.2)
        sigma0 = 0.25 * (hi - lo)
        max_evals = DE_POPSIZE * (2 * n_inv) * DE_MAXITER
        print("\nGlobal search (CMA-ES, batched ask/tell)...")
        t0 = time.perf_counter()
        es = cma.CMAEvolutionStrategy(np.asarray(x0, dtype=float).tolist(), sigma0, {
            'bounds': [[lo] * (2 * n_inv), [hi] * (2 * n_inv)],
            'seed': SEED,
            'maxfevals': max_evals,
            'tolfun': 1e-8,
            'verbose': -9,
        })
        while not es.stop():
            asked = es.ask()
            scores = score_population(np.asarray(asked))
            es.tell(asked, scores.tolist())
        elapsed = time.perf_counter() - t0
        print(f"  CMA-ES best objective: {es.result.fbest:.4f}  "
              f"({es.result.iterations} generations, {es.result.evaluations} evals)")
        return np.asarray(es.result.xbest), float(es.result.fbest), elapsed

    # Nelder-Mead needs roughly n+1 simplex points to move at all; scale the
    # iteration/eval budget with dimensionality instead of the fixed count
    # that sufficed for the old 2-parameter search.
    nm_budget = max(4000, 200 * 2 * n_inv)

    def refine(x_global, pre):
        nm = opt.minimize(
            surrogate_objective, x_global,
            method='Nelder-Mead',
            bounds=bounds,
            options={'xatol': 1e-5, 'fatol': 1e-8, 'maxiter': nm_budget, 'maxfev': nm_budget},
        )
        return (nm.x, float(nm.fun)) if nm.fun < pre else (x_global, pre)

    print(f"\nSearching {2 * n_inv} parameters ({n_inv} inverters x 2 gains each).")
    runners = {'de': run_de, 'cma': run_cma}
    methods = ['de', 'cma'] if optimizer == 'both' else [optimizer]

    results = {}   # label -> dict(pre, x_post, post, seconds)
    for m in methods:
        x_pre, pre, seconds = runners[m]()
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

    best_pos, best_neg = best_x[:n_inv], best_x[n_inv:]
    V_est, Q_droop = simulate_surrogate_droop(best_pos, best_neg, V_base_all, sens,
                                              plugged=plugged, q_mode=q_mode,
                                              p_sched=p_sched, max_curtail_frac=max_curtail_frac)
    terms = so.objective_terms(V_est, Q_droop, s_nominal)

    # A gain pinned to a bound means the box, not the physics, is choosing it.
    span = hi - lo
    for gain_name, values in (('droopgain_pos', best_pos), ('droopgain_neg', best_neg)):
        pinned = [inv_id for inv_id, v in zip(inverter_ids, values)
                  if v <= lo + 0.01 * span or v >= hi - 0.01 * span]
        if pinned:
            print(f"  WARNING: {gain_name} sits at a search bound for {len(pinned)}/{n_inv} "
                  f"inverter(s) {pinned}; widen it and re-run.")

    return best_pos, best_neg, terms


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Tune the per-inverter droop gains on the worst-case day. "
                    "The time resolution defaults to sim_config.STEP_MULT so "
                    "the gains match the resolution the yearly runs use.")
    parser.add_argument('--step', type=float, default=cfg.STEP_MULT,
                        help='Time-resolution multiplier: 1 = 15-minute steps, '
                             '3 = 5-minute steps, 15 = 1-minute steps '
                             f'(default from sim_config.py: {cfg.STEP_MULT})')
    parser.add_argument('--q-mode', type=int, default=cfg.Q_MODE, choices=[0, 1],
                        help='0 = supply Q at all times; 1 = supply Q only '
                             'while an EV is plugged in. Applied both in the '
                             'surrogate tuning and the verification run '
                             f'(default from sim_config.py: {cfg.Q_MODE})')
    parser.add_argument('--optimizer', choices=['de', 'cma', 'both'], default='de',
                        help="Global-search stage: 'de' = Differential Evolution "
                             "(default), 'cma' = CMA-ES, 'both' = run both and keep "
                             'the better refined result (prints a comparison table). '
                             'Both drive the batched surrogate and share the same '
                             'Nelder-Mead polish.')
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
                             'locations/<key>/gains/droop_<scenario>.csv '
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
    N_steps = V_base_all.shape[0]
    n_inv = len(inverters)


    # Apparent-power ceiling at nominal voltage for the shared 32 A current
    # limit, S_max(1.0 p.u.) = V_ln_nominal * I_max. Fixed reference for the
    # objective normalization; the per-step Q clip is the voltage-dependent
    # S_max(V) inside the surrogate.
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

    # 5. Run optimization -- one (pos, neg) pair per inverter
    start_time = time.perf_counter()
    best_pos, best_neg, terms = optimise_droop(V_base_all, sens, s_nominal, inverters,
                                               plugged=plugged, q_mode=q_mode,
                                               optimizer=optimizer,
                                               prev_gains_file=cfg.droop_gains_path(location),
                                               p_sched=p_sched, max_curtail_frac=max_curtail_frac)
    elapsed = time.perf_counter() - start_time
    print(f"\nOptimization completed in {elapsed:.1f}s")
    print("Optimal Droop Gains per inverter:")
    print(f"  {'sgen_id':>8s} {'phase':>5s} {'pos':>8s} {'neg':>8s}")
    for sgen_id, ph, pos, neg in zip(inverters, phases, best_pos, best_neg):
        print(f"  {sgen_id:>8d} {ph:>5s} {pos:>8.4f} {neg:>8.4f}")
    print(f"  mean               {best_pos.mean():>8.4f} {best_neg.mean():>8.4f}")
    print(so.format_breakdown(terms))
    
    # 6. Final verification with full Newton-Raphson simulation
    print("\nRunning verification simulation with optimal optimized Droop gains...")

    # network_setup.py reads the droop gains from this file, so the new gains must
    # be in place BEFORE the verification run. Back up the previous file so it can
    # be restored if the verification crashes.
    gains_path = cfg.droop_gains_path(location)
    os.makedirs(os.path.dirname(gains_path), exist_ok=True)
    gains_backup = gains_path + '.bak'

    if os.path.exists(gains_path):
        if os.path.exists(gains_backup):
            os.remove(gains_backup)
        os.rename(gains_path, gains_backup)

    df_opt = pd.DataFrame({
        'id': inverters,
        'gain_pos': best_pos,
        'gain_neg': best_neg
    })
    df_opt.to_csv(gains_path, index=False)
    print(f"Optimal droop gains saved to: {gains_path}")

    try:
        # Run verification simulation with ControlType=1 (Droop Control);
        # the controller fetches the pos/neg gains from the csv we just wrote
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            total_loss_opt, ow_opt = ns.simulation_with_dynamic_gen(
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
                individualcontrol=CONTROL_TYPE,  # 1 = Droop
                timestamp='optimized_droop',
                seed=NETWORK_SEED,
                return_ow=True,
                q_mode=q_mode,
                pv_file=pv_file,
                droop_gains_file=gains_path,
                max_curtail_frac=max_curtail_frac
            )

        print("\n" + "=" * 70)
        print(f"VERIFICATION RESULTS (Worst-case Day Line Losses, step={step_mult})")
        print("=" * 70)
        print(f"Uncontrolled base case losses               : {total_loss_zero:.4f} kWh")
        print(f"Optimized Droop gains case losses           : {total_loss_opt:.4f} kWh")
        improvement = (total_loss_zero - total_loss_opt) / total_loss_zero * 100
        print(f"Improvement over Uncontrolled               : {improvement:+.2f}%")
        print("=" * 70)

        # 7. Save voltage and Q profiles for plotting
        print("\nSaving optimal simulation profiles...")
        df_profiles = pd.DataFrame({'Time_Step': np.arange(N_steps)})
        for i in range(n_inv):
            sgen_id = inverters[i]
            ph = phases[i]
            b_id = target_buses[i]

            v_col_key = f'res_bus_3ph.vm_{ph}_pu'
            q_col_key = f'asymmetric_sgen.q_{ph}_mvar'

            if v_col_key in ow_opt.output and b_id in ow_opt.output[v_col_key].columns:
                df_profiles[f'v_inv_{sgen_id}_ph_{ph}_pu'] = ow_opt.output[v_col_key][b_id].values
            if q_col_key in ow_opt.output and sgen_id in ow_opt.output[q_col_key].columns:
                df_profiles[f'q_inv_{sgen_id}_ph_{ph}_mvar'] = ow_opt.output[q_col_key][sgen_id].values

        profiles_path = cfg.droop_profiles_path(location)
        df_profiles.to_csv(profiles_path, index=False)
        print(f"Profiles successfully saved to: {profiles_path}")

    except BaseException:
        # Verification failed: put the previous gains file back
        if os.path.exists(gains_backup):
            if os.path.exists(gains_path):
                os.remove(gains_path)
            os.rename(gains_backup, gains_path)
            print(f"Verification failed - previous gains restored to: {gains_path}")
        raise
    else:
        # Success: the new gains stay in place, the backup is no longer needed
        if os.path.exists(gains_backup):
            os.remove(gains_backup)

if __name__ == "__main__":
    main()
