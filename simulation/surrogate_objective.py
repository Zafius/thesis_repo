"""
Shared objective/metric module for all three reactive-power controllers.

Two distinct objectives live here, so every controller sources its scoring from
one place:

* objective_terms/objective_value/score -- the PID and Droop gain optimizers
  score a candidate by replaying the worst-case day on the linear surrogate
  V(t) = V_base(t) + S_QQ[t] @ Q(t-1) and reducing the (V_est, Q) trajectories to
  a single scalar (violation + effort + chatter).
* central_objective -- the per-step objective the centralized
  LinearVoltVarController minimizes online (all-bus voltage penalty + Q effort).
  It used to be an inline closure inside the controller; it lives here now so the
  centralized controller and the two optimizers share one module.

This mirrors how custom_controller_example.pid_q_update / droop_q_update are the
single source of truth for the control laws themselves. voltage_unbalance_factor
is a reporting metric (EN 50160 VUF); it is deliberately NOT part of any
objective -- see network_setup.export_results, which prints it beside the losses.

Design notes
------------
Both objectives are built on the same two primitive terms (violation_core /
effort_core below): the identical [V_LOW, V_HIGH] band, the identical 5/3
over/under-voltage asymmetry, and the identical (Q / S_max)^2 effort measure.
They differ only in aggregation and weighting: the centralized controller
applies a hard violation priority to one instant, the tuning objective
normalizes day-long trajectories and adds the chatter term (which only exists
for trajectories).

Three tuning terms, each normalized by a reference "bad" value so that a term
of 1.0 means roughly the same amount of badness in all three:

    violation   mean((5*over)^2 + (3*under)^2) outside [0.97, 1.05], / V_SCALE^2
                -- over ALL 3n bus-phases of the network, like the centralized
                objective, not just the inverter buses
    effort      mean((Q / s_limit)^2)                    -- already in [0, 1]
    chatter     mean((d2Q / s_limit)^2) / CHATTER_SCALE^2

1. Chatter is measured on the *second* difference of the control signal, not the
   first. A controller tracking a slow PV ramp has a large first difference but
   almost no curvature; a limit cycle reverses direction every few steps and so
   has large curvature. Measured on this loop, d2 separates an oscillating from
   a smooth controller by ~167x where d1 separates them by only ~5x.

   The formulation this replaced counted sign changes of diff(V) above a 1e-4
   p.u. mask. That is a step function of the parameters -- Differential Evolution
   sees no gradient toward smoother solutions -- and it is blind to any limit
   cycle whose per-step voltage swing falls under the mask.

2. The normalization is load-bearing, not cosmetic. A bang-bang relay genuinely
   *minimizes* voltage violation, so without comparable term scales `violation`
   swallows the other two and the search selects the relay. That is precisely
   what the old objective did to the PID loop. Calibrated on a single-inverter
   replica of the PID loop:

       controller             violation  effort  chatter   total
       relay,   Ki = 0.899        0.04     0.03    28.0     28.1
       gentle,  Ki = 0.02        14.16     0.01     0.25    14.4
       moderate,Ki = 0.10         1.65     0.05     1.5      3.2   <- selected

If you retune the weights, print the breakdown (both optimizers do) and keep the
three terms within an order of magnitude of each other for the solutions you
actually care about.
"""

from typing import Dict

import numpy as np

import sim_config as cfg

# Voltage band (p.u.) -- defined once in sim_config.py, re-exported here so
# existing `so.V_LOW` references keep working.
V_LOW, V_HIGH = cfg.V_LOW, cfg.V_HIGH

# Relative severity of over- vs under-voltage
W_OVER, W_UNDER = 5.0, 3.0

# Reference "bad" values -- see the design notes above
V_SCALE = 1e-3        # a 1 mp.u. band excursion = one unit of violation
CHATTER_SCALE = 2e-3  # |d2Q| of 0.2% of range per step^2 = one unit of chatter

# Term weights. Because each term is normalized, these read as actual trade-offs.
# W_VIOLATION was 8.0 when the violation mean ran over the 27 inverter buses
# only; the mean now runs over all ~2700 bus-phases, most of which are healthy,
# which dilutes the term by roughly the bus ratio (~100x). 800 restores the
# old balance; re-check with the printed breakdowns whenever it is retuned.
W_VIOLATION = 8.0
W_EFFORT    = 1.0
W_CHATTER   = 2.0

# Objective returned for a trial whose surrogate blew up numerically.
FAILURE_PENALTY = 1e9

TERM_NAMES = ('violation', 'effort', 'chatter')


# ---------------------------------------------------------------------------
# Shared primitive terms
#
# The single source of truth for "how bad is this voltage" and "how much
# reactive power is this using". BOTH the centralized controller's per-step
# objective and the gain-tuning objective are built from these two sums; they
# differ only in weighting and aggregation.
# ---------------------------------------------------------------------------
def violation_core(V, v_low=V_LOW, v_high=V_HIGH, axis=None):
    """
    Sum of squared band excursions with the 5/3 over/under asymmetry, raw
    p.u.^2 (no normalization). Elementwise over any V shape; `axis` as np.sum.

    Written with explicit out= temporaries rather than the equivalent
    `(W_OVER * np.maximum(0, V - v_high)) ** 2 + ...` one-liner: this is the
    hottest line in the droop/PID surrogate replay (called once per candidate
    per step), and the one-liner allocates ~8 fresh (n_candidates, 3n_bus)
    arrays per call versus 2 here -- profiled ~1.7x faster on the droop
    optimizer's actual candidate/bus-phase count, with identical output
    (over_dev/under_dev are fresh arrays from np.subtract, so V is never
    mutated).
    """
    over_dev = np.subtract(V, v_high)
    np.maximum(over_dev, 0.0, out=over_dev)
    np.multiply(over_dev, W_OVER, out=over_dev)
    np.square(over_dev, out=over_dev)

    under_dev = np.subtract(v_low, V)
    np.maximum(under_dev, 0.0, out=under_dev)
    np.multiply(under_dev, W_UNDER, out=under_dev)
    np.square(under_dev, out=under_dev)

    over_dev += under_dev
    return np.sum(over_dev, axis=axis)


def effort_core(Q, s_limit, axis=None):
    """Sum of squared reactive-power utilization (Q / S_max)^2, dimensionless."""
    return np.sum((Q / s_limit) ** 2, axis=axis)


def objective_terms(V_est: np.ndarray, Q: np.ndarray, s_limit: float) -> Dict[str, float]:
    """
    The three normalized, dimensionless tuning components.

    V_est : (n_steps, 3n) all-bus voltage trajectory from a surrogate replay.
    Q     : (n_steps, n_inverters) reactive-power trajectory.
    """
    return objective_terms_precomputed(
        float(violation_core(V_est)), V_est.size, Q, s_limit)


def objective_terms_precomputed(violation_sum, n_v_entries: int,
                                Q: np.ndarray, s_limit) -> Dict[str, float]:
    """
    objective_terms for callers that already hold the raw violation sum --
    the batched droop replay accumulates violation_core step-by-step instead
    of materializing the whole (steps x candidates x 3n) voltage array.
    """
    violation = violation_sum / n_v_entries / V_SCALE ** 2

    effort = effort_core(Q, s_limit) / Q.size

    if Q.shape[0] > 2:
        chatter = np.mean((np.diff(Q, n=2, axis=0) / s_limit) ** 2) / CHATTER_SCALE ** 2
    else:
        chatter = 0.0

    return {'violation': violation, 'effort': effort, 'chatter': chatter}


def objective_value(terms: Dict[str, float]) -> float:
    """Weighted sum of the three terms."""
    return (W_VIOLATION * terms['violation'] +
            W_EFFORT    * terms['effort'] +
            W_CHATTER   * terms['chatter'])


def score(V_est, Q, s_limit) -> float:
    """Convenience: full objective for one surrogate replay, or FAILURE_PENALTY."""
    if V_est is None:
        return FAILURE_PENALTY
    return objective_value(objective_terms(V_est, Q, s_limit))


def objective_terms_batch(violation_sums: np.ndarray, n_v_entries: int,
                          Q: np.ndarray, s_limit: float) -> Dict[str, np.ndarray]:
    """
    Batched counterpart of objective_terms: scores an entire population of
    candidate replays in one call instead of one candidate at a time, so a
    Differential Evolution generation can be evaluated as a single vectorized
    objective call (see scipy's `vectorized=True`) rather than once per
    candidate -- the Python-level loop overhead is paid once per generation
    instead of once per population member.

    violation_sums : (n_candidates,) raw violation_core sums, accumulated
        step-by-step by the batched replay. The all-bus voltage trajectory of
        a whole population (steps x candidates x 3n bus-phases) would be
        gigabytes, so it is never materialized -- only this running sum is.
    n_v_entries : number of voltage entries behind each sum
        (n_steps * 3n bus-phases), the mean-normalization denominator.
    Q : (n_steps, n_candidates, n_inverters); the middle axis is the
        population/candidate index. Returns one (n_candidates,) array per term.
    """
    violation = violation_sums / n_v_entries / V_SCALE ** 2

    effort = np.mean((Q / s_limit) ** 2, axis=(0, 2))

    if Q.shape[0] > 2:
        chatter = np.mean((np.diff(Q, n=2, axis=0) / s_limit) ** 2, axis=(0, 2)) / CHATTER_SCALE ** 2
    else:
        chatter = np.zeros(Q.shape[1])

    return {'violation': violation, 'effort': effort, 'chatter': chatter}


def objective_value_batch(terms: Dict[str, np.ndarray]) -> np.ndarray:
    """Batched counterpart of objective_value: one weighted sum per candidate."""
    return (W_VIOLATION * terms['violation'] +
            W_EFFORT    * terms['effort'] +
            W_CHATTER   * terms['chatter'])


def score_batch(violation_sums: np.ndarray, n_v_entries: int,
                Q: np.ndarray, s_limit: float,
                valid_mask: np.ndarray = None) -> np.ndarray:
    """
    Batched counterpart of score(): one objective value per candidate.

    violation_sums / n_v_entries: see objective_terms_batch.
    valid_mask, if given, is a boolean array of shape (n_candidates,).
    Candidates flagged False (e.g. a trajectory that went non-finite) get
    FAILURE_PENALTY regardless of what their (possibly NaN/garbage) terms
    computed to -- mirrors score()'s `V_est is None -> FAILURE_PENALTY` for
    the single-candidate case, which has no meaning per-element in a batch.
    """
    terms = objective_terms_batch(violation_sums, n_v_entries, Q, s_limit)
    values = objective_value_batch(terms)
    if valid_mask is not None:
        values = np.where(valid_mask, values, FAILURE_PENALTY)
    return values


def format_breakdown(terms: Dict[str, float]) -> str:
    """Human-readable weighted breakdown, for the optimizers' final report."""
    weights = {'violation': W_VIOLATION, 'effort': W_EFFORT, 'chatter': W_CHATTER}
    lines = ["Objective breakdown (weighted):"]
    for name in TERM_NAMES:
        lines.append(f"  {name:<11s} = {weights[name] * terms[name]:.4f}")
    lines.append(f"  {'TOTAL':<11s} = {objective_value(terms):.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Centralized controller objective
#
# The single-step objective LinearVoltVarController minimizes online with
# L-BFGS-B. It is built from the SAME primitive terms as the tuning objective
# above (violation_core / effort_core: same band, same 5/3 asymmetry, same
# (Q/S_max)^2 effort); it differs only in aggregation and weighting -- one
# instant instead of a day-long trajectory, a hard violation priority instead
# of normalized trade-offs, and no chatter term because a memoryless per-step
# problem has no trajectory to smooth.
# ---------------------------------------------------------------------------
# Hard weight on any voltage-band violation, relative to the Q-effort term.
# Calibrated so the marginal priority matches the pre-unification objective
# (raw 1e6 * excursion^2 vs raw Q^2 in MVar^2): there, a 1 mp.u. under-voltage
# cost 1.0 while one inverter at full Q cost (7.68e-3)^2 ~ 5.9e-5, a ratio of
# ~17,000. With the shared cores, that same excursion costs
# W * (3e-3)^2 = W * 9e-6 and full utilization costs 1.0, so W = 2e9 restores
# the ratio. Any violation still outweighs any plausible effort by orders of
# magnitude ("exact penalty"), and effort remains a pure tie-breaker.
W_CENTRAL_VIOLATION = 2e9
W_CENTRAL_EFFORT = 1.0


def central_objective(dq, v_current, S_Q, q_current, s_max, v_min=V_LOW, v_max=V_HIGH):
    """
    Per-step objective for the centralized Volt/Var controller.

    dq        : candidate change in Q for every inverter.
    v_current : current all-bus voltages (flat [va, vb, vc] over every bus).
    S_Q       : voltage sensitivity dV/dQ, shape (len(v_current), len(dq)).
    q_current : current Q of every inverter.
    s_max     : per-inverter apparent-power ceiling S_max(V), same shape as dq.

    Estimates the resulting voltages, penalizes any excursion outside
    [v_min, v_max] heavily (W_CENTRAL_VIOLATION, shared violation_core with
    the 5/3 over/under asymmetry), and adds a light utilization term (shared
    effort_core) so the optimizer prefers the smallest reactive injection
    that keeps the voltages in band.
    """
    v_est = v_current + S_Q @ dq
    return (W_CENTRAL_VIOLATION * violation_core(v_est, v_low=v_min, v_high=v_max) +
            W_CENTRAL_EFFORT * effort_core(q_current + dq, s_max))


# ---------------------------------------------------------------------------
# Reporting metric (NOT part of any objective)
# ---------------------------------------------------------------------------
def voltage_unbalance_factor(vm_a, va_a_deg, vm_b, va_b_deg, vm_c, va_c_deg):
    """
    EN 50160 Voltage Unbalance Factor [%] = |V2| / |V1| * 100, via symmetrical
    components, where V1/V2 are the positive-/negative-sequence voltages.

    Elementwise on scalars or arrays of per-phase magnitude (p.u.) and angle
    (degrees). This is the documented equivalent of pandapower's
    res_bus_3ph.unbalance_percent column (they agree to ~1e-14); the simulation
    logs that built-in column directly, and this helper exists for documentation
    and as an independent cross-check.
    """
    a = np.exp(1j * 2 * np.pi / 3)
    Va = vm_a * np.exp(1j * np.deg2rad(va_a_deg))
    Vb = vm_b * np.exp(1j * np.deg2rad(va_b_deg))
    Vc = vm_c * np.exp(1j * np.deg2rad(va_c_deg))
    V1 = (Va + a * Vb + a * a * Vc) / 3.0
    V2 = (Va + a * a * Vb + a * Vc) / 3.0
    return np.abs(V2) / np.abs(V1) * 100.0


# ---------------------------------------------------------------------------
# Optimizer reporting
#
# Both gain optimizers (pid_optimizer_approx.py, droop_optimizer_approx.py) can
# run either global search and, with --optimizer both, run each and compare. The
# search spaces differ entirely, but the comparison table they print is the same
# table, so it lives here rather than being duplicated in each.
# ---------------------------------------------------------------------------
LABELS = {'de': 'DE', 'cma': 'CMA-ES'}


def print_comparison_table(results: dict):
    """DE-vs-CMA-ES comparison: pre-/post-refinement objective and wall clock."""
    print("\n" + "-" * 62)
    print(f"  {'method':<8s} {'pre-refine':>14s} {'post-refine':>14s} {'time (s)':>10s}")
    print("  " + "-" * 60)
    for m in ('de', 'cma'):
        if m in results:
            r = results[m]
            print(f"  {LABELS[m]:<8s} {r['pre']:>14.4f} {r['post']:>14.4f} {r['seconds']:>10.1f}")
    print("-" * 62)
