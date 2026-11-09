
import os
from pandapower.control.basic_controller import Controller
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import sim_config as cfg
import surrogate_objective as so
import sensitivity

# ---------------------------------------------------------------------------
# Inverter current limit
#
# The binding constraint on a charger is its rated CURRENT, not a fixed
# apparent power: the semiconductors and the connecting conductor are sized for
# a maximum |I|. Because S = V * I, the apparent-power headroom is therefore
# voltage dependent, S_max(V) = V * I_max, and shrinks under-voltage — exactly
# when reactive support is most needed (thesis Ch. 2, Eq. for Q_max(P, V)).
#
# The chargers here are single-phase, connected phase-to-neutral on the IEEE
# European LV feeder, whose nominal phase-to-neutral voltage is
# 0.416 kV / sqrt(3) = 0.240 kV. The two constants live in sim_config.py (the
# single source of simulation defaults) and are re-exported here so the 32 A
# limit is defined in exactly one place and the optimizer surrogates can reuse
# the identical S_max(V) law.
# ---------------------------------------------------------------------------
I_MAX_A = cfg.I_MAX_A                     # rated charger/inverter current [A]
V_LN_NOMINAL_KV = cfg.V_LN_NOMINAL_KV     # phase-to-neutral base voltage [kV]
MAX_P_CURTAIL_FRAC = cfg.MAX_P_CURTAIL_FRAC  # max share of P an inverter may curtail for Q


def s_max_from_current(vm_pu, i_max_a=I_MAX_A, v_ln_nominal_kv=V_LN_NOMINAL_KV):
    """
    Voltage-dependent apparent-power ceiling [MVA] for a fixed current limit:

        S_max(V) = V * I_max,   with  V = vm_pu * V_ln_nominal.

    Returned in MVA so it compares directly with the P/Q setpoints (MW/MVar).
    Works elementwise on a scalar or an array of per-unit voltages.
    """
    v_kv = vm_pu * v_ln_nominal_kv          # kV, phase-to-neutral
    return v_kv * i_max_a * 1e-3            # kV * A = kVA  ->  * 1e-3 = MVA


def required_current(p_mw, q_mvar, vm_pu, v_ln_nominal_kv=V_LN_NOMINAL_KV):
    """
    Current [A] a charger draws to deliver apparent power sqrt(P^2 + Q^2) at the
    connection voltage V = vm_pu * V_ln_nominal. Elementwise; the counterpart of
    s_max_from_current used to decide whether a requested (P, Q) is feasible.
    """
    s_mva = np.sqrt(p_mw ** 2 + q_mvar ** 2)
    v_kv = vm_pu * v_ln_nominal_kv
    return s_mva / v_kv * 1e3               # MVA / kV = kA  ->  * 1e3 = A


def apply_current_limit(p_sched, q_req, s_max, max_curtail_frac=MAX_P_CURTAIL_FRAC):
    """Enforce the inverter current limit sqrt(P^2 + Q^2) <= S_max(V), curtailing
    ACTIVE power by at most `max_curtail_frac` of its scheduled magnitude before
    the reactive setpoint has to give way.

    Priority (bounded reactive-power priority):
      1. If (p_sched, q_req) already fits under S_max, keep both untouched.
      2. Otherwise curtail P toward the value that just fits q_req, but never
         below the floor (1 - max_curtail_frac) * |p_sched|.
      3. If that P floor still leaves the pair over S_max, reduce |Q| to
         sqrt(S_max^2 - p_floor^2) so the current limit is met.

    max_curtail_frac = 1.0 reproduces the original strategy exactly (P curtailed
    all the way to sqrt(S_max^2 - Q^2), Q never reduced); 0.0 never curtails P
    (Q is fully sacrificed). Vectorized: p_sched/q_req/s_max may be scalars or
    equal-length arrays. Returns (p_new, q_new) with the input signs preserved.
    """
    p_sched = np.asarray(p_sched, dtype=float)
    q_req = np.asarray(q_req, dtype=float)
    s_max = np.asarray(s_max, dtype=float)

    p_mag, q_mag = np.abs(p_sched), np.abs(q_req)
    p_floor = p_mag * (1.0 - max_curtail_frac)          # P may not drop below this
    over = (p_mag ** 2 + q_mag ** 2) > (s_max ** 2)      # current-limit violated?

    # P that would just fit the requested Q, floored by the curtailment cap.
    p_fit = np.sqrt(np.maximum(0.0, s_max ** 2 - q_mag ** 2))
    p_new = np.where(over, np.maximum(p_fit, p_floor), p_mag)

    # Where the floor binds (P can't be cut enough), pull Q down to fit the floor.
    floor_binds = over & (p_fit < p_floor)
    q_fit = np.sqrt(np.maximum(0.0, s_max ** 2 - p_floor ** 2))
    q_new = np.where(floor_binds, q_fit, q_mag)

    return np.sign(p_sched) * p_new, np.sign(q_req) * q_new


# ---------------------------------------------------------------------------
# Reactive-power supply mode
#
# Every controller below can either support the grid at all times (even with
# no vehicle plugged in) or only while an EV is plugged in — the session's
# connection window, carried per charger and per step by the plugged_in
# column of the profile data (see preprocess_charging_records.py). A plugged
# vehicle that has finished charging (P == 0) is NORMAL operation: the
# inverter keeps its full S_max available for Q. This is a single index
# shared by all three controllers, set from
# network_setup.simulation_with_dynamic_gen and, ultimately, run.bat's
# Q_MODE config value.
# ---------------------------------------------------------------------------
Q_MODE_ALWAYS = 0        # supply Q regardless of connection state
Q_MODE_PLUGGED_ONLY = 1  # supply Q only while an EV is plugged in


# ---------------------------------------------------------------------------
# Shared control laws
#
# These pure functions are the single source of truth for how voltage maps to
# a reactive-power setpoint. The pandapower controller classes below call them
# with live network voltages, and the surrogate models in
# pid_optimizer_approx.py / droop_optimizer_approx.py call them with estimated
# voltages — so the optimizers tune exactly the equation the simulation runs.
# ---------------------------------------------------------------------------

def pid_q_update(v, integral, prev_v, q_prev, Kp, Ki, Kd, q_decay, s_limit,
                 v_low=cfg.V_LOW, v_high=cfg.V_HIGH, q_rate=None, hysteresis=0.0):
    """
    One discrete PID step for a vector of inverters.

    The plant here is an algebraic gain (dV/dQ) plus exactly one step of
    feedback delay: pandapower calls time_step() before the power flow, so the
    voltages read are the previous step's. Three details keep that loop from
    limit-cycling, and all three matter:

    1. Back-calculation anti-windup. Subtracting the last increment from the
       integral when the output saturates ("conditional integration") cancels
       the increment exactly, so with a large Ki the integral never leaves zero
       and the loop degenerates into a relay. Feeding the saturation residual
       (q - q_unsat)/Ki back instead keeps the integral consistent with the Q
       actually applied, and bounds the I term to +/- s_limit.
    2. Derivative on the measurement, not on the error. The error steps
       discontinuously from 0 the instant v crosses a deadband edge, so
       (error - prev_error) delivers a kick outward on exit and an equal kick
       inward on re-entry — a self-sustaining oscillation at the band edge.
       -(v - prev_v) is continuous across the edge.
    3. Slew limit (q_rate) and deadband hysteresis. q_rate caps |dQ| per step,
       holding the actuator bandwidth below what the one-step delay can
       tolerate. hysteresis shrinks the band in which Q is allowed to bleed
       away, so the decay cannot walk the voltage straight back into violation.

    Parameters
    ----------
    v, integral, prev_v, q_prev : arrays of shape (n_inverters,)
        Live voltages and the three state arrays. None are mutated.
    q_rate : float or None
        Maximum change in Q per time step (MVar). None disables slew limiting.
    hysteresis : float
        Deadband shrink (p.u.) applied to the integral-decay region only.
        0.0 reproduces "decay everywhere inside [v_low, v_high]".

    Returns
    -------
    (q_output, integral, v) — the caller keeps `integral` and stores `v` as the
    next step's `prev_v` and `q_output` as the next step's `q_prev`.
    """
    error = np.zeros_like(v)

    # Over-voltage: error is negative -> negative Q (absorb reactive power)
    over_mask = v > v_high
    error[over_mask] = v_high - v[over_mask]

    # Under-voltage: error is positive -> positive Q (inject reactive power)
    under_mask = v < v_low
    error[under_mask] = v_low - v[under_mask]

    # Q-minimization: decay the integral of inverters that are comfortably
    # inside the deadband so their Q output gradually returns to zero. The
    # hysteresis margin keeps an inverter that is only just back inside the
    # band from immediately bleeding off the Q that put it there.
    integral = integral.copy()
    if q_decay < 1.0:
        relaxed_mask = (v >= v_low + hysteresis) & (v <= v_high - hysteresis)
        integral[relaxed_mask] *= q_decay

    integral += error

    # Derivative on measurement (see note 2 above); zero inside the deadband,
    # where there is no error to differentiate.
    active_mask = over_mask | under_mask
    derivative = np.where(active_mask, -(v - prev_v), 0.0)

    q_unsat = (Kp * error) + (Ki * integral) + (Kd * derivative)

    # Output clipping, then slew limiting against the previous applied Q
    q_output = np.clip(q_unsat, -s_limit, s_limit)
    if q_rate is not None:
        q_output = np.clip(q_output, q_prev - q_rate, q_prev + q_rate)

    # Back-calculation anti-windup (see note 1 above). The clip bounds the I
    # term itself to +/- s_limit, which also covers Ki small enough that the
    # residual division would otherwise blow up.
    if Ki > 1e-9:
        integral += (q_output - q_unsat) / Ki
        i_limit = s_limit / Ki
        np.clip(integral, -i_limit, i_limit, out=integral)

    return q_output, integral, v.copy()


def droop_q_update(v, droopgain_pos, droopgain_neg, q_limit,
                   v_low=cfg.V_LOW, v_high=cfg.V_HIGH):
    """
    Volt/Var droop law: linear Q response outside the deadband, clipped to the
    inverter limit. Works elementwise on scalars or arrays.
    """
    q = np.where(v < v_low, (v_low - v) * droopgain_neg,
        np.where(v > v_high, -(v - v_high) * droopgain_pos, 0.0))
    return np.clip(q, -q_limit, q_limit)


class AsymmetricDynamicDroopController(Controller):
    """
    A custom pandapower controller that dynamically adjusts the output of an
    asymmetric static generator based on the current load on a specific phase of a target bus.
    Now supports individual positive/negative custom droop gains per inverter.
    """
    def __init__(self, net, sgen_id, target_bus_id, phase, target_low_voltage=cfg.V_LOW,
                 target_high_voltage=cfg.V_HIGH, i_max_a=I_MAX_A, v_ln_kv=V_LN_NOMINAL_KV, droopgain=0,
                 custom_droop_gains=None, q_mode=Q_MODE_ALWAYS, plugged=None,
                 max_curtail_frac=MAX_P_CURTAIL_FRAC, **kwargs):

        # **kwargs holds pandapower controller args (e.g. order, level)
        super().__init__(net, **kwargs)

        self.sgen_id = sgen_id
        self.target_bus_id = target_bus_id
        self.phase = phase.lower() # ensure 'a', 'b', or 'c'
        self.target_low_voltage = target_low_voltage
        self.target_high_voltage = target_high_voltage
        # Charger current rating; the apparent-power ceiling is derived per step
        # from the live voltage as S_max(V) = V * i_max_a (see s_max_from_current).
        self.i_max_a = i_max_a
        self.v_ln_kv = v_ln_kv
        # Cap on how much active power may be curtailed for reactive support.
        self.max_curtail_frac = max_curtail_frac
        # Q_MODE_ALWAYS or Q_MODE_PLUGGED_ONLY (see the module-level constants)
        self.q_mode = q_mode
        # Plugged-in mask for THIS inverter: 1-D bool, one entry per simulation
        # step (the plugged_in profile column), indexed by time_step's time arg.
        if q_mode == Q_MODE_PLUGGED_ONLY:
            if plugged is None:
                raise ValueError(
                    "q_mode=Q_MODE_PLUGGED_ONLY requires the per-step plugged "
                    "mask (plugged_in profile column)")
            self.plugged = np.asarray(plugged, dtype=bool)
        else:
            self.plugged = None

        # Per-inverter custom droop gains, with a uniform fallback
        # Check if we have custom gains for this specific sgen_id
        if custom_droop_gains is not None and sgen_id in custom_droop_gains:
            self.droopgain_pos = custom_droop_gains[sgen_id].get('pos', droopgain)
            self.droopgain_neg = custom_droop_gains[sgen_id].get('neg', droopgain)
        else:
            # Fallback to the uniform default gain if no custom data exists
            self.droopgain_pos = droopgain
            self.droopgain_neg = droopgain

    def is_converged(self, net):
        return True

    def time_step(self, net, time):
        # 1. Read the current voltage of the target bus for the specific phase
        current_vm_pu = net.res_bus_3ph.at[self.target_bus_id, f'vm_{self.phase}_pu']

        # 2. Voltage-dependent apparent-power ceiling for the 32 A current limit
        s_max = s_max_from_current(current_vm_pu, self.i_max_a, self.v_ln_kv)

        # 3. Shared droop law (same equation the optimizer surrogate uses), with
        # Q clipped to the inverter's current-limited headroom S_max(V).
        optimal_power_mw = float(droop_q_update(
            current_vm_pu, self.droopgain_pos, self.droopgain_neg, s_max,
            v_low=self.target_low_voltage, v_high=self.target_high_voltage))

        # 4. Plugged-only mode: with no vehicle connected the charger supplies
        # no Q. Droop is memoryless, so there is no internal state to reset --
        # forcing the output to 0 here is the whole story. (A plugged vehicle
        # that has finished charging, P == 0, keeps supporting the grid.)
        if self.q_mode == Q_MODE_PLUGGED_ONLY and not self.plugged[time]:
            optimal_power_mw = 0.0
        p_current = net.asymmetric_sgen.at[self.sgen_id, f'p_{self.phase}_mw']

        # 5. Plant-side current limit with a bounded curtailment cap: P is
        # curtailed for grid support by at most max_curtail_frac of its
        # scheduled value; once that cap is hit, Q is reduced instead
        # (max_curtail_frac=1.0 = old behaviour, P curtailed freely).
        p_new, q_new = apply_current_limit(p_current, optimal_power_mw, s_max,
                                           self.max_curtail_frac)
        net.asymmetric_sgen.at[self.sgen_id, f'q_{self.phase}_mvar'] = float(q_new)
        net.asymmetric_sgen.at[self.sgen_id, f'p_{self.phase}_mw'] = float(p_new)

class VectorizedAsymmetricPIDController(Controller):
    """
    A highly optimized, vectorized PID controller that adjusts reactive power
    for MULTIPLE asymmetric generators on a specific phase simultaneously.

    The per-step control law lives in the module-level pid_q_update() so the
    optimizer surrogate provably runs the same equation.

    Parameters
    ----------
    v_high : float
        Upper voltage threshold in p.u. (default 1.05)
    v_low : float
        Lower voltage threshold in p.u. (default 0.97)
    q_decay : float
        Integral decay factor applied each time step when a generator's voltage
        is inside the deadband [v_low, v_high].  Range (0, 1].
        - 1.0  → old behaviour (no decay, Q holds its last value)
        - 0.95 → gentle decay (Q bleeds to ~10% in ~45 steps)
        - 0.80 → fast decay   (Q bleeds to ~10% in ~10 steps)
        Default: 0.95
    q_rate : float or None
        Slew limit on Q, in MVar per time step. Caps how fast the inverter can
        traverse its range and is the most robust guard against limit cycling
        against the loop's one-step feedback delay. None = unlimited.
    hysteresis : float
        Margin (p.u.) by which the integral-decay band is narrower than the
        control deadband. Default 0.0 = decay anywhere inside the deadband.
    """
    def __init__(self, net, sgen_ids, target_bus_ids, phase, Kp=0.02, Ki=0.09, Kd=0.002,
                 i_max_a=I_MAX_A, v_ln_kv=V_LN_NOMINAL_KV, q_mode=Q_MODE_ALWAYS,
                 v_high=cfg.V_HIGH, v_low=cfg.V_LOW, q_decay=0.93, q_rate=None, hysteresis=0.0,
                 plugged=None, max_curtail_frac=MAX_P_CURTAIL_FRAC, **kwargs):
        super().__init__(net, **kwargs)
        self.sgen_ids = sgen_ids
        self.target_bus_ids = target_bus_ids
        self.phase = phase.lower()
        self.n_gens = len(sgen_ids)

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        # Charger current rating; the apparent-power ceiling used to clip Q and
        # to curtail P is derived per step from the live voltage as
        # S_max(V) = V * i_max_a (see s_max_from_current).
        self.i_max_a = i_max_a
        self.v_ln_kv = v_ln_kv
        # Cap on how much active power may be curtailed for reactive support.
        self.max_curtail_frac = max_curtail_frac
        # Q_MODE_ALWAYS or Q_MODE_PLUGGED_ONLY (see the module-level constants)
        self.q_mode = q_mode
        # Plugged-in mask, shape (n_steps, n_gens) column-aligned with
        # sgen_ids (the plugged_in profile columns), indexed by time_step's
        # time arg.
        if q_mode == Q_MODE_PLUGGED_ONLY:
            if plugged is None:
                raise ValueError(
                    "q_mode=Q_MODE_PLUGGED_ONLY requires the per-step plugged "
                    "mask (plugged_in profile columns)")
            self.plugged = np.asarray(plugged, dtype=bool)
        else:
            self.plugged = None

        # Q-minimization: decay factor applied to the integral of each generator
        # that is currently inside the voltage deadband.
        self.q_decay = q_decay

        # Oscillation guards (see pid_q_update)
        self.q_rate = q_rate
        self.hysteresis = hysteresis

        # Voltage thresholds
        self.v_high = v_high
        self.v_low  = v_low

        # Vectorized state arrays (one entry per controlled generator).
        # prev_v is seeded on the first time step from the real voltages, so the
        # derivative term does not see a spurious step out of zero.
        self.integral = np.zeros(self.n_gens)
        self.prev_v   = None
        self.q_prev   = np.zeros(self.n_gens)

        # Cache integer positions for fast array look-ups
        self.sgen_idx = net.asymmetric_sgen.index.get_indexer(sgen_ids)
        self.bus_idx  = net.res_bus_3ph.index.get_indexer(target_bus_ids)
        self.v_col = f'vm_{self.phase}_pu'
        self.q_col = f'q_{self.phase}_mvar'
        self.p_col = f'p_{self.phase}_mw'

    def is_converged(self, net):
        return True

    def time_step(self, net, time):
        # 1. Read all voltages in a single memory grab
        current_vm_pu = net.res_bus_3ph[self.v_col].values[self.bus_idx]

        if self.prev_v is None:
            self.prev_v = current_vm_pu.copy()

        # 2. Voltage-dependent apparent-power ceiling for the 32 A current limit,
        # one entry per inverter from that inverter's live connection voltage.
        s_limit = s_max_from_current(current_vm_pu, self.i_max_a, self.v_ln_kv)

        # 3. Shared PID law (same equation the optimizer surrogate uses); Q is
        # clipped inside the law to +/- s_limit = S_max(V).
        q_output, self.integral, self.prev_v = pid_q_update(
            current_vm_pu, self.integral, self.prev_v, self.q_prev,
            self.Kp, self.Ki, self.Kd, self.q_decay, s_limit,
            v_low=self.v_low, v_high=self.v_high,
            q_rate=self.q_rate, hysteresis=self.hysteresis)

        # 4. Plugged-only mode: inverters without a connected vehicle supply
        # no Q. Their integral is reset too, so unplugged time never leaves a
        # stale windup that would spike Q the instant a vehicle plugs back in.
        # A plugged vehicle at P == 0 (finished charging) is normal operation:
        # the integrator keeps its state and Q stays live.
        p_current = net.asymmetric_sgen[self.p_col].values[self.sgen_idx].copy()
        if self.q_mode == Q_MODE_PLUGGED_ONLY:
            unplugged = ~self.plugged[time]
            q_output[unplugged] = 0.0
            self.integral[unplugged] = 0.0

        self.q_prev = q_output

        # 5. Plant-side current limit with a bounded curtailment cap (network
        # effect, not part of the Q law): P is curtailed for grid support by at
        # most max_curtail_frac of its scheduled value; once that cap binds, the
        # applied Q is reduced instead of curtailing P further. The PID law state
        # (integral, q_prev) tracks the unclipped q_output above, so the tuned
        # dynamics are unchanged; only the setpoints written respect the cap.
        # max_curtail_frac=1.0 reproduces the old free-curtailment behaviour.
        p_new, q_new = apply_current_limit(p_current, q_output, s_limit, self.max_curtail_frac)
        net.asymmetric_sgen.loc[self.sgen_ids, self.p_col] = p_new
        net.asymmetric_sgen.loc[self.sgen_ids, self.q_col] = q_new

class LinearVoltVarController(Controller):
    """
    Centralized Volt/Var controller. Estimates the 3-phase voltage sensitivity
    matrix (dV/dQ) via perturbation and re-estimates it every
    cfg.SENS_REESTIMATE_HOURS of simulated time (holding it constant in
    between), then solves a bounded optimization (L-BFGS-B) at every step to
    find near-optimal Q setpoints for all inverters simultaneously. The
    matrix construction, cadence and hold are identical to the S_QQ anchors
    the PID/droop gain optimizers replay against (sensitivity.py /
    network_setup.calculate_time_varying_sensitivity).
    """
    def __init__(self, net, sgen_ids, savingDest, i_max_a=I_MAX_A, v_ln_kv=V_LN_NOMINAL_KV,
                 q_mode=Q_MODE_ALWAYS, v_min=cfg.V_LOW, v_max=cfg.V_HIGH, saving_voltages=False,
                 step_mult=cfg.STEP_MULT, plugged=None, max_curtail_frac=MAX_P_CURTAIL_FRAC, **kwargs):
        super().__init__(net, **kwargs)
        self.sgen_ids = sgen_ids
        # Charger current rating; each inverter's apparent-power ceiling is
        # derived per step from its live voltage as S_max(V) = V * i_max_a.
        self.i_max_a = i_max_a
        self.v_ln_kv = v_ln_kv
        # Cap on how much active power may be curtailed for reactive support.
        self.max_curtail_frac = max_curtail_frac
        # Q_MODE_ALWAYS or Q_MODE_PLUGGED_ONLY (see the module-level constants)
        self.q_mode = q_mode
        # Plugged-in mask, shape (n_steps, n_inv) column-aligned with sgen_ids
        # (the plugged_in profile columns); the current step is remembered by
        # time_step for use inside control_step.
        if q_mode == Q_MODE_PLUGGED_ONLY:
            if plugged is None:
                raise ValueError(
                    "q_mode=Q_MODE_PLUGGED_ONLY requires the per-step plugged "
                    "mask (plugged_in profile columns)")
            self.plugged = np.asarray(plugged, dtype=bool)
        else:
            self.plugged = None
        self._step_time = 0
        self.v_min = v_min
        self.v_max = v_max
        self.saving_voltages = saving_voltages
        self.applied = False
        self.savingDest = savingDest
        self.sgen_idx = net.asymmetric_sgen.index.get_indexer(self.sgen_ids)
        self.phases = np.array([net.asymmetric_sgen.loc[sid, 'phi'].lower() for sid in self.sgen_ids])
        # Connection-bus id per inverter, for reading its live per-unit voltage
        self.bus_ids = net.asymmetric_sgen['bus'].values[self.sgen_idx]
        self.step = 0
        # S_Q is estimated lazily at the first time step and re-estimated
        # every cfg.SENS_REESTIMATE_HOURS of simulated time, so every anchor
        # is taken with that instant's load/generation profiles applied --
        # exactly like the gain optimizers' surrogate anchors.
        self.S_Q = None
        self.reestimate_every = cfg.sens_anchor_steps(step_mult)
        # Rows of the flat all-bus voltage vector for each inverter's own
        # connection bus and phase (used by the optional saving path).
        self.v_rows = sensitivity.voltage_row_indices(net, self.bus_ids, self.phases)

    def _get_all_voltages(self, net):
        """Return a flat array with the a, b, c voltages of every bus."""
        return sensitivity.all_bus_voltages(net)

    def time_step(self, net, time):
        # Periodic sensitivity re-estimation (zero-order hold in between).
        # This runs after the ConstControl profile updates (they are order=0,
        # this controller is order=1), so the matrix is linearized around the
        # instant's true operating point -- the same convention, solver
        # options and shared routine as the surrogate anchors. strict=False:
        # a non-converged perturbed flow yields a zero column instead of
        # aborting the run.
        if self.S_Q is None or self.step % self.reestimate_every == 0:
            self.S_Q, _ = sensitivity.compute_sensitivity_matrix(
                net, self.sgen_ids,
                run_kwargs={"lightsim2grid": True},
                base_init="flat" if self.S_Q is None else "results",
                strict=False)
        self._step_time = time
        self.applied = False

    def is_converged(self, net):
        return self.applied

    def control_step(self, net):
        # 1. Read the current Q and V values
        q_current = np.zeros(len(self.sgen_ids))
        for i, (idx, ph) in enumerate(zip(self.sgen_idx, self.phases)):
            q_current[i] = net.asymmetric_sgen[f'q_{ph}_mvar'].values[idx]

        v_current = self._get_all_voltages(net)

        # 2. Reactive-power priority under the 32 A current limit. Give Q the full
        # voltage-dependent headroom S_max(V) = V * I_max per inverter; the
        # charging power P is curtailed afterwards (step 5) so grid support wins.
        # L-BFGS-B respects bounds strictly, so it can never request an infeasible Q.
        #
        # Plugged-only mode: an inverter without a connected vehicle gets a
        # degenerate bound (-q_current[i], -q_current[i]), which forces
        # dq = -q_current[i] and therefore q_opt[i] = 0 exactly -- L-BFGS-B
        # has no other choice.
        s_max_i = np.array([
            s_max_from_current(net.res_bus_3ph.at[self.bus_ids[i], f'vm_{self.phases[i]}_pu'],
                               self.i_max_a, self.v_ln_kv)
            for i in range(len(self.sgen_ids))
        ])
        if self.q_mode == Q_MODE_PLUGGED_ONLY:
            unplugged = ~self.plugged[self._step_time]
        else:
            unplugged = np.zeros(len(self.sgen_ids), dtype=bool)
        bounds = []
        for i in range(len(self.sgen_ids)):
            if unplugged[i]:
                bounds.append((-q_current[i], -q_current[i]))
            else:
                q_max = s_max_i[i]
                min_dq = -q_max - q_current[i]
                max_dq = q_max - q_current[i]
                bounds.append((min_dq, max_dq))

        # 3. Objective: all-bus voltage-band penalty + Q effort. The objective
        # itself lives in surrogate_objective.central_objective so all three
        # controllers share one objective module (behaviour is unchanged).
        # 4. Solve with L-BFGS-B (fast and stable on bounded problems)
        dq_init = np.zeros(len(self.sgen_ids))

        res = minimize(
            lambda dq: so.central_objective(dq, v_current, self.S_Q, q_current, s_max_i,
                                            v_min=self.v_min, v_max=self.v_max),
            dq_init, method='L-BFGS-B', bounds=bounds)

        # 5. Apply the solution under the current limit with a bounded
        # curtailment cap (see apply_current_limit): P is curtailed for the
        # optimizer's Q by at most max_curtail_frac of its scheduled value; once
        # that cap binds, the applied Q is pulled below q_opt to meet the limit.
        # |q_opt| <= S_max(V) already (bounds above), so at max_curtail_frac=1.0
        # the cap never binds and this reproduces the old P-only curtailment.
        q_opt = q_current + res.x
        p_current = np.array([net.asymmetric_sgen.at[sid, f'p_{ph}_mw']
                              for sid, ph in zip(self.sgen_ids, self.phases)])
        p_new, q_new = apply_current_limit(p_current, q_opt, s_max_i, self.max_curtail_frac)
        for i, (sid, ph) in enumerate(zip(self.sgen_ids, self.phases)):
            net.asymmetric_sgen.at[sid, f'q_{ph}_mvar'] = q_new[i]
            net.asymmetric_sgen.at[sid, f'p_{ph}_mw'] = p_new[i]

        # 6. Optionally save the step results to CSV. v_current is the flat
        # all-bus [va, vb, vc] vector; v_rows picks each inverter's own
        # connection bus AND phase out of it.
        v_to_save = v_current[self.v_rows]

        if self.saving_voltages:
            df_results = pd.DataFrame({
                'id': self.sgen_ids,
                'phase': self.phases,
                'v_before': v_to_save,
                'q_opt': q_opt
            })
            csv_name=f"q_opt_step_{self.step}.csv"
            dest = os.path.join(self.savingDest, csv_name)
            if not os.path.exists(os.path.dirname(dest)):
                    os.makedirs(os.path.dirname(dest))
            # index=False drops pandas' automatic row numbering
            df_results.to_csv(dest, index=False)
        self.step += 1
        self.applied = True