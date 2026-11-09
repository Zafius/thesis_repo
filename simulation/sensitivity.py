"""
Single source of truth for the perturbation-based voltage sensitivity matrix.

Thesis Ch. 3 defines one estimator (Eq. sensitivity-perturb): perturb inverter
j's reactive power by a small known dq, re-run the unbalanced power flow, and
take the column S[:, j] = (V(Q_j + dq) - V(Q_j)) / dq. Two consumers previously
carried their own copy of that sweep:

* LinearVoltVarController (custom_controller_example.py) -- the all-bus S_Q
  the centralized optimizer uses online, re-estimated every
  cfg.SENS_REESTIMATE_HOURS with zero-order hold in between;
* network_setup.calculate_time_varying_sensitivity -- the anchor-instant
  matrices the PID/droop gain optimizers replay offline, on the SAME cadence
  and hold.

Both call compute_sensitivity_matrix below with the same solver options and
init convention, so the perturbation size, the voltage read-out, the column
construction and the re-estimation instants are defined in exactly one place --
the offline anchors and the online controller matrices are bit-identical for
the same operating point (verified by the unified-sensitivity test).
"""

import numpy as np
import pandapower as pp

import sim_config as cfg

# Reactive-power perturbation: 1 kVAr, the value quoted in thesis Ch. 3.
# Defined in sim_config.py, re-exported here.
DELTA_Q_MVAR = cfg.DELTA_Q_MVAR


def all_bus_voltages(net):
    """
    Flat (3 * n_buses,) voltage vector [va..., vb..., vc...] over every bus,
    in net.bus order. NaN entries (phases pandapower leaves unsolved) read as
    1.0 p.u. so they contribute zero sensitivity.
    """
    va = net.res_bus_3ph.vm_a_pu.fillna(1.0).values
    vb = net.res_bus_3ph.vm_b_pu.fillna(1.0).values
    vc = net.res_bus_3ph.vm_c_pu.fillna(1.0).values
    return np.concatenate([va, vb, vc])


def voltage_row_indices(net, buses, phases):
    """
    Rows of the all_bus_voltages vector for (bus, phase) pairs -- e.g. the
    connection bus and phase of each inverter. Slicing a sensitivity matrix
    with these rows yields the (n_inv x m) sub-matrix the gain-optimizer
    surrogates replay.
    """
    n_buses = len(net.bus)
    offset = {'a': 0, 'b': 1, 'c': 2}
    bus_pos = net.bus.index.get_indexer(buses)
    return np.array([offset[str(ph).lower()] * n_buses + p
                     for p, ph in zip(bus_pos, phases)], dtype=int)


def compute_sensitivity_matrix(net, sgen_ids, delta_q=DELTA_Q_MVAR,
                               run_kwargs=None, base_init=None, strict=True):
    """
    Perturbation estimate of S_Q = dV/dQ around the network's current
    operating point (thesis Ch. 3, Eq. sensitivity-perturb).

    Runs one base power flow plus one per inverter in sgen_ids, perturbing
    that inverter's own phase (the 'phi' column of net.asymmetric_sgen) by
    delta_q and restoring it afterwards. The net is left at the base
    operating point's Q values.

    Parameters
    ----------
    net : pandapower network with asymmetric sgens (mutated during the sweep,
        restored before returning).
    sgen_ids : ids into net.asymmetric_sgen -- one matrix column each.
    delta_q : perturbation in MVar (default 1 kVAr).
    run_kwargs : extra kwargs for pp.runpp_3ph (e.g. {"lightsim2grid": True}).
    base_init : 'init' for the base power flow; None uses pandapower's
        default. Perturbed flows always warm-start with init="results".
    strict : if False, a perturbed flow that fails to converge yields a zero
        column instead of raising (the anchor sweep's fallback). The base
        flow's LoadflowNotConverged always propagates -- without a base
        voltage there is nothing meaningful to return.

    Returns
    -------
    (S_Q, v_base) : S_Q of shape (3 * n_buses, len(sgen_ids)) over every bus
        and phase in all_bus_voltages order; v_base the base-flow voltages.
    """
    run_kwargs = dict(run_kwargs or {})
    phases = [str(net.asymmetric_sgen.at[sid, 'phi']).lower() for sid in sgen_ids]

    if base_init is None:
        pp.runpp_3ph(net, **run_kwargs)
    else:
        pp.runpp_3ph(net, init=base_init, **run_kwargs)
    v_base = all_bus_voltages(net)

    S_Q = np.zeros((len(v_base), len(sgen_ids)))
    for j, (sid, ph) in enumerate(zip(sgen_ids, phases)):
        q_col = f'q_{ph}_mvar'
        orig_q = net.asymmetric_sgen.at[sid, q_col]
        net.asymmetric_sgen.at[sid, q_col] = orig_q + delta_q
        try:
            pp.runpp_3ph(net, init="results", **run_kwargs)
            S_Q[:, j] = (all_bus_voltages(net) - v_base) / delta_q
        except pp.LoadflowNotConverged:
            if strict:
                raise
            S_Q[:, j] = 0.0
        finally:
            net.asymmetric_sgen.at[sid, q_col] = orig_q

    return S_Q, v_base
