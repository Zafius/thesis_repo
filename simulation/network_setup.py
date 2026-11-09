import os
import re
import copy
import time
import random
import warnings
import pandas as pd
import pandapower as pp
import pandapower.timeseries as ts
from pandapower.control import ConstControl
from pandapower.timeseries import DFData
import numpy as np
import scipy.sparse.linalg as _spla

import sim_config as cfg
from csv_to_plot import plot_simulation_results as psr
from csv_to_plot import plot_combined_phases as pcp
from custom_controller_example import VectorizedAsymmetricPIDController
from custom_controller_example import AsymmetricDynamicDroopController
from custom_controller_example import LinearVoltVarController
import surrogate_objective as so
import sensitivity


def _place_single_phase_sgens(net, loads, buses_with_loads, count, name_prefix, set_phi):
    """
    Place `count` single-phase asymmetric sgens on random load-serving buses
    and random active phases (uncoordinated installation). Shared by the EV
    and PV placement in setup_simulation; the np.random call sequence per unit
    (one bus choice, one phase choice) is identical for both, so the seeded
    placement streams are reproducible.

    set_phi: write the chosen phase into the sgen table's 'phi' column (used
    by the reactive-power controllers; EVs only).
    """
    indices, phases = [], []
    for i in range(count):
        chosen_bus = np.random.choice(buses_with_loads)
        loads_on_bus = loads[loads['bus'] == chosen_bus]

        available_phases = set()
        for _, load_row in loads_on_bus.iterrows():
            if load_row['p_a_mw'] > 0: available_phases.add('a')
            if load_row['p_b_mw'] > 0: available_phases.add('b')
            if load_row['p_c_mw'] > 0: available_phases.add('c')

        available_phases = list(available_phases)
        if not available_phases:
            available_phases = ['a', 'b', 'c']

        chosen_phase = np.random.choice(available_phases)

        sgen_idx = pp.create_asymmetric_sgen(net, chosen_bus, p_a_mw=0, p_b_mw=0, p_c_mw=0,
                                             name=f"{name_prefix}_{i}")
        if set_phi:
            net.asymmetric_sgen.loc[sgen_idx, 'phi'] = chosen_phase
        indices.append(sgen_idx)
        phases.append(chosen_phase)
    return indices, phases


def setup_simulation(net, seed=cfg.NETWORK_SEED, contribution=cfg.EV_CONTRIBUTION,
                     pv_contribution=cfg.PV_CONTRIBUTION):

    np.random.seed(seed)  # Fixed seed for reproducibility

    loads = net.asymmetric_load
    num_evs = int(len(loads) * contribution)
    buses_with_loads = loads['bus'].unique()

    ev_sgen_indices, ev_phases = _place_single_phase_sgens(
        net, loads, buses_with_loads, num_evs, "EV_Charger", set_phi=True)

    # Controller index lists: the EV chargers and their reactive-power
    # controllers target the same generators, split per phase.
    dyn_sgen_indices = list(ev_sgen_indices)
    dyn_sgen_indices_a = [s for s, ph in zip(ev_sgen_indices, ev_phases) if ph == 'a']
    dyn_sgen_indices_b = [s for s, ph in zip(ev_sgen_indices, ev_phases) if ph == 'b']
    dyn_sgen_indices_c = [s for s, ph in zip(ev_sgen_indices, ev_phases) if ph == 'c']

    # Add PV generation (separate seed stream so PV placement is independent of EVs)
    np.random.seed(seed + 1)
    num_pvs = int(len(loads) * pv_contribution)
    pv_sgen_indices, pv_phases = _place_single_phase_sgens(
        net, loads, buses_with_loads, num_pvs, "PV_Generator", set_phi=False)

    return net, ev_sgen_indices, ev_phases, pv_sgen_indices, pv_phases, dyn_sgen_indices, dyn_sgen_indices_a, dyn_sgen_indices_b, dyn_sgen_indices_c


def CostControl_setup(net, ds, ds1, variables):
    # ACTIVE POWER (P) Column names
    cols_p_a = [f"load_{i}_p_a" for i in net.asymmetric_load.index]
    cols_p_b = [f"load_{i}_p_b" for i in net.asymmetric_load.index]
    cols_p_c = [f"load_{i}_p_c" for i in net.asymmetric_load.index]
    cols_p=[cols_p_a, cols_p_b, cols_p_c]

    # REACTIVE POWER (Q) Column names
    cols_q_a = [f"load_{i}_q_a" for i in net.asymmetric_load.index]
    cols_q_b = [f"load_{i}_q_b" for i in net.asymmetric_load.index]
    cols_q_c = [f"load_{i}_q_c" for i in net.asymmetric_load.index]
    cols_q=[cols_q_a, cols_q_b, cols_q_c]
    ev_sgen_indices = variables[0]
    ev_phases = variables[1]
    pv_sgen_indices = variables[2]
    pv_phases = variables[3]
      
    for i, sgen_idx in enumerate(ev_sgen_indices):
        phase = ev_phases[i]
        controlled_variable = f"p_{phase}_mw"
        ConstControl(net, element='asymmetric_sgen', variable=controlled_variable, 
                     element_index=[sgen_idx], data_source=ds, profile_name=[f"ev_{sgen_idx}"])
                     
    # Attach a ConstControl for PV asymmetric sgens
    for i, sgen_idx in enumerate(pv_sgen_indices):
        phase = pv_phases[i]
        controlled_variable = f"p_{phase}_mw"
        ConstControl(net, element='asymmetric_sgen', variable=controlled_variable, 
                     element_index=[sgen_idx], data_source=ds, profile_name=[f"pv_{sgen_idx}"])

    # Map the ACTIVE & REACTIVE power controllers for loads
    phases=['a', 'b', 'c']
    cols_index=0
    for phase in phases:
        controlled_variable= f"p_{phase}_mw"
        ConstControl(net, element='asymmetric_load', variable=controlled_variable, element_index=net.asymmetric_load.index, data_source=ds1, profile_name=cols_p[cols_index])
        controlled_variable= f'q_{phase}_mvar'
        ConstControl(net, element='asymmetric_load', variable=controlled_variable, element_index=net.asymmetric_load.index, data_source=ds1, profile_name=cols_q[cols_index])
        cols_index+=1
    
    return net
def export_results(ow, time_index, script_dir, test_dir, test_name=None, control_type=None, net=None):
    phases=['a', 'b', 'c']
    global_voltages_path= os.path.join(script_dir, test_dir, 'global_voltages')
    os.makedirs(global_voltages_path, exist_ok=True)
    for phase in phases:
        # The dictionary key is the table and variable name
        df_phase = ow.output[f"res_bus_3ph.vm_{phase}_pu"]

        # The columns inside the dataframe are the eval_names we defined
        col_name_min = f"vm_{phase}_pu_min"
        col_name_max = f"vm_{phase}_pu_max"

        # Extract both columns as a DataFrame
        voltages = df_phase[[col_name_min, col_name_max]].copy()
        voltages.index = time_index
        output_csv_path = os.path.join(global_voltages_path, f"simulation_results_phase_{phase.upper()}.csv")
        voltages.to_csv(output_csv_path)

        # Full per-bus voltage trajectory (every bus logged this run, not just
        # the summary min/max above or the inverter-bus subset saved below) --
        # saved alongside those, not instead of them. Only present when this
        # run logged every bus (log_all_bus_voltages, the default); the raw
        # per-bus columns are the ones pandapower named by bus index rather
        # than by one of the eval_name strings above.
        bus_cols = [c for c in df_phase.columns if isinstance(c, (int, np.integer))]
        if bus_cols:
            all_bus_voltages = df_phase[bus_cols].astype(np.float32)
            all_bus_voltages.columns = [f"bus_{c}" for c in bus_cols]
            all_bus_voltages.index = time_index
            all_bus_voltages.to_csv(os.path.join(
                global_voltages_path, f"all_bus_voltages_phase_{phase.upper()}.csv"))

    # Extract and plot line losses
    loss_vars = ['pl_a_mw', 'pl_b_mw', 'pl_c_mw']
    total_losses_mw = pd.DataFrame(index=time_index)
    total_losses_mw['Total_Losses_kW'] = 0.0
    has_losses = False
    
    for var_name in loss_vars:
        key = f"res_line_3ph.{var_name}"
        eval_name = f"{var_name}_sum"
        if key in ow.output:
            df_loss = ow.output[key]
            if eval_name in df_loss.columns:
                total_losses_mw['Total_Losses_kW'] += df_loss[eval_name].values * 1000.0
                has_losses = True
            
    if has_losses:
        losses_csv_path = os.path.join(script_dir, test_dir, "simulation_results_losses.csv")
        if not os.path.exists(os.path.dirname(losses_csv_path)):
            os.makedirs(os.path.dirname(losses_csv_path), exist_ok=True)
        total_losses_mw.to_csv(losses_csv_path)
        psr(losses_csv_path, title=f"Combined Network Line Losses {control_type}", ylabel="Losses [kW]", output_pdf=os.path.join(script_dir, test_dir), phase="losses", upperlimit=50, lowerlimit=-50)
    
    
    total_daily_loss = total_losses_mw['Total_Losses_kW'].sum() if has_losses else 0.0
    
    inverters_dir = os.path.join(script_dir, test_dir, "inverter_profiles")
    if not os.path.exists(inverters_dir):
        os.makedirs(inverters_dir, exist_ok=True)

    for phase in ['a', 'b', 'c']:
        p_key = f"asymmetric_sgen.p_{phase}_mw"
        q_key = f"asymmetric_sgen.q_{phase}_mvar"

        # Check if P was successfully logged (Q is optional)
        if p_key in ow.output:
            df_p = ow.output[p_key]
            df_q = ow.output[q_key] if q_key in ow.output else None

            global_p_df = pd.DataFrame(index=time_index)
            global_q_df = pd.DataFrame(index=time_index)
            global_v_df = pd.DataFrame(index=time_index)

            for col in df_p.columns:
                # We only want actual generator columns (integers)
                if isinstance(col, (int, float, np.integer)) or (isinstance(col, str) and col.isdigit()):

                    if df_q is not None and col in df_q.columns:
                        q_vals = df_q[col]
                    else:
                        q_vals = pd.Series(0.0, index=df_p.index)

                    # Filter out empty phases
                    if df_p[col].abs().sum() > 0 or q_vals.abs().sum() > 0:

                        # Convert MW/MVar to kW/kVAr; one column per inverter
                        global_p_df[f"inverter_{col}"] = df_p[col].values * 1000.0
                        global_q_df[f"inverter_{col}"] = q_vals.values * 1000.0

                        if net is not None:
                            bus_idx = net.asymmetric_sgen.at[col, 'bus']
                            v_key = f"res_bus_3ph.vm_{phase}_pu"
                            if v_key in ow.output and bus_idx in ow.output[v_key].columns:
                                global_v_df[f"inverter_{col}"] = ow.output[v_key][bus_idx].values

            # Backfill any NaN values at 00:00:00 caused by initial load flow divergence
            # TODO(initial-instant): the very first timestep (00:00) can diverge under
            # cold-start conditions when voltage controllers are present, so its logged
            # values are backfilled from step 1 rather than being a real solved point.
            # This makes the step-0 voltage/Q an unreliable transient (e.g. Droop's
            # per-day "max" landing at 00:00:00, identical every week regardless of that
            # day's loads). Investigate: seed the droop controller's first step (as the
            # PID seeds prev_v), or add a warm-up power flow, and exclude step 0 from
            # per-day max/violation/effort metrics. A No-Control 0-EV/0-PV run has a
            # clean step 0 (=1.05), so this is a controller-cold-start effect, not the
            # solver. See also the phase-imbalance note below.
            #
            # NOTE(phase-imbalance over-voltage, confirmed): a large share of the
            # >1.05 over-voltages are driven by the 4-wire feeder's per-phase LOAD
            # IMBALANCE, not by PV back-feed. The phase with little/no EV charging
            # over-volts (neutral-point shift pushes the lightly-loaded phase above the
            # 1.05 source) while the loaded phases sit at ~1.05. Verified in the medium
            # run: e.g. at 02:00 phase B (0 kW EV) = 1.0624 while A/C (loaded) = 1.0500,
            # and a 0-EV/0-PV run still over-volts from household-load imbalance alone.
            global_p_df.bfill(inplace=True)
            global_q_df.bfill(inplace=True)
            global_v_df.bfill(inplace=True)

            # Save global dataframes (each holds every inverter of this phase as a column)
            if not global_p_df.empty:
                global_p_df.to_csv(os.path.join(inverters_dir, f"global_p_phase_{phase.upper()}.csv"))

            if not global_q_df.empty:
                global_q_df.to_csv(os.path.join(inverters_dir, f"global_q_phase_{phase.upper()}.csv"))

            if not global_v_df.empty:
                global_v_df.to_csv(os.path.join(inverters_dir, f"global_v_phase_{phase.upper()}.csv"))

            # TODO(curtailment-loss metric): add a per-inverter, per-timestep active-
            # power CURTAILMENT output = P_scheduled - P_actual (kW, >=0), the charging/
            # PV power the inverter gave up to stay within its current rating while
            # supplying Q to fight over/under-voltage. All three controllers curtail P
            # (custom_controller_example.py: droop ~L256, PID ~L389, centralized ~L529):
            # when required_current(P, Q) > i_max they cut P to sqrt(S_max^2 - Q^2).
            #   * P_actual   = global_p_df above (post-curtailment, already saved, kW).
            #   * P_scheduled = the pristine EV/PV profile `sgen_profiles` (== `profiles`
            #     in simulation_with_dynamic_gen; wrapped as ds=DFData(profiles)). The
            #     controllers write net.asymmetric_sgen, NOT the profiles df, so it stays
            #     the pre-curtailment schedule -- but it is currently NOT passed here and
            #     NOT saved, so this must be threaded in (add a `sgen_profiles=` arg to
            #     export_results and pass it from the sim call) or the metric can't be
            #     computed post-hoc for existing runs.
            # Save `active_power_curtailment_phase_{X}.csv` (inverter x timestep) beside
            # global_p; sum over steps/inverters for the per-strategy total (kWh) in
            # compare_control_strategies.py. Curtailment is a No-Control-free metric
            # (NoControl has no controller, so its curtailment is identically 0).

    # Create combined plot for MINIMUM and MAXIMUM voltages
    pcp(os.path.join(global_voltages_path, "simulation_results_phase_A.csv"),
        os.path.join(global_voltages_path, "simulation_results_phase_B.csv"),
        os.path.join(global_voltages_path, "simulation_results_phase_C.csv"),
        os.path.join(script_dir, test_dir, 'pdfs'),
        title="Combined Min/Max Voltage Profiles - All Phases (w/ PV)",
        filename=f"com_min_max_phases_plot_{test_name}.pdf"
        )

    # EN 50160 voltage unbalance factor. vuf_max / vuf_mean are per-step
    # reductions over the LV buses; aggregate to the global worst (max over time)
    # and the grand mean (mean over time, since the bus count is constant each
    # step). pandas skips NaN rows from any diverged step. Stash on the
    # OutputWriter so callers can print it beside the (normalized) loss figure
    # without any change to the return signatures.
    worst_vuf = mean_vuf = float('nan')
    vuf_key = 'res_bus_3ph.unbalance_percent'
    if vuf_key in ow.output:
        df_vuf = ow.output[vuf_key]
        if 'vuf_max' in df_vuf.columns:
            worst_vuf = float(df_vuf['vuf_max'].max())
        if 'vuf_mean' in df_vuf.columns:
            mean_vuf = float(df_vuf['vuf_mean'].mean())
    ow.vuf_worst, ow.vuf_mean = worst_vuf, mean_vuf

    # Voltage-quality metrics (reporting only, not part of any optimization
    # objective): how often the voltage is outside [V_LOW, V_HIGH], by how
    # much, and how far from 1.0 p.u. in general -- summed/computed over
    # every LV bus, phase, and step of the day.
    total_violations = total_excursion = 0.0
    quality_columns = {}
    for phase in ('a', 'b', 'c'):
        key = f'res_bus_3ph.vm_{phase}_pu'
        if key in ow.output:
            df_v = ow.output[key]
            if f'viol_count_{phase}' in df_v.columns:
                total_violations += float(df_v[f'viol_count_{phase}'].sum())
                quality_columns[f'viol_count_{phase}'] = df_v[f'viol_count_{phase}'].values
            if f'viol_mag_{phase}' in df_v.columns:
                total_excursion += float(df_v[f'viol_mag_{phase}'].sum())
                quality_columns[f'viol_mag_{phase}'] = df_v[f'viol_mag_{phase}'].values
            if f'sq_dev_{phase}' in df_v.columns:
                quality_columns[f'sq_dev_{phase}'] = df_v[f'sq_dev_{phase}'].values
    ow.total_violations, ow.total_excursion_pu = total_violations, total_excursion

    if vuf_key in ow.output:
        df_vuf = ow.output[vuf_key]
        if 'vuf_max' in df_vuf.columns:
            quality_columns['vuf_max'] = df_vuf['vuf_max'].values
        if 'vuf_mean' in df_vuf.columns:
            quality_columns['vuf_mean'] = df_vuf['vuf_mean'].values

    # Persist the per-step, all-LV-bus quality metrics so downstream
    # comparison scripts (compare_control_strategies.py) can compute exact
    # violation/RMS-deviation/VUF figures instead of inferring them from the
    # min/max-only voltage CSVs above.
    if quality_columns and net is not None:
        n_lv_buses = int((net.bus.vn_kv < 1.0).sum())
        quality_df = pd.DataFrame(quality_columns, index=time_index)
        quality_df['n_lv_buses'] = n_lv_buses
        quality_csv_path = os.path.join(script_dir, test_dir, "voltage_quality.csv")
        if not os.path.exists(os.path.dirname(quality_csv_path)):
            os.makedirs(os.path.dirname(quality_csv_path), exist_ok=True)
        quality_df.to_csv(quality_csv_path)

    # MV/LV transformer loading: apparent power S and loading_percent at the
    # substation transformer feeding this LV feeder, so congestion caused by
    # reactive-power injection/absorption (not just voltage) can be checked
    # independently of the LV-bus voltage-quality metrics above.
    trafo_key = 'res_trafo_3ph.loading_percent'
    ow.trafo_loading_max_pct = float('nan')
    if trafo_key in ow.output and net is not None and len(net.trafo) > 0:
        trafo_idx = net.trafo.index[0]
        p_total_mw = sum(ow.output[f'res_trafo_3ph.p_{ph}_hv_mw'][trafo_idx]
                         for ph in ('a', 'b', 'c')
                         if f'res_trafo_3ph.p_{ph}_hv_mw' in ow.output)
        q_total_mvar = sum(ow.output[f'res_trafo_3ph.q_{ph}_hv_mvar'][trafo_idx]
                           for ph in ('a', 'b', 'c')
                           if f'res_trafo_3ph.q_{ph}_hv_mvar' in ow.output)
        s_total_mva = np.sqrt(p_total_mw ** 2 + q_total_mvar ** 2)

        trafo_df = pd.DataFrame({
            'loading_percent': ow.output[trafo_key][trafo_idx].values,
            's_kva': s_total_mva.values * 1000.0,
            'p_kw': p_total_mw.values * 1000.0,
            'q_kvar': q_total_mvar.values * 1000.0,
        }, index=time_index)
        for ph in ('a', 'b', 'c'):
            ph_key = f'res_trafo_3ph.loading_{ph}_percent'
            if ph_key in ow.output:
                trafo_df[f'loading_{ph}_percent'] = ow.output[ph_key][trafo_idx].values
        trafo_df['sn_kva'] = float(net.trafo.at[trafo_idx, 'sn_mva']) * 1000.0
        trafo_df.bfill(inplace=True)

        trafo_csv_path = os.path.join(script_dir, test_dir, "transformer_loading.csv")
        if not os.path.exists(os.path.dirname(trafo_csv_path)):
            os.makedirs(os.path.dirname(trafo_csv_path), exist_ok=True)
        trafo_df.to_csv(trafo_csv_path)
        ow.trafo_loading_max_pct = float(trafo_df['loading_percent'].max())

    return total_daily_loss
    
def build_day_inputs(week, weekly_mult, step_mult, ev_sgen_indices, pv_sgen_indices,
                     seed=42, ev_window=None, pv_file=None):
    """
    Build the time-series input tables for one simulated day:
    - household_loads: scaled/interpolated P & Q per load and phase
      (columns load_{i}_p_{ph} / load_{i}_q_{ph}, values in MW/MVAr)
    - sgen_profiles: EV charging (negative MW) and PV generation (positive MW)
      per generator (columns ev_{idx} / pv_{idx})
    - new_steps: number of time steps
    - time_index: HH:MM:SS labels for the output writer
    - ev_session_map: {sgen_idx: {'session_id', 'session_file', 'bucket'}} --
      which sampled charging-session file each EV inverter drew from, for
      traceability back to filtered_apartments_2h/

    Shared by simulation_with_dynamic_gen and the gain-optimizer scripts so the
    surrogate models are built from exactly the same inputs as the real run.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    df_master = pd.read_csv(os.path.join(script_dir, "data_source_internal.csv"), index_col=0)*weekly_mult

    original_steps = len(df_master)
    new_steps = int((original_steps - 1) * step_mult + 1)
    new_idx = np.linspace(0, original_steps - 1, new_steps)
    old_idx = np.arange(original_steps)
    freq=15*60/step_mult
    freq=f'{freq}s'
    household_loads = pd.DataFrame({col: np.interp(new_idx, old_idx, df_master[col]) for col in df_master.columns})
    time_index = pd.date_range(start="00:00:00", periods=new_steps, freq=freq).strftime('%H:%M:%S')

    # Load this week's PV generation profile (column week_N of the
    # consolidated PVGIS table; see pv_data/build_pv_source_internal.py).
    # pv_file selects the per-location cache (sim_config.pv_source_path);
    # None falls back to the active location's cache (sim_config.LOCATION).
    if pv_file is None:
        pv_file = cfg.pv_source_path()
    pv_master = pd.read_csv(pv_file, index_col=0)
    pv_mw_raw = np.nan_to_num(pv_master[f'week_{week + 1}'].values / 1e6, nan=0.0)

    # Interpolate PV curve to match the exact number of 1-min time steps
    pv_mw_interpolated = np.interp(np.linspace(0, 1, new_steps),
                                   np.linspace(0, 1, len(pv_mw_raw)),
                                   pv_mw_raw)

    # Sample one EV charging session per charger from the configured time-of-day
    # window (sim_config.EV_LOAD_WINDOW: 'night'/'morning'/'midday', or 'all' to
    # pool every bucket). run.bat overrides the default via ev_window.
    ev_profile_files = []
    for bucket in cfg.ev_load_buckets(ev_window):
        bucket_dir = os.path.join(script_dir, 'filtered_apartments_2h', bucket)
        ev_profile_files += [os.path.join(bucket_dir, f)
                             for f in os.listdir(bucket_dir) if f.endswith('.csv')]

    profiles_dict = {}
    ev_session_map = {}

    random.seed(seed)
    for sgen_idx in ev_sgen_indices:
        chosen_file = random.choice(ev_profile_files)
        session_name = os.path.splitext(os.path.basename(chosen_file))[0]
        id_match = re.search(r'(\d+)$', session_name)
        ev_session_map[sgen_idx] = {
            'session_id': id_match.group(1) if id_match else session_name,
            'session_file': os.path.basename(chosen_file),
            'bucket': os.path.basename(os.path.dirname(chosen_file)),
        }
        ev_profile = pd.read_csv(chosen_file)
        if 'plugged_in' not in ev_profile.columns:
            raise KeyError(
                f"{chosen_file} has no 'plugged_in' column -- regenerate the "
                "profile data with preprocess_charging_records.py")
        # Missing readings count as no charging (see PV note above)
        ev_load_mw = -1.0 * np.nan_to_num(ev_profile['load_kw'].values, nan=0.0) / 1000.0

        # Interpolate the session curve onto the simulation time grid
        profiles_dict[f"ev_{sgen_idx}"] = np.interp(
            np.linspace(0, 1, new_steps),
            np.linspace(0, 1, len(ev_load_mw)),
            ev_load_mw
        )

        # Plugged-in indicator on the same grid. Linear interpolation of the
        # 0/1 series ramps at the session edges; thresholding at > 0 ("plugged
        # if either neighbouring 15-min sample is plugged") is the one rule
        # that preserves plugged >= (P > 0) under the linear interpolation of
        # the power curve above -- a >= 0.5 threshold (or a floor/ZOH) would
        # leave interpolated charging power outside the plugged window at the
        # ramps. Cost: the window widens by at most one base slot per edge.
        plugged_raw = np.nan_to_num(ev_profile['plugged_in'].values, nan=0.0)
        plugged_interp = np.interp(np.linspace(0, 1, new_steps),
                                   np.linspace(0, 1, len(plugged_raw)),
                                   plugged_raw)
        profiles_dict[f"plugged_{sgen_idx}"] = (plugged_interp > 1e-9).astype(float)

    for sgen_idx in pv_sgen_indices:
        profiles_dict[f"pv_{sgen_idx}"] = pv_mw_interpolated

    # Build the profiles DataFrame in a single pass
    sgen_profiles = pd.DataFrame(profiles_dict)

    return household_loads, sgen_profiles, new_steps, time_index, ev_session_map


def calculate_time_varying_sensitivity(net, inverters, phases, target_buses,
                                       household_loads, sgen_profiles,
                                       ev_sgen_indices, ev_phases,
                                       pv_sgen_indices, pv_phases,
                                       anchor_every=1):
    """
    Time-varying voltage sensitivity for the gain optimizers' surrogate.

    Anchor matrices dV/dQ are estimated every `anchor_every` steps with that
    instant's household loads, EV charging power and PV injection applied,
    and held constant until the next anchor (zero-order hold). The routine
    (sensitivity.compute_sensitivity_matrix), the cadence
    (cfg.sens_anchor_steps) and the hold are identical to the online
    re-estimation inside LinearVoltVarController, so the surrogate plant is
    built exactly the way the real centralized controller builds its model.

    Returns a dict:
      'S_full'         : (n_anchors, 3*n_buses, n_inv) all-bus anchor matrices
      'anchor_of_step' : (n_steps,) index of the anchor active at each step
      'v_rows'         : rows of the flat all-bus voltage vector for each
                         inverter's own connection bus/phase (the feedback
                         slice for the local control laws)

    Shared by pid_optimizer_approx.py and droop_optimizer_approx.py.
    """
    n_inv = len(inverters)
    n_steps = len(household_loads)
    anchor_every = max(1, int(anchor_every))

    anchor_steps = list(range(0, n_steps, anchor_every))
    n_anchors = len(anchor_steps)
    anchor_of_step = np.minimum(np.arange(n_steps) // anchor_every, n_anchors - 1)

    print(f"Calculating sensitivity anchors (dV/dQ): "
          f"{n_anchors} anchors over {n_steps} steps "
          f"({n_anchors * (n_inv + 1)} power flows)...")
    t_start = time.perf_counter()

    net_temp = copy.deepcopy(net)

    # Pre-compute the household-load column names in load-table order
    load_ids = list(net_temp.asymmetric_load.index)
    p_cols = {ph: [f'load_{i}_p_{ph}' for i in load_ids] for ph in ('a', 'b', 'c')}
    q_cols = {ph: [f'load_{i}_q_{ph}' for i in load_ids] for ph in ('a', 'b', 'c')}

    # Rows of the all-bus voltage vector for each inverter's connection
    # bus/phase: slicing an anchor matrix with these rows yields the
    # (n_inv x n_inv) sub-matrix the local control laws are fed back with.
    v_rows = sensitivity.voltage_row_indices(net_temp, target_buses, phases)

    S_full = np.zeros((n_anchors, 3 * len(net_temp.bus), n_inv))

    for a, t in enumerate(anchor_steps):
        # Apply this instant's household loads (P & Q, all phases)
        for ph in ('a', 'b', 'c'):
            net_temp.asymmetric_load[f'p_{ph}_mw'] = household_loads.loc[t, p_cols[ph]].values
            net_temp.asymmetric_load[f'q_{ph}_mvar'] = household_loads.loc[t, q_cols[ph]].values

        # Apply this instant's EV charging power and PV injection
        for k, s_idx in enumerate(ev_sgen_indices):
            net_temp.asymmetric_sgen.at[s_idx, f'p_{ev_phases[k]}_mw'] = sgen_profiles.at[t, f'ev_{s_idx}']
        for k, s_idx in enumerate(pv_sgen_indices):
            net_temp.asymmetric_sgen.at[s_idx, f'p_{pv_phases[k]}_mw'] = sgen_profiles.at[t, f'pv_{s_idx}']

        # Shared perturbation sweep (strict=False: a non-converged perturbed
        # flow yields a zero column, exactly the old per-column fallback).
        try:
            S_a, _ = sensitivity.compute_sensitivity_matrix(
                net_temp, inverters,
                run_kwargs={"lightsim2grid": True},
                base_init="flat" if a == 0 else "results",
                strict=False)
            S_full[a] = S_a
        except pp.LoadflowNotConverged:
            # Reuse the previous anchor's matrix if the base flow fails
            S_full[a] = S_full[a-1] if a > 0 else 0.0
            continue

        if (a + 1) % 10 == 0 or a == n_anchors - 1:
            print(f"  sensitivity: anchor {a + 1}/{n_anchors}")

    print(f"Sensitivity sweep finished in {time.perf_counter() - t_start:.1f}s")
    return {'S_full': S_full, 'anchor_of_step': anchor_of_step, 'v_rows': v_rows}


def extract_all_bus_v_base(ow, net):
    """
    (n_steps, 3*n_buses) voltage trajectory from the OutputWriter of a run
    executed with log_all_bus_voltages=True, ordered exactly like
    sensitivity.all_bus_voltages ([va..., vb..., vc...] in net.bus order).
    NaN entries (e.g. a diverged step) read as 1.0 p.u., matching
    sensitivity.all_bus_voltages. Shared by the PID and droop optimizers.
    """
    bus_order = list(net.bus.index)
    blocks = []
    for ph in ('a', 'b', 'c'):
        df = ow.output[f'res_bus_3ph.vm_{ph}_pu']
        blocks.append(np.nan_to_num(df[bus_order].to_numpy(dtype=float), nan=1.0))
    return np.concatenate(blocks, axis=1)


def simulation_with_dynamic_gen(net, ev_sgen_indices, ev_phases, pv_sgen_indices, pv_phases,
                                        dyn_sgen_indices,  contribution, pv_contribution, dyn_sgen_indices_a, dyn_sgen_indices_b, dyn_sgen_indices_c, weekly_mult=1.0, step_mult=cfg.STEP_MULT, week=0, min_max_indx=None, individualcontrol=0, timestamp='1', seed=cfg.NETWORK_SEED, pid_gains=[0.05, 0.03, 0.007, 0.95], return_ow=False, saving_optimal_values=False, q_mode=cfg.Q_MODE, log_all_bus_voltages=True, pv_file=None, droop_gains_file=None, max_curtail_frac=cfg.MAX_P_CURTAIL_FRAC):
    """
    q_mode : 0 = the controllers may supply Q at all times (cce.Q_MODE_ALWAYS);
             1 = they only supply Q while an EV is plugged in
             (cce.Q_MODE_PLUGGED_ONLY) -- the session's connection window from
             the plugged_in profile column; a plugged vehicle that finished
             charging (P == 0) keeps supporting the grid. Applies uniformly to
             PID, Droop and the Centralized controller; ignored when
             individualcontrol==3 (No Control, which never sets Q anyway).
    log_all_bus_voltages : log every bus's per-phase voltage trajectory to the
             OutputWriter instead of only the inverter buses, and have
             export_results write it to global_voltages/all_bus_voltages_phase_*.csv
             alongside the existing min/max summary and inverter-subset CSVs.
             Default True so every real run keeps the full node-level record;
             the gain optimizers' DE search itself never calls this function
             (it runs on the surrogate), so this only adds cost to the
             handful of real power-flow runs per optimizer call plus the
             yearly batch. Pass False to fall back to the leaner inverter-bus-
             only log (e.g. for a quick exploratory run where the per-bus CSVs
             aren't needed and disk space is a concern -- at fine step
             resolutions they can reach tens of MB per phase per run).
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_master_10sec, profiles, new_steps, time_index, ev_session_map = build_day_inputs(
        week, weekly_mult, step_mult, ev_sgen_indices, pv_sgen_indices, seed=seed, pv_file=pv_file)

    ds1 = DFData(df_master_10sec)
    ds = DFData(profiles)

    net=CostControl_setup(net, ds, ds1, [ev_sgen_indices, ev_phases, pv_sgen_indices, pv_phases])

    # Attach the selected reactive-power controller to the EV inverters
    target_bus_id_a = [net.asymmetric_sgen.at[indx, 'bus'] for indx in dyn_sgen_indices_a]
    target_bus_id_b = [net.asymmetric_sgen.at[indx, 'bus'] for indx in dyn_sgen_indices_b]
    target_bus_id_c = [net.asymmetric_sgen.at[indx, 'bus'] for indx in dyn_sgen_indices_c]

    # Per-step plugged-in masks (the plugged_{idx} profile columns), stacked in
    # each controller's own inverter ordering. Only consulted by the
    # controllers when q_mode == Q_MODE_PLUGGED_ONLY.
    plugged_of = {i: profiles[f"plugged_{i}"].to_numpy(dtype=bool)
                  for i in ev_sgen_indices}

    def plugged_stack(ids):
        return (np.column_stack([plugged_of[i] for i in ids]) if ids
                else np.empty((new_steps, 0), dtype=bool))

    if individualcontrol==0:        #PID control
        # pid_gains = [Kp, Ki, Kd] with optional [q_decay, q_rate, hysteresis]
        q_decay = pid_gains[3] if len(pid_gains) > 3 else 0.95
        # q_rate <= 0 means "no slew limit" (that is how it is encoded in the CSV)
        q_rate = pid_gains[4] if len(pid_gains) > 4 else None
        if q_rate is not None and q_rate <= 0:
            q_rate = None
        hysteresis = pid_gains[5] if len(pid_gains) > 5 else 0.0

        pid_kwargs = dict(order=1, Kp=pid_gains[0], Ki=pid_gains[1], Kd=pid_gains[2],
                          q_decay=q_decay, q_rate=q_rate, hysteresis=hysteresis, q_mode=q_mode,
                          max_curtail_frac=max_curtail_frac)
        # order=1 runs the controller after the ConstControl profile updates
        VectorizedAsymmetricPIDController(net, sgen_ids=dyn_sgen_indices_a, target_bus_ids=target_bus_id_a, phase='a', plugged=plugged_stack(dyn_sgen_indices_a), **pid_kwargs)
        VectorizedAsymmetricPIDController(net, sgen_ids=dyn_sgen_indices_b, target_bus_ids=target_bus_id_b, phase='b', plugged=plugged_stack(dyn_sgen_indices_b), **pid_kwargs)
        VectorizedAsymmetricPIDController(net, sgen_ids=dyn_sgen_indices_c, target_bus_ids=target_bus_id_c, phase='c', plugged=plugged_stack(dyn_sgen_indices_c), **pid_kwargs)

        my_run_kwargs = { "lightsim2grid": True, "init": "flat" } 

    elif individualcontrol==1:      # droop control
        # Per-location tuned droop gains (sim_config.droop_gains_path). If the
        # active location has not been droop-tuned yet, fall back to the default
        # location's gains so a droop run still works, with a warning.
        if droop_gains_file is None:
            droop_gains_file = cfg.droop_gains_path()
        if not os.path.exists(droop_gains_file):
            fallback = cfg.droop_gains_path(cfg.LOCATION)
            print(f"WARNING: droop gains {droop_gains_file} not found; falling "
                  f"back to '{cfg.LOCATION}' gains ({os.path.basename(fallback)}). "
                  f"Tune this scenario with droop_optimizer_approx.py --location.")
            droop_gains_file = fallback
        df_gains = pd.read_csv(droop_gains_file)

        # Per-inverter gains. Current column names are gain_pos/gain_neg; fall
        # back to the pre-rename mean_positive/mean_negative so a CSV written
        # by an older optimizer run still loads.
        pos_col = 'gain_pos' if 'gain_pos' in df_gains.columns else 'mean_positive'
        neg_col = 'gain_neg' if 'gain_neg' in df_gains.columns else 'mean_negative'
        gains_dict = {
            row['id']: {'pos': row[pos_col], 'neg': row[neg_col]}
            for _, row in df_gains.iterrows()
        }
        my_run_kwargs = { "lightsim2grid": True, "init": "flat" }   
        for i, dyn_idx in enumerate(dyn_sgen_indices):
            # Safe to use ev_phases here since dyn_sgen_indices maps 1:1 with ev_sgen_indices
            phase = ev_phases[i] 
            target_bus = net.asymmetric_sgen.at[dyn_idx, 'bus']
            
            # order=1 runs the controller after the ConstControl profile updates
            AsymmetricDynamicDroopController(net, sgen_id=dyn_idx, target_bus_id=target_bus, phase=phase, custom_droop_gains=gains_dict, q_mode=q_mode, plugged=plugged_of[dyn_idx], max_curtail_frac=max_curtail_frac, order=1, level=0)

    elif individualcontrol==2:      #centralized control
        savingpath=os.path.join(script_dir, f"q_opt_results", f"{timestamp}", min_max_indx , f"week_{int(week)}")
        # The controller creates the directory itself before its first write;
        # saving_voltages just switches the per-step CSV dump on or off.
        # order=1 runs the controller after the ConstControl profile updates
        LinearVoltVarController(net, sgen_ids=dyn_sgen_indices, savingDest=savingpath,
                                saving_voltages=saving_optimal_values, q_mode=q_mode,
                                step_mult=step_mult, plugged=plugged_stack(dyn_sgen_indices),
                                max_curtail_frac=max_curtail_frac, order=1)
        my_run_kwargs = { "lightsim2grid": True, "init": "flat" }
    else:           # No control
        my_run_kwargs = { "lightsim2grid": True, "init": "flat" }
    ow = ts.OutputWriter(net, time_steps=range(new_steps), output_path=None, log_variables=list())

    # Track voltages for specific buses where inverters are attached. The
    # optimizers' baseline runs need every bus instead (all-bus V_base for the
    # tuning objective), so the flag widens this same log -- a second
    # log_variable registration for the same (table, variable) would collide.
    if log_all_bus_voltages:
        all_buses = net.bus.index.tolist()
        sgen_buses_a = sgen_buses_b = sgen_buses_c = all_buses
    else:
        sgen_buses_a = net.asymmetric_sgen[net.asymmetric_sgen.phi == 'a'].bus.unique().tolist()
        sgen_buses_b = net.asymmetric_sgen[net.asymmetric_sgen.phi == 'b'].bus.unique().tolist()
        sgen_buses_c = net.asymmetric_sgen[net.asymmetric_sgen.phi == 'c'].bus.unique().tolist()
    
    if sgen_buses_a:
        ow.log_variable('res_bus_3ph', 'vm_a_pu', index=sgen_buses_a)
    if sgen_buses_b:
        ow.log_variable('res_bus_3ph', 'vm_b_pu', index=sgen_buses_b)
    if sgen_buses_c:
        ow.log_variable('res_bus_3ph', 'vm_c_pu', index=sgen_buses_c)

    # Track minimum voltage
    ow.log_variable('res_bus_3ph', 'vm_a_pu', eval_function=np.min, eval_name='vm_a_pu_min')
    ow.log_variable('res_bus_3ph', 'vm_b_pu', eval_function=np.min, eval_name='vm_b_pu_min')
    ow.log_variable('res_bus_3ph', 'vm_c_pu', eval_function=np.min, eval_name='vm_c_pu_min')
    
    # Track maximum voltage
    ow.log_variable('res_bus_3ph', 'vm_a_pu', eval_function=np.max, eval_name='vm_a_pu_max')
    ow.log_variable('res_bus_3ph', 'vm_b_pu', eval_function=np.max, eval_name='vm_b_pu_max')
    ow.log_variable('res_bus_3ph', 'vm_c_pu', eval_function=np.max, eval_name='vm_c_pu_max')

    # Track phase unbalance (EN 50160 VUF = |V2|/|V1| %, pandapower's built-in
    # unbalance_percent column). Restrict to LV buses so the single 11 kV MV bus
    # (a different voltage base, always near-balanced) does not dilute the mean.
    lv_buses = net.bus.index[net.bus.vn_kv < 1.0].tolist()
    ow.log_variable('res_bus_3ph', 'unbalance_percent', index=lv_buses,
                    eval_function=np.max, eval_name='vuf_max')
    ow.log_variable('res_bus_3ph', 'unbalance_percent', index=lv_buses,
                    eval_function=np.mean, eval_name='vuf_mean')

    # Voltage-quality metrics (reporting only -- see export_results for the
    # aggregation and network_setup.py's caller for the print statement):
    #   viol_count_* -- how many of the LV buses (this phase) sit outside
    #                   [so.V_LOW, so.V_HIGH] at this step
    #   viol_mag_*   -- total p.u. excursion beyond whichever band edge was
    #                   crossed, summed over those buses at this step
    def _count_out_of_band(v):
        return np.sum((v < so.V_LOW) | (v > so.V_HIGH))

    def _excursion_magnitude(v):
        return np.sum(np.maximum(0.0, v - so.V_HIGH) + np.maximum(0.0, so.V_LOW - v))

    #   sq_dev_*     -- sum of (V - 1.0 p.u.)^2 over those buses at this step,
    #                   over ALL buses (not just violating ones) -- the
    #                   accumulator behind an RMS voltage deviation metric,
    #                   a continuous quality measure that doesn't require
    #                   crossing the band to register.
    def _squared_deviation(v):
        return np.sum((v - 1.0) ** 2)

    for phase in ('a', 'b', 'c'):
        ow.log_variable('res_bus_3ph', f'vm_{phase}_pu', index=lv_buses,
                        eval_function=_count_out_of_band, eval_name=f'viol_count_{phase}')
        ow.log_variable('res_bus_3ph', f'vm_{phase}_pu', index=lv_buses,
                        eval_function=_excursion_magnitude, eval_name=f'viol_mag_{phase}')
        ow.log_variable('res_bus_3ph', f'vm_{phase}_pu', index=lv_buses,
                        eval_function=_squared_deviation, eval_name=f'sq_dev_{phase}')

    # Track line losses (active power)
    ow.log_variable('res_line_3ph', 'pl_a_mw', eval_function=np.sum, eval_name='pl_a_mw_sum')
    ow.log_variable('res_line_3ph', 'pl_b_mw', eval_function=np.sum, eval_name='pl_b_mw_sum')
    ow.log_variable('res_line_3ph', 'pl_c_mw', eval_function=np.sum, eval_name='pl_c_mw_sum')
    
    
    # Track individual generator active power (P)
    ow.log_variable('asymmetric_sgen', 'p_a_mw')
    ow.log_variable('asymmetric_sgen', 'p_b_mw')
    ow.log_variable('asymmetric_sgen', 'p_c_mw')

    # Track individual generator reactive power (Q)
    ow.log_variable('asymmetric_sgen', 'q_a_mvar')
    ow.log_variable('asymmetric_sgen', 'q_b_mvar')
    ow.log_variable('asymmetric_sgen', 'q_c_mvar')

    # Track MV/LV transformer loading -- whether the Q injected/absorbed by
    # the controllers causes congestion at the substation transformer, not
    # just at the LV buses. loading_percent is pandapower's own combined
    # (worst-phase) figure; the per-phase loading and HV-side P/Q let
    # export_results also derive the transformer's apparent power S.
    ow.log_variable('res_trafo_3ph', 'loading_percent')
    ow.log_variable('res_trafo_3ph', 'loading_a_percent')
    ow.log_variable('res_trafo_3ph', 'loading_b_percent')
    ow.log_variable('res_trafo_3ph', 'loading_c_percent')
    ow.log_variable('res_trafo_3ph', 'p_a_hv_mw')
    ow.log_variable('res_trafo_3ph', 'p_b_hv_mw')
    ow.log_variable('res_trafo_3ph', 'p_c_hv_mw')
    ow.log_variable('res_trafo_3ph', 'q_a_hv_mvar')
    ow.log_variable('res_trafo_3ph', 'q_b_hv_mvar')
    ow.log_variable('res_trafo_3ph', 'q_c_hv_mvar')


    with warnings.catch_warnings():
        # Suppress benign pandapower warnings that occur during Newton-Raphson
        # intermediate iterations on heavily loaded networks (all steps converge).
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", category=_spla.MatrixRankWarning)
        # Wall-clock timing of the whole day's control+power-flow loop (thesis
        # Ch. 5 computational-cost metric), single-process wall time -- not
        # normalized for parallel execution, which yearly_tests.py does across
        # weeks via ProcessPoolExecutor.
        _t0 = time.perf_counter()
        ts.run_timeseries(net, time_steps=range(new_steps), run=pp.runpp_3ph, continue_on_divergence=True, run_kwargs=my_run_kwargs)
        _elapsed_s = time.perf_counter() - _t0


    # Save to CSV
    if individualcontrol==0:
        control_type='PIDcontrol'
    elif individualcontrol==1:
        control_type='Droopcontrol'
    elif individualcontrol==2:
        control_type='CentralizedControl'
    else:
        control_type='NoControl'
    test_name = f"pv_{int(pv_contribution*100)}_ev_{int(contribution*100)}_files_{min_max_indx}"
    test_name=os.path.join(control_type, test_name)    
    test_dir = os.path.join('sim_results_combined', test_name)
    test_name = f'week_{int(week)}'
    test_dir = os.path.join(test_dir, test_name)
    

    # cfg.SIM_RESULTS_DIR, not script_dir/'sim_results': results can be sent
    # somewhere other than the checkout (THESIS_SIM_RESULTS). Unset, it
    # resolves to exactly the old path.
    script_dir = os.path.join(cfg.SIM_RESULTS_DIR, timestamp)

    total_daily_loss = export_results(ow, time_index, script_dir, test_dir, test_name, control_type=control_type, net=net)

    # Persist the per-day timing beside the other per-run outputs, so
    # compare_control_strategies.py can aggregate a real t_cpu/step figure
    # per strategy (thesis Ch. 5) instead of leaving it undocumented.
    timing_csv_path = os.path.join(script_dir, test_dir, "timing.csv")
    os.makedirs(os.path.dirname(timing_csv_path), exist_ok=True)
    pd.DataFrame([{
        "total_seconds": _elapsed_s,
        "n_steps": new_steps,
        "seconds_per_step": _elapsed_s / new_steps if new_steps else float('nan'),
    }]).to_csv(timing_csv_path, index=False)

    # Record which charging-session file each EV inverter drew from this run,
    # alongside the other per-run outputs, so results can be traced back to
    # filtered_apartments_2h/.
    session_rows = []
    for i, sgen_idx in enumerate(ev_sgen_indices):
        info = ev_session_map.get(sgen_idx, {})
        session_rows.append({
            'inverter_id': sgen_idx,
            'bus': net.asymmetric_sgen.at[sgen_idx, 'bus'],
            'phase': ev_phases[i],
            'session_id': info.get('session_id'),
            'session_file': info.get('session_file'),
            'bucket': info.get('bucket'),
        })
    sessions_csv_path = os.path.join(script_dir, test_dir, "ev_charging_session_ids.csv")
    os.makedirs(os.path.dirname(sessions_csv_path), exist_ok=True)
    # Columns named explicitly so an EV penetration of 0 still writes a valid
    # header. pd.DataFrame([]) has no columns, and to_csv() then produces a
    # completely empty file that pandas refuses to read back ("No columns to
    # parse"), which broke compute_curtailment.py on any PV-only scenario.
    pd.DataFrame(session_rows,
                 columns=['inverter_id', 'bus', 'phase', 'session_id',
                          'session_file', 'bucket']).to_csv(sessions_csv_path,
                                                            index=False)

    # Integrate instantaneous kW into kWh: energy = sum(kW) * hours_per_step.
    # One step is 15 min at step_mult=1, so hours_per_step = 0.25 / step_mult.
    #
    # This used to read `/ step_mult * 4`, i.e. it multiplied by STEPS PER HOUR
    # where it needed HOURS PER STEP -- the reciprocal, so every printed figure
    # was 16x too high at every resolution. Only ever printed, or used in
    # improvement ratios where a constant factor cancels, so no stored metric
    # or optimizer decision was affected; compare_control_strategies.py's
    # total_active_losses_kwh has always integrated this correctly and is what
    # strategy_summary.csv (and the thesis) reports.
    daily_loss_kwh = total_daily_loss * (0.25 / step_mult)

    # Print the losses with the EN 50160 voltage unbalance factor and the two
    # voltage-quality metrics (all computed and stashed on `ow` by
    # export_results) beside them, at the end of the run.
    print(f"[{control_type}] Daily line losses = {daily_loss_kwh:.4f} kWh | "
          f"VUF worst = {ow.vuf_worst:.3f} %  mean = {ow.vuf_mean:.3f} %  (EN 50160 limit 2 %) | "
          f"Voltage violations = {ow.total_violations:.0f} pts, "
          f"total excursion = {ow.total_excursion_pu:.4f} p.u. | "
          f"Trafo loading max = {ow.trafo_loading_max_pct:.1f} %")

    if return_ow:
        return daily_loss_kwh, ow
    return daily_loss_kwh
