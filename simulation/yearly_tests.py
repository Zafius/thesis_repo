import os
import subprocess
import sys
import pandas as pd
import pandapower.networks as nw
import network_setup as ns
import concurrent.futures
import numpy as np
import time
from functools import partial
# Define script_dir globally so worker processes can access it
script_dir = os.path.dirname(os.path.abspath(__file__))
from sim_utils import send_discord_alert, compress_and_delete

# All simulation defaults live in sim_config.py (shared with the gain
# optimizers); the command line -- in practice run.bat -- is the only override.
import sim_config as cfg


def load_location_load_pu(location=None):
    """Weekly max/min load multipliers (Peak[pu]/Min[pu]) for a location.

    Reads locations/<key>/load_weekly_peak_min.csv, the ENTSO-E load zone bound
    to the location in sim_config.LOCATIONS (materialised by
    sim_config.ensure_location). Returns two numpy arrays (one value per ISO
    week), the per-week load scaling for the max-load and min-load batches.
    """
    cfg.ensure_location(location)
    df = pd.read_csv(cfg.region_load_path(location))
    return df['Peak[pu]'].to_numpy(), df['Min[pu]'].to_numpy()


# Default (module-import) load table: the active location in sim_config. The
# yearly workers receive their multiplier and PV file explicitly, so this only
# backs the __main__ index/argmax bookkeeping and any importer that expects it.
yearly_load_max_pu, yearly_load_min_pu = load_location_load_pu()
seed = cfg.SEED
network_seed = cfg.NETWORK_SEED  # fixed seed for EV/PV placement and charging-profile sampling
ev_contribution = cfg.EV_CONTRIBUTION
pv_contribution = cfg.PV_CONTRIBUTION
step = cfg.STEP_MULT
controlType=0 #Control types to run: 0=PID, 1=Droop, 2=Centralized, 3=No control
np.random.seed(seed)

def yearly_sim_gen_max(mult_index, mult, timestamp='2', ControlType=0, pid_gains=[0.05, 0.03, 0.007, .95], step_mult=step, q_mode=cfg.Q_MODE, pv_file=None, droop_gains_file=None, max_curtail_frac=cfg.MAX_P_CURTAIL_FRAC):
    """
    Worker function to run a single simulation for a given week index.
    'mult' is that week's peak load multiplier; 'pv_file'/'droop_gains_file' the
    location's cached PV table and tuned droop gains (all passed explicitly so
    spawn-started workers need no globals).
    """

    # Create a new network object for each process
    net = nw.ieee_european_lv_asymmetric()
    set_seed = network_seed # fixed; swap for np.random.randint(1000) to randomize placement
    # Set up the simulation environment
    net, ev_sgen_indices, ev_phases, pv_sgen_indices, pv_phases, dyn_sgen_indices, dyn_sgen_indices_a, dyn_sgen_indices_b, dyn_sgen_indices_c = ns.setup_simulation(net, seed = set_seed
                                                                                                                                                      , contribution=ev_contribution, pv_contribution=pv_contribution)

    # Run the simulation
    total_loss = ns.simulation_with_dynamic_gen(net, ev_sgen_indices, ev_phases, pv_sgen_indices, pv_phases, dyn_sgen_indices, ev_contribution, pv_contribution, dyn_sgen_indices_a, dyn_sgen_indices_b, dyn_sgen_indices_c, weekly_mult=mult, step_mult=step_mult, week=mult_index, min_max_indx='max', individualcontrol=ControlType, timestamp=timestamp,  seed = set_seed, pid_gains=pid_gains, q_mode=q_mode, pv_file=pv_file, droop_gains_file=droop_gains_file, max_curtail_frac=max_curtail_frac)

    return total_loss

def yearly_sim_gen_min(mult_index, mult, timestamp='3', ControlType=0, pid_gains=[0.05, 0.03, 0.007, .95], step_mult=step, q_mode=cfg.Q_MODE, pv_file=None, droop_gains_file=None, max_curtail_frac=cfg.MAX_P_CURTAIL_FRAC):
    """
    Worker function to run a single simulation for a given week index.
    'mult' is that week's minimum load multiplier; 'pv_file'/'droop_gains_file'
    the location's cached PV table and tuned droop gains (all passed explicitly
    so spawn-started workers need no globals).
    """

    # Create a new network object for each process
    net = nw.ieee_european_lv_asymmetric()
    set_seed = network_seed # fixed; swap for np.random.randint(1000) to randomize placement
    # Set up the simulation environment
    net, ev_sgen_indices, ev_phases, pv_sgen_indices, pv_phases, dyn_sgen_indices, dyn_sgen_indices_a, dyn_sgen_indices_b, dyn_sgen_indices_c = ns.setup_simulation(net, seed=set_seed, contribution=ev_contribution, pv_contribution=pv_contribution)

    # Run the simulation
    total_loss = ns.simulation_with_dynamic_gen(net, ev_sgen_indices, ev_phases, pv_sgen_indices, pv_phases, dyn_sgen_indices, ev_contribution, pv_contribution, dyn_sgen_indices_a, dyn_sgen_indices_b, dyn_sgen_indices_c, weekly_mult=mult, step_mult=step_mult, week=mult_index, min_max_indx='min', individualcontrol=ControlType, timestamp=timestamp,  seed = set_seed, pid_gains=pid_gains, q_mode=q_mode, pv_file=pv_file, droop_gains_file=droop_gains_file, max_curtail_frac=max_curtail_frac)

    return total_loss


def run_y_sim(max_mults, min_mults, max_indx=True, timestamp='0', ControlType=0, pid_gains=[0.15, 0.03, 0.007], step_mult=step, q_mode=cfg.Q_MODE, pv_file=None, droop_gains_file=None, max_curtail_frac=cfg.MAX_P_CURTAIL_FRAC):
    np.random.seed(seed)

    if max_indx:
        indices = np.arange(len(max_mults))
        frozen_func = partial(yearly_sim_gen_max, timestamp=timestamp, ControlType=ControlType, step_mult=step_mult, q_mode=q_mode, pv_file=pv_file, droop_gains_file=droop_gains_file, max_curtail_frac=max_curtail_frac)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # One task per ISO week; each gets its (index, peak multiplier) pair
            results_generator = executor.map(frozen_func, indices, max_mults)
    else:
        indices = np.arange(len(min_mults))
        frozen_func = partial(yearly_sim_gen_min, timestamp=timestamp, ControlType=ControlType, pid_gains=pid_gains, step_mult=step_mult, q_mode=q_mode, pv_file=pv_file, droop_gains_file=droop_gains_file, max_curtail_frac=max_curtail_frac)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # One task per ISO week; each gets its (index, minimum multiplier) pair
            results_generator = executor.map(frozen_func, indices, min_mults)
    return results_generator

def Start_sim_test(max_mults, min_mults, pv_file=None, droop_gains_file=None, ControlType=0, timestamp='0', pid_gains=[0.15, 0.03, 0.007, .95], step_mult=step, q_mode=cfg.Q_MODE, max_curtail_frac=cfg.MAX_P_CURTAIL_FRAC):

        send_discord_alert("Starting Simulation ⌛")
        # list(...) forces the pool to run to completion; per-week losses are
        # already written to sim_results/.../simulation_results_losses.csv by
        # each worker, and compare_control_strategies.py integrates those into
        # the annual total -- no separate summary needed here.
        list(run_y_sim(max_mults, min_mults, max_indx=True, timestamp=timestamp, ControlType=ControlType, pid_gains=pid_gains, step_mult=step_mult, q_mode=q_mode, pv_file=pv_file, droop_gains_file=droop_gains_file, max_curtail_frac=max_curtail_frac))
        send_discord_alert("✅ Max load Simulation ended ")

        list(run_y_sim(max_mults, min_mults, max_indx=False, timestamp=timestamp, ControlType=ControlType, pid_gains=pid_gains, step_mult=step_mult, q_mode=q_mode, pv_file=pv_file, droop_gains_file=droop_gains_file, max_curtail_frac=max_curtail_frac))
        send_discord_alert("✅ Min load Simulation ended ")
        if ControlType == 2:
            # Compress the results folder for this test
            output_name = f"{timestamp}_CentralizedControl_results"
            folder_to_compress = os.path.join(script_dir, 'q_opt_results', timestamp)
            compress_and_delete(folder_to_compress, output_name)

        print("Simulation completed")
        send_discord_alert(f"✅ **Success!** The Simulations have finished running for ControlType: {ControlType}! 🎉")    
    
def _prompt_yes(question, default=False):
    """Ask a yes/no question. Non-interactive (piped/background) stdin can't
    answer, so it returns `default` rather than hanging on input()."""
    if not sys.stdin.isatty():
        print(f"{question} -> non-interactive stdin, using default "
              f"({'yes' if default else 'no'}).")
        return default
    try:
        return input(f"{question} [y/N]: ").strip().lower() in ('y', 'yes')
    except EOFError:
        return default


def _run_optimizer(kind, location, step_mult, q_mode):
    """Run a gain optimizer as a subprocess at the run's step/q-mode so the
    tuned gains match the resolution and Q-availability they'll be used at."""
    script = 'pid_optimizer_approx.py' if kind == 'PID' else 'droop_optimizer_approx.py'
    print(f"\nRunning {script} for '{location}' (step={step_mult}, q_mode={q_mode}) ...")
    subprocess.run(
        [sys.executable, os.path.join(script_dir, script),
         '--location', str(location), '--step', str(step_mult),
         '--q-mode', str(q_mode)],
        check=True)


def ensure_tuned_gains(controls, location, step_mult, q_mode, pid_gains_explicit):
    """For each requested controller that has no tuned gains for this
    location+scenario, offer to run its optimizer now; declining proceeds with
    the default parameters (built-in PID gains, or the fallback droop gains).

    Runs BEFORE the batch so the freshly tuned files are on disk when the
    workers start. A tuned run overwrites nothing else -- gains are keyed by
    scenario (sim_config.scenario_tag)."""
    scenario = cfg.scenario_tag()

    # PID (control 0) -- skipped when the user passed explicit --pid-gains.
    if 0 in controls and not pid_gains_explicit and not os.path.exists(cfg.pid_gains_path(location)):
        if _prompt_yes(f"No tuned PID gains for '{location}' scenario {scenario}. "
                       f"Run pid_optimizer_approx.py now?"):
            _run_optimizer('PID', location, step_mult, q_mode)
        else:
            print("Proceeding with built-in default PID gains.")

    # Droop (control 1).
    if 1 in controls and not os.path.exists(cfg.droop_gains_path(location)):
        if _prompt_yes(f"No tuned droop gains for '{location}' scenario {scenario}. "
                       f"Run droop_optimizer_approx.py now?"):
            _run_optimizer('Droop', location, step_mult, q_mode)
        else:
            print(f"Proceeding with fallback droop gains (default location '{cfg.LOCATION}').")


CONTROL_NAMES = {0: 'PID', 1: 'Droop', 2: 'Centralized', 3: 'No'}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Voltage-control simulations on the IEEE European LV network. "
                    "By default runs the single worst-case day (highest load multiplier) "
                    "for PID only, and prints the daily losses.")
    parser.add_argument('--controls', type=int, nargs='+', default=[controlType], choices=[0, 1, 2, 3],
                        metavar='N',
                        help='Control types to run: 0=PID, 1=Droop, 2=Centralized, '
                             f'3=No control (default: {controlType})')
    parser.add_argument('--yearly', action='store_true',
                        help='Run the full 53-week yearly batch (max & min load, multiprocessing) '
                             'instead of the single worst-case day')
    parser.add_argument('--day-extreme', choices=['max', 'min'], default='max',
                        help='Which single day to run when --yearly is not given. '
                             '"max" (default) is the peak-load day of the year, the '
                             'worst case for charging undervoltage; "min" is the '
                             'lowest-load day, where PV generation meets the load '
                             'floor and overvoltage binds instead. Ignored with --yearly, '
                             'which covers both envelopes for all 53 weeks.')
    parser.add_argument('--pid-gains', type=float, nargs='+', default=None,
                        metavar='G',
                        help='PID params: Kp Ki Kd [q_decay [q_rate [hysteresis]]]. '
                             'q_decay .8=fast, .95=slow, 1=no decay. q_rate = max change '
                             'in Q per step (MVar), <=0 disables slew limiting. hysteresis '
                             '= p.u. shrink of the integral-decay band. '
                             '(default: the location+scenario tuned gains under '
                             'locations/<key>/gains/ if present, '
                             'otherwise built-in values)')
    parser.add_argument('--step', type=float, default=step,
                        help='Time-resolution multiplier: 1 = 15-minute steps (96/day), '
                             '3 = 5-minute steps, 15 = 1-minute steps. Higher = finer '
                             'resolution but proportionally slower '
                             f'(default from sim_config.py: {step})')
    parser.add_argument('--q-mode', type=int, default=cfg.Q_MODE, choices=[0, 1],
                        help='Reactive-power supply mode for PID/Droop/Centralized: '
                             '0 = supply Q at all times, even while idle; '
                             '1 = supply Q only while actively charging (P != 0) '
                             f'(default from sim_config.py: {cfg.Q_MODE})')
    parser.add_argument('--max-curtail-frac', type=float, default=cfg.MAX_P_CURTAIL_FRAC,
                        metavar='F',
                        help='Max fraction [0-1] of an inverter\'s scheduled active '
                             'power it may curtail for reactive support before Q is '
                             'reduced instead: 1.0 = curtail P freely (Q priority), '
                             '0.2 = give up at most 20%% of P, 0.0 = never curtail P '
                             f'(default from sim_config.py: {cfg.MAX_P_CURTAIL_FRAC})')
    parser.add_argument('--location', default=cfg.LOCATION,
                        choices=sorted(cfg.LOCATIONS),
                        help='Simulation location: sets both the ENTSO-E load '
                             'zone (weekly max/min multipliers) and the PVGIS '
                             'PV point (optimum tilt/azimuth). The PV curve is '
                             'fetched and cached on first use. '
                             f'(default from sim_config.py: {cfg.LOCATION})')
    parser.add_argument('--timestamp', default=None,
                        help='Output folder name under sim_results/ (default: the '
                             'current time, %%Y-%%m-%%d_%%H.%%M). Pass an existing '
                             'value to make multiple invocations -- e.g. one per '
                             '--controls value, run separately for per-controller '
                             'timing -- share a single sim_results/<timestamp>/ '
                             'folder instead of each getting its own.')
    args = parser.parse_args()

    # Resolve the location: its weekly peak/min load multipliers, the cached PV
    # table and the tuned droop gains (load copied + PV fetched on first use).
    max_mults, min_mults = load_location_load_pu(args.location)
    pv_file = cfg.pv_source_path(args.location)
    droop_gains_file = cfg.droop_gains_path(args.location)
    loc = cfg.get_location(args.location)
    print(f"Location '{args.location}': load zone {loc['region']}, "
          f"PV point {loc['lat']},{loc['lon']} -> {os.path.basename(cfg.location_dir(args.location))}/")

    # If a requested controller lacks tuned gains for this location+scenario,
    # offer to run its optimizer now (else fall back to default params).
    ensure_tuned_gains(args.controls, args.location, args.step, args.q_mode,
                       pid_gains_explicit=args.pid_gains is not None)

    # Resolve PID gains: explicit flag > this location's tuned CSV > built-in
    # fallback. Columns are read in order and any trailing ones may be absent,
    # so a CSV written by an older optimizer run still loads.
    PID_PARAM_ORDER = ['Kp', 'Ki', 'Kd', 'q_decay', 'q_rate', 'hysteresis']
    if args.pid_gains is None:
        pid_gains_csv = cfg.pid_gains_path(args.location)
        if os.path.exists(pid_gains_csv):
            row = pd.read_csv(pid_gains_csv).iloc[0]
            args.pid_gains = [float(row[c]) for c in PID_PARAM_ORDER if c in row.index]
            print(f"PID gains loaded from {args.location}/{os.path.basename(pid_gains_csv)}: {args.pid_gains}")
        else:
            # Conservative, non-oscillating defaults: Ki * typical error stays
            # well under s_limit, and Q may move at most 10% of range per step.
            args.pid_gains = [0.05, 0.02, 0.0, 0.95, 0.00079, 0.004]
            print(f"No tuned PID gains for '{args.location}'; using built-in defaults "
                  f"{args.pid_gains} (run pid_optimizer_approx.py --location {args.location}).")
    elif not 3 <= len(args.pid_gains) <= 6:
        parser.error(f"--pid-gains takes 3 to 6 values ({'  '.join(PID_PARAM_ORDER)}), "
                     f"got {len(args.pid_gains)}")
    else:
        print(f"PID gains from command line: {args.pid_gains}")

    try:
        timestamp = args.timestamp or time.strftime("%Y-%m-%d_%H.%M")

        if args.yearly:
            for control in args.controls:
                Start_sim_test(max_mults, min_mults, pv_file=pv_file,
                               droop_gains_file=droop_gains_file,
                               ControlType=control, timestamp=timestamp,
                               pid_gains=args.pid_gains, step_mult=args.step,
                               q_mode=args.q_mode, max_curtail_frac=args.max_curtail_frac)
        else:
            # Single-day run on the most extreme week of the chosen load envelope.
            # "max" takes the week with the highest peak-load multiplier (the worst
            # case for charging undervoltage); "min" the week with the lowest
            # minimum-load multiplier, where the load floor is thinnest and PV
            # overvoltage is what binds. The two are different days of the year and
            # each writes into its own files_<extreme> scenario folder, so both can
            # share one --timestamp tree without colliding.
            if args.day_extreme == 'min':
                mult_index = int(np.argmin(min_mults))
                day_sim, mult = yearly_sim_gen_min, min_mults[mult_index]
            else:
                mult_index = int(np.argmax(max_mults))
                day_sim, mult = yearly_sim_gen_max, max_mults[mult_index]
            print(f"Single-day run: {args.day_extreme}-load envelope, "
                  f"week index {mult_index} (multiplier {mult:.4f})")
            for control in args.controls:
                loss = day_sim(mult_index=mult_index, mult=mult,
                               timestamp=timestamp, ControlType=control,
                               pid_gains=args.pid_gains, step_mult=args.step,
                               q_mode=args.q_mode, pv_file=pv_file,
                               droop_gains_file=droop_gains_file,
                               max_curtail_frac=args.max_curtail_frac)
                print(f'{CONTROL_NAMES[control]} Control Loses:{loss} kWh')
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        send_discord_alert(f"⚠️ **Error in yearly_tests.py!** The script crashed.\nError: `{e}`\n```python\n{error_msg[-1500:]}\n```")
        raise
