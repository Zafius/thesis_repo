"""
sim_config.py -- single source of truth for the simulation defaults.

Every script in this folder (yearly_tests.py, pid_optimizer_approx.py,
droop_optimizer_approx.py, network_setup.py, custom_controller_example.py,
surrogate_objective.py, sensitivity.py) reads its defaults from here, so the
gain optimizers, the yearly batch runs and the controllers are guaranteed to
agree on the network build, the time resolution and the physical constants.

These values are only ever overridden from the command line -- in practice by
run.bat, which passes explicit --step / --q-mode / --controls flags. Running
any script without flags uses exactly what is defined below.

The thesis states (Ch. 3) that controller gains are only valid for the time
step T_s they were tuned at; keeping STEP_MULT here, imported by both the
optimizers and yearly_tests.py, is what enforces that consistency.
"""

import os
import shutil
import subprocess
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Simulation location (load zone + PV point)
# ---------------------------------------------------------------------------
# A "location" binds together the geographically-dependent inputs of the
# simulation, and every per-location artefact lives in one self-contained
# folder, locations/<Country>/<key>/ (grouped by country so the tree stays
# navigable with 60+ locations), so adding a location never clutters the rest
# of the tree:
#   locations/
#     load_zones/                         -- the ENTSO-E load-zone library to
#                                            pick a 'region' from (one file per
#                                            zone, from entsoe/build_yearly_load.py)
#     <Country>/<key>/
#       load_weekly_peak_min.csv          -- this location's zone file; column
#                                            Peak[pu] drives the max-load runs,
#                                            Min[pu] the min-load runs
#       pv_weekly.csv                     -- weekly PV curve for the location's
#                                            lat/lon at PVGIS optimum tilt/azimuth
#                                            (pv_data/fetch_pvgis_timeseries.py)
#       gains/                            -- tuned gains, one file per scenario
#         pid_<scenario>.csv              -- PID gains (pid_optimizer_approx.py)
#         droop_<scenario>.csv            -- droop gains (droop_optimizer_approx.py)
#         droop_profiles_<scenario>.csv   -- droop optimizer diagnostic dump
#       where <scenario> = evNN_pvNN_seedNN (EV share, PV share, placement seed;
#       see scenario_tag) so each penetration/seed combination is cached apart.
#
# LOCATIONS below has one entry per ENTSO-E load zone reported in entsoe/
# (locations/load_zones/, 63 zones after dropping one exact duplicate),
# keyed by <city>_<zone code> (or plain <city> where a country reports only one
# zone) and pointing at a central city inside that zone: e.g. Norway's five
# bidding zones NO1..NO5 map to oslo_no1/kristiansand_no2/trondheim_no3/
# tromso_no4/bergen_no5, each nested under locations/Norway/. 'region' names a
# file in load_zones/, lat/lon fix the PV point.
#
# NONE of this is pre-fetched: adding a LOCATIONS entry costs nothing until it
# is actually used. ensure_location() materialises a location's
# load_weekly_peak_min.csv (a local copy from load_zones/, instant) and
# pv_weekly.csv (fetched from PVGIS over the network, ~seconds) the first time
# that key is requested -- by yearly_tests.py, the gain optimizers, or
# ensure_tuned_gains()'s "run the optimizer now?" prompt.
#
# Pick the active location with LOCATION, or override per run with the
# --location flag every script accepts. 'oslo_no1' (the default) reproduces
# the original dataset (NO1 load zone + the SA3 Oslo rooftop).
LOCATION = 'oslo_no1'

LOCATIONS = {
    # -- Norway (NO): 5 bidding zones + country total -----------------------
    'oslo_no1':          {'region': '2020_NO1',          'lat': 59.961, 'lon': 10.868, 'country': 'Norway'},
    'kristiansand_no2':  {'region': '2020_NO2',          'lat': 58.146, 'lon':  7.998, 'country': 'Norway'},
    'trondheim_no3':     {'region': '2020_NO3',          'lat': 63.430, 'lon': 10.395, 'country': 'Norway'},
    'tromso_no4':        {'region': '2020_NO4',          'lat': 69.649, 'lon': 18.956, 'country': 'Norway'},
    'bergen_no5':        {'region': '2020_NO5',          'lat': 60.393, 'lon':  5.324, 'country': 'Norway'},
    'oslo_no':           {'region': '2020_Norway_NO',    'lat': 59.961, 'lon': 10.868, 'country': 'Norway'},

    # -- Sweden (SE): 4 bidding zones + country total ------------------------
    'lulea_se1':         {'region': '2020_SE1',          'lat': 65.585, 'lon': 22.156, 'country': 'Sweden'},
    'sundsvall_se2':     {'region': '2020_SE2',          'lat': 62.391, 'lon': 17.306, 'country': 'Sweden'},
    'stockholm_se3':     {'region': '2020_SE3',          'lat': 59.329, 'lon': 18.069, 'country': 'Sweden'},
    'malmo_se4':         {'region': '2020_SE4',          'lat': 55.605, 'lon': 13.000, 'country': 'Sweden'},
    'stockholm_se':      {'region': '2020_Sweden_SE',    'lat': 59.329, 'lon': 18.069, 'country': 'Sweden'},

    # -- Denmark (DK): 2 bidding zones + country total (2020_DK is an exact
    #    duplicate of 2020_Denmark_DK and is intentionally not mapped here) --
    'aarhus_dk1':        {'region': '2020_DK1',          'lat': 56.163, 'lon': 10.203, 'country': 'Denmark'},
    'copenhagen_dk2':    {'region': '2020_DK2',          'lat': 55.676, 'lon': 12.568, 'country': 'Denmark'},
    'copenhagen_dk':     {'region': '2020_Denmark_DK',   'lat': 55.676, 'lon': 12.568, 'country': 'Denmark'},

    # -- Germany (DE): 4 TSO control areas + the DE-LU bidding zone + country total
    'frankfurt_delu':        {'region': '2020_DE-LU',           'lat': 50.110, 'lon':  8.682, 'country': 'Germany'},
    'berlin_50hertz':        {'region': '2020_DE_50Hertz',      'lat': 52.520, 'lon': 13.405, 'country': 'Germany'},
    'dortmund_amprion':      {'region': '2020_DE_Amprion',      'lat': 51.514, 'lon':  7.466, 'country': 'Germany'},
    'hannover_tennet':       {'region': '2020_DE_TenneT_GER',   'lat': 52.375, 'lon':  9.732, 'country': 'Germany'},
    'stuttgart_transnetbw':  {'region': '2020_DE_TransnetBW',   'lat': 48.775, 'lon':  9.183, 'country': 'Germany'},
    'berlin_de':              {'region': '2020_Germany_DE',      'lat': 52.520, 'lon': 13.405, 'country': 'Germany'},

    # -- Italy (IT): 7 zones + country total ---------------------------------
    'milan_it_north':          {'region': '2020_IT-North',         'lat': 45.464, 'lon':  9.190, 'country': 'Italy'},
    'florence_it_centre_north':{'region': '2020_IT-Centre-North',  'lat': 43.769, 'lon': 11.256, 'country': 'Italy'},
    'rome_it_centre_south':    {'region': '2020_IT-Centre-South',  'lat': 41.902, 'lon': 12.496, 'country': 'Italy'},
    'naples_it_south':         {'region': '2020_IT-South',         'lat': 40.852, 'lon': 14.268, 'country': 'Italy'},
    'palermo_it_sicily':       {'region': '2020_IT-Sicily',        'lat': 38.116, 'lon': 13.362, 'country': 'Italy'},
    'cagliari_it_sardinia':    {'region': '2020_IT-Sardinia',      'lat': 39.223, 'lon':  9.122, 'country': 'Italy'},
    'catanzaro_it_calabria':   {'region': '2020_IT-Calabria',      'lat': 38.910, 'lon': 16.597, 'country': 'Italy'},
    'rome_it':                 {'region': '2020_Italy_IT',         'lat': 41.902, 'lon': 12.496, 'country': 'Italy'},

    # -- United Kingdom: GB and the whole-UK series are distinct ENTSO-E
    #    series (not duplicates); Northern Ireland (NIE) is politically UK
    #    though its market is the island-of-Ireland SEM -------------------
    'london_gb':          {'region': '2020_GB',                'lat': 51.507, 'lon': -0.128, 'country': 'United_Kingdom'},
    'london_uk':          {'region': '2020_United_Kingdom_UK', 'lat': 51.507, 'lon': -0.128, 'country': 'United_Kingdom'},
    'belfast_nie':        {'region': '2020_NIE',                'lat': 54.597, 'lon': -5.930, 'country': 'United_Kingdom'},

    # -- Ireland: the SEM (island-wide) and the IE-only series are distinct --
    'dublin_sem':         {'region': '2020_IE_SEM',      'lat': 53.350, 'lon': -6.260, 'country': 'Ireland'},
    'dublin_ie':          {'region': '2020_Ireland_IE',  'lat': 53.350, 'lon': -6.260, 'country': 'Ireland'},

    # -- Ukraine: the Burshtyn "energy island" (UA-BEI) is a separate
    #    synchronous area from the main IPS grid -----------------------------
    'ivanofrankivsk_uabei': {'region': '2020_UA-BEI',    'lat': 48.923, 'lon': 24.711, 'country': 'Ukraine'},
    'kyiv_uaips':           {'region': '2020_UA-IPS',    'lat': 50.450, 'lon': 30.523, 'country': 'Ukraine'},
    'kyiv_ua':              {'region': '2020_Ukraine_UA','lat': 50.450, 'lon': 30.523, 'country': 'Ukraine'},

    # -- Single-zone countries (one ENTSO-E series each; key = capital city) -
    'vienna':          {'region': '2020_Austria_AT',        'lat': 48.208, 'lon': 16.373, 'country': 'Austria'},
    'brussels':        {'region': '2020_Belgium_BE',        'lat': 50.850, 'lon':  4.352, 'country': 'Belgium'},
    'sarajevo':        {'region': '2020_Bosnia_and_Herz_BA','lat': 43.856, 'lon': 18.413, 'country': 'Bosnia_and_Herzegovina'},
    'sofia':           {'region': '2020_Bulgaria_BG',       'lat': 42.698, 'lon': 23.319, 'country': 'Bulgaria'},
    'zagreb':          {'region': '2020_Croatia_HR',        'lat': 45.815, 'lon': 15.982, 'country': 'Croatia'},
    'nicosia':         {'region': '2020_Cyprus_CY',         'lat': 35.170, 'lon': 33.360, 'country': 'Cyprus'},
    'prague':          {'region': '2020_Czech_Republic_CZ', 'lat': 50.076, 'lon': 14.438, 'country': 'Czech_Republic'},
    'tallinn':         {'region': '2020_Estonia_EE',        'lat': 59.437, 'lon': 24.754, 'country': 'Estonia'},
    'helsinki':        {'region': '2020_Finland_FI',        'lat': 60.169, 'lon': 24.938, 'country': 'Finland'},
    'paris':           {'region': '2020_France_FR',         'lat': 48.857, 'lon':  2.352, 'country': 'France'},
    'athens':          {'region': '2020_Greece_GR',         'lat': 37.983, 'lon': 23.728, 'country': 'Greece'},
    'budapest':        {'region': '2020_Hungary_HU',        'lat': 47.498, 'lon': 19.040, 'country': 'Hungary'},
    'riga':            {'region': '2020_Latvia_LV',         'lat': 56.950, 'lon': 24.106, 'country': 'Latvia'},
    'vilnius':         {'region': '2020_Lithuania_LT',      'lat': 54.687, 'lon': 25.280, 'country': 'Lithuania'},
    'luxembourg_city': {'region': '2020_Luxembourg_LU',     'lat': 49.611, 'lon':  6.131, 'country': 'Luxembourg'},
    'chisinau':        {'region': '2020_Moldova_MD',        'lat': 47.011, 'lon': 28.862, 'country': 'Moldova'},
    'podgorica':       {'region': '2020_Montenegro_ME',     'lat': 42.441, 'lon': 19.263, 'country': 'Montenegro'},
    'skopje':          {'region': '2020_North_Macedonia_MK','lat': 41.997, 'lon': 21.428, 'country': 'North_Macedonia'},
    'amsterdam':       {'region': '2020_Netherlands_NL',    'lat': 52.367, 'lon':  4.904, 'country': 'Netherlands'},
    'warsaw':          {'region': '2020_Poland_PL',         'lat': 52.230, 'lon': 21.011, 'country': 'Poland'},
    'lisbon':          {'region': '2020_Portugal_PT',       'lat': 38.722, 'lon': -9.139, 'country': 'Portugal'},
    'bucharest':       {'region': '2020_Romania_RO',        'lat': 44.427, 'lon': 26.104, 'country': 'Romania'},
    'belgrade':        {'region': '2020_Serbia_RS',         'lat': 44.787, 'lon': 20.457, 'country': 'Serbia'},
    'bratislava':      {'region': '2020_Slovakia_SK',       'lat': 48.148, 'lon': 17.107, 'country': 'Slovakia'},
    'ljubljana':       {'region': '2020_Slovenia_SI',       'lat': 46.056, 'lon': 14.505, 'country': 'Slovenia'},
    'madrid':          {'region': '2020_Spain_ES',          'lat': 40.417, 'lon': -3.703, 'country': 'Spain'},
    'bern':            {'region': '2020_Switzerland_CH',    'lat': 46.948, 'lon':  7.447, 'country': 'Switzerland'},
}

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOCATIONS_DIR = os.path.join(_THIS_DIR, 'locations')
LOAD_ZONES_DIR = os.path.join(LOCATIONS_DIR, 'load_zones')

# Where raw run trees are written. A full yearly batch is ~16 GB before
# compression, and the drive holding the code is often not where that belongs,
# so this can be pointed elsewhere with THESIS_SIM_RESULTS.
#
# An environment variable rather than a flag: nothing on the command line
# reaches network_setup.py's writer without threading a new argument through
# yearly_tests -> Start_sim_test -> simulation_with_dynamic_gen, and every
# script inherits an env var for free. It also matches how the desktop app
# already overrides the checkout and the interpreter (THESIS_REPO_ROOT,
# THESIS_PYTHON). Unset keeps the historical layout exactly.
SIM_RESULTS_DIR = os.environ.get('THESIS_SIM_RESULTS') or os.path.join(_THIS_DIR, 'sim_results')


def get_location(key=None):
    """Return the {region, lat, lon, country} dict for a location key (default LOCATION)."""
    key = key or LOCATION
    if key not in LOCATIONS:
        raise ValueError(f"Unknown location {key!r}; choose one of "
                         f"{sorted(LOCATIONS)} or add it to sim_config.LOCATIONS")
    return LOCATIONS[key]


def location_dir(key=None):
    """Folder holding every artefact for a location key, grouped by country
    (locations/<Country>/<key>/)."""
    loc = get_location(key)
    return os.path.join(LOCATIONS_DIR, loc['country'], key or LOCATION)


def zone_library_path(region):
    """Path to an ENTSO-E zone file in the shared load_zones/ library."""
    return os.path.join(LOAD_ZONES_DIR, f'{region}_weekly_peak_min.csv')


def region_load_path(key=None):
    """Path to a location's weekly peak/min load file (Peak[pu]/Min[pu])."""
    return os.path.join(location_dir(key), 'load_weekly_peak_min.csv')


def pv_source_path(key=None):
    """Path to a location's cached weekly PV table."""
    return os.path.join(location_dir(key), 'pv_weekly.csv')


def scenario_tag(ev=None, pv=None, seed=None):
    """Scenario key for tuned gains: EV share, PV share and placement seed.

    Tuned gains are only valid for the scenario they were tuned on -- the EV and
    PV penetration set how much P/Q the feeder sees, and the network seed fixes
    where the chargers and panels are placed. Encoding all three in the filename
    (e.g. ev40_pv5_seed42) lets each combination be cached side by side instead
    of overwriting, so switching EV_CONTRIBUTION / PV_CONTRIBUTION / NETWORK_SEED
    reuses previously tuned gains. Defaults come from the active sim_config
    values, so callers that don't sweep get today's scenario automatically.
    """
    ev = EV_CONTRIBUTION if ev is None else ev
    pv = PV_CONTRIBUTION if pv is None else pv
    seed = NETWORK_SEED if seed is None else seed
    return f"ev{int(round(ev * 100))}_pv{int(round(pv * 100))}_seed{int(seed)}"


def _gains_dir(key=None):
    """Per-location folder holding the scenario-tagged tuned-gain files."""
    return os.path.join(location_dir(key), 'gains')


def pid_gains_path(key=None, ev=None, pv=None, seed=None):
    """Path to a location+scenario's tuned PID gains (pid_optimizer_approx.py)."""
    return os.path.join(_gains_dir(key), f'pid_{scenario_tag(ev, pv, seed)}.csv')


def droop_gains_path(key=None, ev=None, pv=None, seed=None):
    """Path to a location+scenario's tuned droop gains (droop_optimizer_approx.py)."""
    return os.path.join(_gains_dir(key), f'droop_{scenario_tag(ev, pv, seed)}.csv')


def droop_profiles_path(key=None, ev=None, pv=None, seed=None):
    """Path to a location+scenario's droop-optimizer verification profile dump."""
    return os.path.join(_gains_dir(key), f'droop_profiles_{scenario_tag(ev, pv, seed)}.csv')


def ensure_location(key=None):
    """Make sure a location's load and PV inputs exist on disk; return the key.

    Copies the ENTSO-E zone file into locations/<key>/load_weekly_peak_min.csv
    and, if the PV cache is missing, fetches it from PVGIS at the location's
    optimum tilt/azimuth (pv_data/fetch_pvgis_timeseries.py --location <key>
    --build). Idempotent, so every entry point can call it cheaply on startup.
    """
    key = key or LOCATION
    loc = get_location(key)
    os.makedirs(location_dir(key), exist_ok=True)

    load_dst = region_load_path(key)
    if not os.path.exists(load_dst):
        src = zone_library_path(loc['region'])
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Load-zone file for location {key!r} not found: {src}. "
                f"Add {loc['region']}_weekly_peak_min.csv to {LOAD_ZONES_DIR} "
                f"(see entsoe/build_yearly_load.py).")
        shutil.copyfile(src, load_dst)

    if not os.path.exists(pv_source_path(key)):
        print(f"PV cache for location {key!r} missing; fetching from PVGIS ...")
        subprocess.run(
            [sys.executable,
             os.path.join(_THIS_DIR, 'pv_data', 'fetch_pvgis_timeseries.py'),
             '--location', key, '--build'],
            check=True)
    return key


# ---------------------------------------------------------------------------
# Randomness
# ---------------------------------------------------------------------------
SEED = 42          # general RNG seed (Differential Evolution search, batch order)
NETWORK_SEED = 42  # EV/PV placement and charging-profile sampling; fixed so
                   # every script rebuilds the identical network and day

# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------
EV_CONTRIBUTION = 0.2   # EV chargers as a fraction of the number of loads
PV_CONTRIBUTION = 0.05  # PV units as a fraction of the number of loads

# Time-resolution multiplier: 1 = 15-minute steps (96/day), 3 = 5-minute
# steps, 15 = 1-minute steps. Shared by the yearly runs AND the gain
# optimizers so gains are tuned at the resolution they are used at.
STEP_MULT = 15

# Reactive-power supply mode (custom_controller_example.Q_MODE_*):
# 0 = inverters may supply Q at all times, even with no vehicle plugged in;
# 1 = inverters supply Q only while an EV is plugged in (the session's
#     connection window -- the plugged_in profile column). A plugged vehicle
#     that has finished charging (P = 0) keeps supporting the grid.
Q_MODE = 1

# EV charging time-of-day window: which pre-sorted bucket of measured 2-hour
# charging sessions the chargers sample from (sub-folders under
# filtered_apartments_2h/):
#   "night"   -- evening / overnight sessions (the worst case for LV feeders)
#   "morning" -- morning sessions
#   "midday"  -- midday sessions (coincides with PV, partially self-cancelling)
#   "all"     -- pool all three buckets and sample across the whole day
EV_LOAD_WINDOW = "all"

# The individual buckets that exist on disk; "all" pools every one of them.
EV_LOAD_WINDOWS = ("night", "morning", "midday")


def ev_load_buckets(window=None):
    """
    Directory names under filtered_apartments_2h/ to draw EV charging sessions
    from for the requested window. Defaults to EV_LOAD_WINDOW; "all" pools
    every bucket, a single window returns just that one.
    """
    window = (window or EV_LOAD_WINDOW).lower()
    if window == "all":
        return list(EV_LOAD_WINDOWS)
    if window not in EV_LOAD_WINDOWS:
        raise ValueError(
            f"EV_LOAD_WINDOW must be one of {EV_LOAD_WINDOWS + ('all',)}, "
            f"got {window!r}")
    return [window]


# ---------------------------------------------------------------------------
# Charger / inverter physical constants (thesis Ch. 2)
# ---------------------------------------------------------------------------
I_MAX_A = 32.0                        # rated charger/inverter current [A]
V_LN_NOMINAL_KV = 0.416 / np.sqrt(3)  # phase-to-neutral base voltage [kV]

# Maximum fraction of an inverter's scheduled ACTIVE power that may be curtailed
# to make current headroom for reactive support. Bounds the "reactive-power
# priority" strategy: an inverter can trade at most this share of its charging/
# PV output for Q; once that cap is hit, Q is reduced instead of curtailing P
# further. Range [0, 1]:
#   1.0 = curtail P freely (Q always wins, the original behaviour),
#   0.2 = give up at most 20% of P, then Q yields,
#   0.0 = never curtail P (Q is fully sacrificed to the current limit).
# All three controllers read this (custom_controller_example.py); set it here
# before a run, or override per run via yearly_tests.py --max-curtail-frac.
MAX_P_CURTAIL_FRAC = 0.05

# ---------------------------------------------------------------------------
# Voltage band (p.u.) -- the design deadband every controller and objective
# term is built around (thesis Ch. 1 / Ch. 3)
# ---------------------------------------------------------------------------
V_LOW, V_HIGH = 0.97, 1.05

# Reactive-power perturbation for the sensitivity sweep (thesis Ch. 3): 1 kVAr
DELTA_Q_MVAR = 0.001

# ---------------------------------------------------------------------------
# Sensitivity-matrix re-estimation cadence
#
# BOTH consumers of the dV/dQ matrix follow this: the centralized controller
# re-estimates its S_Q online every SENS_REESTIMATE_HOURS of simulated time
# and holds it constant in between, and the gain optimizers' surrogate uses
# anchor matrices computed at exactly the same instants with the same hold.
# One cadence, one behavior (thesis Ch. 3).
# ---------------------------------------------------------------------------
SENS_REESTIMATE_HOURS = 4


def sens_anchor_steps(step_mult):
    """
    Number of simulation steps between sensitivity re-estimations for a given
    time-resolution multiplier: 4 base steps/hour (15-min data) times the
    multiplier, times SENS_REESTIMATE_HOURS. Never below 1.
    """
    return max(1, int(4 * SENS_REESTIMATE_HOURS * step_mult))


# ---------------------------------------------------------------------------
# Inverter conduction-loss model (compare_control_strategies.py, "Estimated
# Additional Inverter Losses"): a simplified I^2R model where conduction loss
# is proportional to the square of the apparent power delivered,
# Loss(S) = k * S^2. Supplying P alone would cost k*P^2; supplying (P, Q)
# costs k*(P^2+Q^2) = k*P^2*(1+(Q/P)^2) -- the "loss increases by a factor of
# 1+(Q/P)^2" relation the metric is specified against -- so the EXTRA loss
# caused specifically by reactive-power provision is k*Q^2 regardless of P.
# Computing it this way (rather than literally evaluating (Q/P)^2) also
# sidesteps that ratio's singularity at P=0, e.g. an idle charger still
# supplying Q under Q_MODE_ALWAYS.
#
# k is calibrated so a charger delivering its full rated apparent power
# S_max_rated at unity power factor loses INVERTER_RATED_CONDUCTION_LOSS_PCT
# percent of that power to conduction -- a typical single-phase EVSE/inverter
# datasheet figure (order 1-2%), not measured for this thesis's hardware, so
# treat the resulting kWh figures as a relative comparison across control
# strategies rather than an absolute loss measurement.
# ---------------------------------------------------------------------------
INVERTER_RATED_CONDUCTION_LOSS_PCT = 1.5


def inverter_conduction_loss_coeff_kw_per_kvar2(i_max_a=I_MAX_A, v_ln_nominal_kv=V_LN_NOMINAL_KV,
                                                loss_pct=INVERTER_RATED_CONDUCTION_LOSS_PCT):
    """
    k [kW / kVAr^2] in the Loss(S) = k * S^2 conduction-loss model above,
    derived from the rated apparent power S_max_rated = V_LN_NOMINAL_KV *
    I_MAX_A [kVA] and the assumed loss fraction at that rating.
    """
    s_max_rated_kva = v_ln_nominal_kv * i_max_a  # kV * A = kVA
    return (loss_pct / 100.0) / s_max_rated_kva


# ---------------------------------------------------------------------------
# Hosting-capacity sweep default (hosting_capacity_sweep.py): the voltage
# violation percentage (of all LV-bus/phase/step measurements -- the same
# figure compare_control_strategies.voltage_violation_percentage reports)
# above which an EV penetration level is considered infeasible.
# ---------------------------------------------------------------------------
HOSTING_CAPACITY_VIOLATION_THRESHOLD_PCT = 5.0

# Sign-flip frequency (compare_control_strategies.py): |Q| below this deadband
# is treated as "not delivering reactive power" rather than as a sign, so
# controller/measurement noise dithering around zero isn't counted as
# repeated capacitive/inductive switching.
SIGN_FLIP_DEADBAND_KVAR = 0.05
