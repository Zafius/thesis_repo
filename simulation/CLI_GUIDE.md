# Running the simulation

`yearly_tests.py` is the single entry point. There are two ways to run it:

1. **`run.bat`** — the easy way: edit a small config block, then double-click. No typing.
2. **The command line** — for one-off runs and full control over the flags.

Both do exactly the same thing; `run.bat` just builds the command line for you.

---

## Option 1 — `run.bat` (recommended)

1. Open [run.bat](run.bat) in any text editor.
2. Change the values in the **CONFIG** block near the top:

   | Variable | What it sets | Example values |
   |---|---|---|
   | `LOCATION` | Simulation site: ENTSO-E load zone **and** PV point (optimum tilt/azimuth from PVGIS). One entry per ENTSO-E zone (63 total) in `sim_config.py` → `LOCATIONS`, keyed `<city>_<zonecode>` (plain `<city>` for single-zone countries). Nothing is pre-fetched — the load file and PV curve are fetched and cached under `locations/<Country>/<key>/` the first time a key is used | `oslo_no1` (default), `athens`, `madrid`, `rome_it`, `berlin_50hertz`, `paris`, `kristiansand_no2`, `stockholm_se3`, `london_gb`, `warsaw`, `vienna`, ... |
   | `CONTROLS` | Which strategies to run (space-separated) | `1 2 3`, `0`, `0 1 2 3` |
   | `YEARLY` | `0` = single worst-case day, `1` = full 53-week batch | `0` or `1` |
   | `PID_GAINS` | `Kp Ki Kd q_decay` (control type 0 only) | `0.07986 0.86629 0.03253 0.84110` |
   | `STEP` | Time-resolution multiplier: `1` = 15-min steps, `3` = 5-min, `15` = 1-min | `1` |
   | `Q_MODE` | `0` = supply Q at all times, `1` = supply Q only while an EV is plugged in | `0` or `1` |

3. Save, then **double-click `run.bat`** (or run `.\run.bat` in a terminal).

It finds the virtual environment automatically, prints the exact command it runs, and pauses at the end so you can read the output.

---

## Option 2 — command line

Run everything from inside the `simulation/` folder:

```bash
cd simulation
```

Then activate the venv once:

```bash
..\.venv\Scripts\activate          # Windows
```

…or just prefix each command with the venv's python (no activation needed):

```bash
..\.venv\Scripts\python yearly_tests.py --help
```

### Common runs

```bash
# Single worst-case day (fast). Prints daily losses in kWh to the screen.
python yearly_tests.py                      # default: PID only
python yearly_tests.py --controls 3         # No Control baseline only
python yearly_tests.py --controls 0 1 2 3   # all four strategies

# Full 53-week yearly batch (slow, uses multiprocessing across weeks).
python yearly_tests.py --yearly --controls 1        # Droop, all 53 weeks
python yearly_tests.py --yearly --controls 0 1 2 3  # full yearly comparison

# Custom PID gains
python yearly_tests.py --controls 0 --pid-gains 0.08 0.86 0.03 0.84

# Coarser time resolution than the default 1-minute steps
python yearly_tests.py --controls 3 --step 3     # 5-minute steps
python yearly_tests.py --controls 3 --step 1     # 15-minute steps
python yearly_tests.py --controls 3 --step 0.5   # 30-minute steps (49/day)

# Reactive-power support at all times, rather than only while plugged in
python yearly_tests.py --controls 0 1 2 --q-mode 0

# Different site (Athens load zone + Athens PV, optimum angles). The load file
# and PV curve are fetched/cached under locations/<Country>/<key>/ the first
# time each location is used.
python yearly_tests.py --controls 0 1 2 3 --location athens
python yearly_tests.py --yearly --controls 0 1 2 3 --location madrid
python yearly_tests.py --controls 0 1 2 3 --location kristiansand_no2   # Norway, NO2 zone
python yearly_tests.py --controls 0 1 2 3 --location london_gb          # United Kingdom, GB zone
```

### All flags

| Flag | Meaning | Default |
|---|---|---|
| `--location KEY` | Simulation site: any of the 63 keys in `sim_config.LOCATIONS` (or a key you add). Picks the ENTSO-E load zone's weekly max/min multipliers and the PVGIS PV curve for that zone's representative city, at its optimum tilt/azimuth. Both are fetched and cached under `locations/<Country>/<key>/` the first time a location is used | `oslo_no1` (`sim_config.LOCATION`) |
| `--controls N [N ...]` | Strategies to run: `0`=PID, `1`=Droop, `2`=Centralized, `3`=No Control | `0` (`yearly_tests.controlType`) |
| `--yearly` | Run all 53 weeks (max & min load) instead of the single worst-case day | off |
| `--pid-gains KP KI KD Q_DECAY` | The four PID values | the location+scenario tuned gains in `locations/<key>/gains/pid_<scenario>.csv` if present (written by `pid_optimizer_approx.py`), else built-in set |
| `--step FLOAT` | Time-resolution multiplier: `0.5` = 30-min steps (49/day), `1` = 15-min (97), `3` = 5-min, `15` = 1-min (1441). Higher = finer but proportionally slower; losses are normalized so results stay comparable | `15` (`sim_config.STEP_MULT`) |
| `--q-mode {0,1}` | Reactive-power supply mode for PID/Droop/Centralized: `0` = supply Q at all times (even with no EV plugged in), `1` = supply Q only while an EV is plugged in (a plugged EV that finished charging still supplies Q) | `1` (`sim_config.Q_MODE`) |
| `--max-curtail-frac F` | Largest share `[0-1]` of an inverter's scheduled active power it may give up to make current headroom for reactive support. `0` = never curtail P, `1` = reactive support always wins | `0.05` (`sim_config.MAX_P_CURTAIL_FRAC`) |
| `--timestamp NAME` | Output folder under `sim_results/`. Pass an existing value to make several invocations share one results tree (one per `--controls` value, for per-controller timings) | the current time, `%Y-%m-%d_%H.%M` |
| `-h`, `--help` | Show the built-in help and exit | — |

> **Defaults come from `sim_config.py`, so they are what that file currently
> says** — the table above records the values as shipped. `--controls`,
> `--step` and `--q-mode` were each documented here with a different default
> than the code used; if you are unsure, `--help` prints the live values.

Notes:
- The `--controls` order is preserved: `--controls 3 0` runs No Control first, then PID.
- `--pid-gains` matters only when `0` is in `--controls`.
- `--q-mode` is ignored for `--controls 3` (No Control never sets Q).
- **Everything a location needs lives in `locations/<Country>/<key>/`**: its load zone (`load_weekly_peak_min.csv`), PV curve (`pv_weekly.csv`), and tuned gains under `gains/`. `sim_config.LOCATIONS` has one entry per ENTSO-E zone (`locations/load_zones/`, 63 zones), grouped by country folder so the tree stays navigable — e.g. Norway's 5 bidding zones (NO1–NO5) plus its country total live under `locations/Norway/`. Nothing is pre-fetched: the load file (a local copy) and PV curve (fetched from PVGIS) only appear on disk the first time that key is used — by a run, an optimizer, or the "run the optimizer now?" prompt. Add a location by adding one entry to `sim_config.LOCATIONS` (region + lat/lon + country).
- **Tuned gains are cached per scenario.** Gain filenames encode the EV share, PV share and placement seed (e.g. `gains/pid_ev40_pv5_seed42.csv`), taken from `sim_config.EV_CONTRIBUTION` / `PV_CONTRIBUTION` / `NETWORK_SEED`. Change any of those in `sim_config.py` and re-tune (`pid_optimizer_approx.py --location <key>` / `droop_optimizer_approx.py --location <key>`, or `run_full_year.bat`): the new gains are saved alongside the old under a new name, and `yearly_tests.py` reads the file matching the current scenario.
- **Missing gains at run time prompt to tune.** If you run PID (`0`) or Droop (`1`) and no tuned gains exist for the current location+scenario, `yearly_tests.py` asks `[y/N]` whether to run the matching optimizer first (at the run's `--step`/`--q-mode`). Answer **y** to tune then run; answer **n** (or run non-interactively, e.g. piped/background) to proceed with default params — built-in PID gains, or the default location's droop gains with a warning.

---

## Preparing a location

Most of the 63 configured locations have no data on disk. `yearly_tests.py`
builds one implicitly the first time it needs it, which means a run can stall
for up to three minutes on a PVGIS request with nothing to say why. To do it
up front, and get a clear answer when it fails:

```bash
python build_location.py --location lisbon --check   # report only, change nothing
python build_location.py --location lisbon           # copy the load zone, fetch the PV curve
```

Exit codes distinguish the two ways it can fail, because they need different
fixes: **2** = PVGIS could not be reached (a network problem, retry later),
**3** = there is no ENTSO-E load-zone file for that location's region (nothing
to retry — the file has to be added, see `entsoe/build_yearly_load.py`). **0**
is returned only after both files are confirmed on disk, so a truncated fetch
is reported rather than mistaken for success.

## Where results go

By default, results are written under `simulation/sim_results/`. Set
`THESIS_SIM_RESULTS` to put them somewhere else — a full yearly tree is ~16 GB
before compression, and the drive holding the code is often not where that
belongs:

```bash
set THESIS_SIM_RESULTS=D:\thesis-results
python yearly_tests.py --yearly --controls 3
```

The analysis scripts take `--output-dir` separately, so the summaries can live
apart from the raw trees.

Results are written under a timestamped folder named `YYYY-MM-DD_HH.MM`:

| Location | Written by | Contents |
|---|---|---|
| `sim_results/<timestamp>/…` | every run (including `--yearly`, one subfolder per week) | Per-week: per-phase min/max voltage CSVs, the full per-bus voltage trajectory (`global_voltages/all_bus_voltages_phase_*.csv`, archival only — stored gzipped, see below), a losses CSV + PDF, global per-inverter P/Q/V CSVs (one column per inverter), `voltage_quality.csv` (exact all-LV-bus violation/excursion/VUF figures), `transformer_loading.csv` (MV/LV substation loading, see below), and a combined 3-phase voltage PDF |

For the annual total, run `compare_control_strategies.py --root sim_results/<timestamp>`, which integrates every week's `simulation_results_losses.csv` into `total_active_losses_kwh` per strategy, plus the additional metrics below, and writes `strategy_summary.csv` + PNGs to `analysis_output/` (`--output-dir` to change).

A single-day run also prints its result to the terminal, e.g. `No Control Loses:684.02 kWh`, alongside the voltage-quality figures and the transformer's peak loading for that run.

### Shrinking a finished result tree

A `--yearly` tree is ~16 GB, and ~98% of that is the three `global_voltages/all_bus_voltages_phase_{A,B,C}.csv` files per week — the full per-bus voltage trajectory, which is written as archival evidence and which no analysis script reads back. `compress_results.py` gzips those and deletes the originals (~4.9x, so ~16 GB → ~3.5 GB); every file the analysis scripts *do* open stays plain CSV, so nothing else changes.

```bash
python compress_results.py sim_results/<timestamp>     # one tree
python compress_results.py --all                       # everything eligible
python compress_results.py --all --dry-run             # report only
python compress_results.py sim_results/<ts> --decompress   # put the CSVs back
```

`run_test_plan.py` does this automatically as a tracked `compress` task once a batch's `compare` and `curtail` steps have both succeeded. Each file's round trip is SHA-256 verified before the original is deleted, and `--all` skips trees modified in the last 15 minutes so it can't touch a run still in progress.

### Additional tracked metrics

Beyond losses and voltage quality, `compare_control_strategies.py` computes and adds these columns to `strategy_summary.csv` (all derived from the per-run CSVs above, no extra simulation runs needed):

| Metric | Column(s) | What it measures |
|---|---|---|
| **MV/LV transformer loading** | `transformer_loading_max_pct`, `transformer_loading_mean_pct`, `transformer_pct_time_over_100pct`, `transformer_peak_s_kva` | Whether the controllers' Q injection/absorption causes congestion at the substation transformer (apparent power S), independent of whether LV-bus voltages stayed in band. Read from `transformer_loading.csv` (pandapower's `res_trafo_3ph`, logged per step in `network_setup.py`). |
| **Reactive-power fairness** | `reactive_fairness_gini`, `reactive_fairness_cv` | How evenly the total Q effort is shared across chargers over the whole scenario: a Gini index (0 = perfectly even, 1 = one inverter does all the work) and the coefficient of variation. Most informative for the Centralized controller, which can concentrate effort on whichever inverters its sensitivity matrix favors. |
| **Estimated additional inverter losses** | `estimated_additional_inverter_losses_kwh` | Extra energy the chargers' inverters consume specifically because they supply Q, from a simplified I²R conduction-loss model (`Loss(S) = k·S²`, so the Q-attributable share is `k·Q²`). `k` is calibrated from `sim_config.INVERTER_RATED_CONDUCTION_LOSS_PCT` (default 1.5% loss at rated apparent power) -- an assumption, not a measured hardware figure; treat the kWh values as relative across strategies rather than absolute. |
| **Sign-flip frequency** | `sign_flips_total`, `sign_flips_mean_per_inverter_per_day` | Number of times a charger's Q switches between capacitive and inductive per inverter (a DC-link capacitor stress proxy), counted with a `sim_config.SIGN_FLIP_DEADBAND_KVAR` deadband around zero so noise/dither isn't counted as switching. |
| **Hosting capacity gain** | separate `hosting_capacity_summary.csv` | See `hosting_capacity_sweep.py` below -- not part of `strategy_summary.csv` since it requires its own sweep of simulations, not just aggregating one completed batch. |

---

## Hosting-capacity sweep

`hosting_capacity_sweep.py` estimates the **Hosting Capacity Gain**: how much higher an EV penetration percentage each control strategy tolerates, versus No Control, before voltage violations exceed a threshold (default 5% of all LV-bus/phase/step measurements, `sim_config.HOSTING_CAPACITY_VIOLATION_THRESHOLD_PCT`).

For each requested strategy it binary-searches `EV_CONTRIBUTION` (the fraction of network loads fitted with a charger) for the largest value that stays within the threshold on the single worst-case day, then reports the gain over the No Control baseline in percentage points.

```bash
python hosting_capacity_sweep.py --controls 0 1 2                    # PID, Droop, Centralized vs No Control
python hosting_capacity_sweep.py --controls 1 2 --location athens --ev-max 0.8
python hosting_capacity_sweep.py --controls 0 1 2 --violation-threshold-pct 2
```

**Read the module docstring before citing these numbers.** In short:
- PID/Droop gains are **reused** from whatever is already tuned for the current location+scenario at every EV% tested, not retuned per point (retuning at every point would multiply runtime by the DE optimizer's own cost per point). The Centralized controller is unaffected since it re-estimates its sensitivity matrix online every run.
- Each point is evaluated on the single worst-case day, not the full 53-week batch.
- The bisection assumes violations are monotonically non-decreasing in EV penetration.

Output: `analysis_output/hosting_capacity_summary.csv` (`hosting_capacity_ev_pct`, `violation_pct_at_capacity`, `hosting_capacity_gain_pct_points`, `censored`) and `analysis_output/hosting_capacity_bar.png`. Every evaluated point also writes its usual per-run simulation output under `sim_results/hosting_capacity_<timestamp>_<strategy>/...` -- expect several small result folders per strategy, one per bisection step.

## Control types at a glance

| # | Strategy | How it decides reactive power (Q) |
|---|---|---|
| `0` | **PID** | Per-phase discrete PID loop driving voltage back into the deadband |
| `1` | **Droop** | Per-inverter voltage-vs-Q droop curve; gains read from `locations/<key>/gains/droop_<scenario>.csv` |
| `2` | **Centralized** | Solves a bounded optimization each step using a dV/dQ sensitivity matrix |
| `3` | **No Control** | Q fixed at zero — the baseline for comparison |
