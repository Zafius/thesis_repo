# EV/PV reactive-power voltage control — simulation code

This package contains everything needed to **run** the thesis simulation: all
scripts, configuration, and the cached input/reproducibility data (network
load data, PV curves, EV charging sessions, tuned controller gains). It
contains **no simulation results** — no per-run output, no aggregated metrics,
no plots. The corresponding results package (produced by running this code)
is distributed separately.

## What it does

Uses [pandapower](https://pandapower.readthedocs.io/) to run 3-phase
unbalanced power-flow simulations (`runpp_3ph`) on the IEEE European LV test
feeder (55 single-phase loads) under EV charging and rooftop PV, and compares
four reactive-power (Q) control strategies for the EV/PV inverters:

* **PID Control** — per-phase discrete PID loop driving the local bus voltage
  back into a deadband, with integral decay so Q bleeds back to zero once
  voltage is back in band.
* **Droop Control** — per-inverter voltage-vs-Q droop curve, gains tunable
  individually per inverter.
* **Centralized Optimisation** — a bounded least-squares solve every step over
  a dV/dQ sensitivity matrix re-estimated periodically; no offline tuning.
* **No Control (baseline)** — Q held at zero.

Controller gains (PID, droop) are tuned offline per location/scenario by
`pid_optimizer_approx.py` / `droop_optimizer_approx.py`, using a fast linear
surrogate model instead of full power-flow simulation, then verified against
one real run.

## Getting started

```bash
# from this folder
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r simulation/requirements.txt

cd simulation
python yearly_tests.py --controls 0 1 2 3   # single worst-case day, all 4 strategies
```

Full usage — every script, every flag, `run.bat`'s config block, and what each
one writes — is documented in **[simulation/CLI_GUIDE.md](simulation/CLI_GUIDE.md)**.
`thesis_test_plan.md` (this folder) records how the thesis results were
produced, and `simulation/run_test_plan.bat` / `run_test_plan.py` re-run that
plan (resumable; it writes a `TEST_PLAN_EXECUTION.md` progress index as it goes).

## Locations & cached data

Each simulation site (ENTSO-E load zone + PVGIS solar point) lives under
`simulation/locations/<Country>/<key>/`: a weekly load-multiplier curve, a
weekly PV-output curve, and (once tuned) that location's PID/droop gains per
penetration scenario. The three locations used in the thesis
(`oslo_no1`, `athens`, `berlin_50hertz`) ship here fully pre-fetched and
pre-tuned, so re-running their simulations needs no network access and no
re-tuning. Any other ENTSO-E zone can be added by name — `sim_config.py`
fetches and caches it on first use — but will need its own gain-tuning pass
(`pid_optimizer_approx.py` / `droop_optimizer_approx.py`) before `yearly_tests.py`
can use it without a fallback-gains warning.

EV charging demand is sampled from real apartment-building charging sessions
(Baek, Lee & Kim, 2024 — https://doi.org/10.6084/m9.figshare.22495141),
shipped already preprocessed as per-session profile files under
`simulation/filtered_apartments_2h/{night,morning,midday}/` (the raw dataset
and the preprocessing script are not included).

## Requirements

See [simulation/requirements.txt](simulation/requirements.txt): `pandapower`
+ `lightsim2grid` (3-phase power flow), `pandas`/`numpy` (data handling),
`scipy`/`cma` (gain optimisation), `matplotlib`/`seaborn` (plotting),
`python-dotenv` (optional Discord run-notification webhook).
