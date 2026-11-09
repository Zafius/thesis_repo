@echo off
REM ===========================================================================
REM  run.bat  -  Edit the CONFIG block below, then double-click this file
REM             (or run it from a terminal) to launch the simulation.
REM
REM  You do NOT need to activate the virtual environment or type any flags.
REM  See CLI_GUIDE.md for the full explanation of every option.
REM ===========================================================================

REM ------------------------------ CONFIG -------------------------------------
REM The values set here OVERRIDE the shared defaults in sim_config.py.
REM Running any script directly (without this .bat) uses sim_config.py as-is.

REM Simulation location: sets BOTH the ENTSO-E load zone (weekly max/min load
REM multipliers) AND the PV point, whose optimum tilt/azimuth PVGIS picks.
REM Nothing is pre-fetched: the load file and PV curve are fetched and cached
REM under locations/<Country>/<key>/ the first time a key is used.
REM One entry per ENTSO-E load zone (63 total, see sim_config.py -> LOCATIONS),
REM keyed <city>_<zonecode> (or plain <city> for single-zone countries), e.g.:
REM   oslo_no1 (default) | athens | madrid | rome_it | berlin_50hertz | paris
REM   kristiansand_no2 | stockholm_se3 | london_gb | warsaw | vienna | ...
set LOCATION=budapest

REM Which control strategies to run. Space-separated.
REM   0 = PID   1 = Droop   2 = Centralized   3 = No Control (baseline)
REM Examples:  "1 2 3"   "0"   "0 1 2 3"
set CONTROLS=3

REM Run mode:
REM   0 = single worst-case day  (fast, prints losses to the screen)
REM   1 = full 53-week yearly batch  (slow, writes one week's results per run to sim_results\)
set YEARLY=0

REM PID params: Kp Ki Kd q_decay q_rate hysteresis  (only used by control type 0).
REM   q_decay    .8 = fast bleed, .95 = slow, 1 = no decay
REM   q_rate     max change in Q per step (MVar); 0 = no slew limit
REM   hysteresis p.u. shrink of the integral-decay band
REM Leave this line BLANK to use the tuned values from
REM optimal_pid_gains_worst_case.csv (regenerate it with pid_optimizer_approx.py).
set PID_GAINS=

REM Time-resolution multiplier:
REM   1 = 15-minute steps (96/day, fastest)   3 = 5-minute steps   15 = 1-minute steps
REM Higher = finer resolution but proportionally slower.
set STEP=1

REM Reactive-power supply mode for PID / Droop / Centralized control:
REM   0 = always       - inverters may supply Q at all times, even unplugged
REM   1 = plugged-only - inverters only supply Q while an EV is plugged in
REM                      (a plugged EV that finished charging still supplies Q)
set Q_MODE=1

REM --------------------------- END OF CONFIG ---------------------------------


REM Move to this script's folder so all relative paths resolve correctly.
cd /d "%~dp0"

REM Locate the virtual-environment python (repo-root .venv).
set PYTHON=..\.venv\Scripts\python.exe
if not exist "%PYTHON%" (
    echo Could not find the virtual environment at %PYTHON%
    echo Create it first:  python -m venv ..\.venv  ^&^&  ..\.venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Build the command line from the CONFIG block.
set ARGS=--controls %CONTROLS% --step %STEP% --q-mode %Q_MODE% --location %LOCATION%
if not "%PID_GAINS%"=="" set ARGS=%ARGS% --pid-gains %PID_GAINS%
if "%YEARLY%"=="1" set ARGS=%ARGS% --yearly

REM Re-tune the droop gains at the SAME time resolution AND Q-availability
REM mode the simulation will run at (thesis Ch. 3: gains are only valid for
REM the step they were tuned at).
REM "%PYTHON%" droop_optimizer_approx.py --step %STEP% --q-mode %Q_MODE%
echo.
echo Running: python yearly_tests.py %ARGS%
echo.
"%PYTHON%" yearly_tests.py %ARGS%

echo.
echo Done. (exit code %errorlevel%)
pause
