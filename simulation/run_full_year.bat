@echo off
REM ===========================================================================
REM  run_full_year.bat  -  Full pipeline: re-tune gains, then run all four
REM                        control strategies over the full 53-week year.
REM
REM  Unlike run.bat, this file has no CONFIG block of its own: every value
REM  (LOCATION, time-resolution STEP_MULT, Q_MODE, EV/PV share, seeds, ...) is
REM  left to default from sim_config.py, so editing sim_config.py is the only
REM  thing you need to do before running this. That keeps the gain optimizers
REM  and the yearly runs guaranteed to agree on the same location, resolution
REM  and Q-mode (thesis Ch. 3). All three steps below follow sim_config.LOCATION
REM  and tune/read gains for the current scenario (EV/PV share + placement seed).
REM
REM  Steps (for the active location <key> = sim_config.LOCATION):
REM    1. pid_optimizer_approx.py    -> locations/<key>/gains/pid_<scenario>.csv
REM    2. droop_optimizer_approx.py  -> locations/<key>/gains/droop_<scenario>.csv
REM    3. yearly_tests.py --yearly --controls 0 1 2 3
REM         (PID, Droop, Centralized, No Control - one full year each)
REM ===========================================================================

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

echo.
echo === Step 1/3: Tuning PID gains (pid_optimizer_approx.py) ===
echo.
"%PYTHON%" pid_optimizer_approx.py
if errorlevel 1 (
    echo.
    echo PID gain optimization failed - aborting.
    pause
    exit /b 1
)

echo.
echo === Step 2/3: Tuning droop gains (droop_optimizer_approx.py) ===
echo.
"%PYTHON%" droop_optimizer_approx.py
if errorlevel 1 (
    echo.
    echo Droop gain optimization failed - aborting.
    pause
    exit /b 1
)

echo.
echo === Step 3/3: Full year, all four control strategies (yearly_tests.py) ===
echo.
"%PYTHON%" yearly_tests.py --yearly --controls 0 1 2 3
if errorlevel 1 (
    echo.
    echo Yearly simulation run failed.
    pause
    exit /b 1
)

echo.
echo Done. (exit code %errorlevel%)
pause
