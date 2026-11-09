@echo off
REM ===========================================================================
REM  run_test_plan.bat  -  Runs (or resumes) the full ../thesis_test_plan.md
REM                        test matrix: double-click, or run from a terminal.
REM
REM  This just launches run_test_plan.py, which does the real work: it prints
REM  progress + an estimated remaining time, asks for confirmation, then runs
REM  every not-yet-done task in the plan's priority order (gain tuning ->
REM  single worst-case day -> full-year robustness -> summaries). Progress is
REM  saved to test_plan_progress.json after every task, so closing this
REM  window and double-clicking it again later resumes where it left off.
REM
REM  Phase 1 (gain tuning) and Phase 2 (single worst-case day) tasks run
REM  several locations/controllers at once (--workers, default 6); Phase 3
REM  (--yearly) always runs one controller at a time -- it already saturates
REM  every core internally, so a second concurrent batch would just contend.
REM
REM  Useful flags (edit the line near the bottom, or run the .py directly):
REM    --status                 show progress + time estimate, run nothing
REM    --dry-run                list every task and its current status
REM    --high-location <key>    set the Phase 3 stress-test location
REM                              (oslo_no1 | athens | berlin_50hertz)
REM    --workers N              max concurrent Phase 1/2 subprocesses (default 6)
REM    --yes                    skip the start confirmation (unattended runs)
REM  See CLI_GUIDE.md and ../thesis_test_plan.md for the plan itself.
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

"%PYTHON%" run_test_plan.py %*

echo.
echo Done for now. (exit code %errorlevel%)
echo Re-run this file any time to resume from where it stopped.
pause
