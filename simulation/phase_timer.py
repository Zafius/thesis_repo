"""
phase_timer.py -- per-phase subprocess timing (tune gains -> simulate ->
compare) as a JSON log plus a human-readable summary, and a couple of small
formatting helpers.

Extracted from the PhaseTimer in the now-removed run_berlin_medium_yearly.py
so any script running such a sequence would share one timing format. Its
current consumer is the desktop app: the Run Monitor's timing table reads this
exact JSON shape, and format_hms() is what it displays. See scenario_config.py
for the same note about consumers.
"""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def format_hms(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class PhaseTimer:
    """Runs a subprocess, times it, and persists the timing after every
    phase so a killed run still leaves a record of whatever finished.

    log_path: where the JSON timing log is read from / written to. Reusing
    an existing log_path resumes appending to it (e.g. after a killed run
    is re-launched), so a re-run that skips already-finished phases still
    leaves one continuous record.
    cwd: working directory for every subprocess launched via run(); None
    (default) inherits the caller's current directory.
    """

    def __init__(self, log_path: Path, cwd: Path | None = None):
        self.log_path = log_path
        self.cwd = cwd
        self.phases = []
        if log_path.exists():
            try:
                self.phases = json.loads(log_path.read_text(encoding="utf-8")).get("phases", [])
            except (json.JSONDecodeError, OSError):
                self.phases = []

    def run(self, name, args):
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}\n>>> {' '.join(args)}\n")
        started = datetime.now()
        t0 = time.perf_counter()
        proc = subprocess.run(args, cwd=self.cwd)
        elapsed = time.perf_counter() - t0
        finished = datetime.now()
        self.phases.append(dict(
            phase=name, args=args,
            started=started.isoformat(timespec="seconds"),
            finished=finished.isoformat(timespec="seconds"),
            seconds=round(elapsed, 1),
            exit_code=proc.returncode,
        ))
        self._save()
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"-- {name}: {status} in {format_hms(elapsed)}")
        return proc.returncode

    def skip(self, name, reason):
        """Record a phase that was not run because its output already
        existed, so the timing log/summary still accounts for it."""
        print(f"-- {name}: skipped ({reason})")
        self.phases.append(dict(
            phase=name, args=None,
            started=None, finished=datetime.now().isoformat(timespec="seconds"),
            seconds=0.0, exit_code=None, skipped=True, reason=reason,
        ))
        self._save()

    def _save(self):
        self.log_path.write_text(json.dumps({"phases": self.phases}, indent=2), encoding="utf-8")

    def summary(self):
        lines = ["", "Phase timing summary:", "-" * 52]
        total = 0.0
        for rec in self.phases:
            total += rec["seconds"]
            if rec.get("skipped"):
                status = f"(skipped: {rec['reason']})"
            else:
                status = f"(exit {rec['exit_code']})"
            lines.append(f"  {rec['phase']:<24s} {format_hms(rec['seconds']):>12s}   {status}")
        lines.append("-" * 52)
        lines.append(f"  {'TOTAL':<24s} {format_hms(total):>12s}")
        return "\n".join(lines)
