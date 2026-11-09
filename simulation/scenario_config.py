"""
scenario_config.py -- EV/PV penetration-scenario helpers, used by the desktop
app's job runner (desktop_app/backend/).

sim_config.py's EV_CONTRIBUTION/PV_CONTRIBUTION are plain module constants
with no CLI override -- every optimizer/yearly_tests.py subprocess reads
them at import time. Anything that needs to run more than one scenario
switches them by rewriting these two lines in sim_config.py in place before
each subprocess call.

CURRENT CONSUMERS: the desktop app only. This was extracted (with
phase_timer.py) from the duplicated logic in run_test_plan.py and the
now-removed run_berlin_medium_yearly.py so every caller would share one
tested implementation -- but run_test_plan.py has since grown its own
scenario_tag()/set_scenario() again, so the near-identical copies this was
meant to prevent do exist. They agree today (there is an equivalence check in
the desktop app's test notes); if sim_config.py's format ever changes, BOTH
have to be updated.

Callers that mutate the scenario are responsible for restoring it
afterwards (typically via a try/finally around read_scenario() ->
set_scenario(...) -> ... -> set_scenario(*original)) -- this module does
not track "the" original value itself, since some callers intentionally
restore to a fixed default rather than whatever was there before.
"""
import re
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
SIM_CONFIG_PATH = SIM_DIR / "sim_config.py"
LOCATIONS_DIR = SIM_DIR / "locations"

NETWORK_SEED = 42  # must match sim_config.NETWORK_SEED to find the right gains file


def read_scenario():
    """Current (EV_CONTRIBUTION, PV_CONTRIBUTION) as written in sim_config.py."""
    text = SIM_CONFIG_PATH.read_text(encoding="utf-8")
    ev = float(re.search(r"(?m)^EV_CONTRIBUTION = ([0-9.]+)", text).group(1))
    pv = float(re.search(r"(?m)^PV_CONTRIBUTION = ([0-9.]+)", text).group(1))
    return ev, pv


def read_seed():
    """Current NETWORK_SEED as written in sim_config.py.

    Read from the file rather than from the NETWORK_SEED constant below,
    which is only a mirror for naming. Nothing passes a seed on the command
    line -- every script reads it at import -- so the file is the truth.
    """
    text = SIM_CONFIG_PATH.read_text(encoding="utf-8")
    match = re.search(r"(?m)^NETWORK_SEED = (\d+)", text)
    return int(match.group(1)) if match else NETWORK_SEED


def read_constant(name, default=None):
    """A plain numeric constant's value as written in sim_config.py, or default.

    Same reason read_scenario() exists rather than importing sim_config:
    callers here are GUI-side and must not pull numpy (and transitively
    pandapower) into their process just to show what a default currently is.
    Only matches a bare `NAME = <number>` at the start of a line, so a
    commented-out or indented occurrence is not mistaken for the real one.
    """
    text = SIM_CONFIG_PATH.read_text(encoding="utf-8")
    match = re.search(rf"(?m)^{re.escape(name)} = ([0-9.]+)", text)
    if match is None:
        return default
    raw = match.group(1)
    return float(raw) if "." in raw else int(raw)


def read_q_mode():
    """sim_config.Q_MODE: 0 = supply Q always, 1 = only while plugged in."""
    return read_constant("Q_MODE", 1)


def read_max_curtail_frac():
    """sim_config.MAX_P_CURTAIL_FRAC: max share of scheduled P an inverter may
    curtail to make current headroom for Q."""
    return read_constant("MAX_P_CURTAIL_FRAC", 0.05)


def set_scenario(ev, pv):
    """Rewrite sim_config.py's EV_CONTRIBUTION/PV_CONTRIBUTION lines in place.

    Only these two numeric literals are touched; every other line (including
    any other uncommitted edits already in the file) is left untouched.
    """
    text = SIM_CONFIG_PATH.read_text(encoding="utf-8")
    new_text, n_ev = re.subn(r"(?m)^EV_CONTRIBUTION = [0-9.]+", f"EV_CONTRIBUTION = {ev}", text)
    new_text, n_pv = re.subn(r"(?m)^PV_CONTRIBUTION = [0-9.]+", f"PV_CONTRIBUTION = {pv}", new_text)
    if n_ev != 1 or n_pv != 1:
        raise RuntimeError(
            f"Expected exactly one EV_CONTRIBUTION and one PV_CONTRIBUTION line in "
            f"{SIM_CONFIG_PATH}, found {n_ev}/{n_pv}. Refusing to edit -- the file may "
            f"have been restructured; update set_scenario()'s regex to match.")
    if new_text != text:
        SIM_CONFIG_PATH.write_text(new_text, encoding="utf-8")


def set_seed(seed):
    """Rewrite sim_config.py's NETWORK_SEED line in place.

    Same anchored-prefix substitution as set_scenario(), and for the same
    reason: the match ends before the trailing comment on that line, so only
    the number changes. Refuses rather than guessing if the file no longer
    has exactly one such line.

    NETWORK_SEED feeds both the EV/PV placement and scenario_tag(), so
    changing it changes which gains file a run looks for -- callers must pass
    the seed explicitly to gains_path()/gains_exist() rather than relying on
    the module default, which is a naming mirror and does not follow this.
    """
    text = SIM_CONFIG_PATH.read_text(encoding="utf-8")
    new_text, n = re.subn(r"(?m)^NETWORK_SEED = \d+", f"NETWORK_SEED = {int(seed)}", text)
    if n != 1:
        raise RuntimeError(
            f"Expected exactly one NETWORK_SEED line in {SIM_CONFIG_PATH}, found "
            f"{n}. Refusing to edit -- the file may have been restructured; "
            f"update set_seed()'s regex to match.")
    if new_text != text:
        SIM_CONFIG_PATH.write_text(new_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Crash safety for the in-place scenario rewrite
# ---------------------------------------------------------------------------
# set_scenario() mutates a file every other tool reads. Callers restore it in a
# try/finally, which covers an exception or a cancel -- but not the process
# dying: a crash, a power loss, or the user closing the app mid-run leaves
# sim_config.py holding someone else's scenario, and the NEXT simulation
# silently runs with it. There is no in-process way to defend against that, so
# the intended value is written down before the mutation and removed after the
# restore. Whatever finds this file next knows a run died and what to put back.
#
# Currently written by the desktop app's job runner. run_test_plan.py has its
# own copy of set_scenario() and does not use these yet; adopting them would
# give the CLI the same protection.
PENDING_RESTORE_PATH = SIM_DIR / ".scenario_restore.json"


def record_pending_restore(ev, pv, seed=None):
    """Note the config that must be put back if this process does not finish.

    `seed` is written only when given, so a note this function produces stays
    readable by an older build's pending_restore() -- and an older note, with
    no seed key, stays readable here. Both directions matter: a note is
    written by whatever died and read by whatever starts next, and those are
    not always the same build.
    """
    import json
    state = {"ev": ev, "pv": pv, "written": _now()}
    if seed is not None:
        state["seed"] = int(seed)
    PENDING_RESTORE_PATH.write_text(json.dumps(state), encoding="utf-8")


def clear_pending_restore():
    """Called once the scenario really has been restored."""
    PENDING_RESTORE_PATH.unlink(missing_ok=True)


def pending_restore():
    """(ev, pv) a previous run left unrestored, or None.

    None also covers an unreadable or malformed file: a corrupt note is not
    worth refusing to start over, and the worst case is the same as having no
    note at all.
    """
    state = pending_restore_state()
    return None if state is None else (state["ev"], state["pv"])


def pending_restore_state():
    """Everything a previous run left unrestored, as a dict, or None.

    {"ev": float, "pv": float, "seed": int | None}. Kept separate from
    pending_restore() so that function's two-tuple contract -- which
    main.py's startup recovery and the safety tests both rely on -- does not
    change shape underneath them.
    """
    import json
    try:
        data = json.loads(PENDING_RESTORE_PATH.read_text(encoding="utf-8"))
        seed = data.get("seed")
        return {"ev": float(data["ev"]), "pv": float(data["pv"]),
                "seed": None if seed is None else int(seed)}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _now():
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def scenario_tag(ev, pv, seed=NETWORK_SEED):
    """Matches sim_config.scenario_tag()'s naming without importing that
    module -- sim_config.py may be mid-rewrite around a scenario switch
    when this is called, and importing it here would risk caching a stale
    value for the rest of the process's lifetime."""
    return f"ev{round(ev * 100)}_pv{round(pv * 100)}_seed{round(seed)}"


def gains_path(kind, location, ev, pv, seed=NETWORK_SEED):
    """Path to a tuned-gains CSV (kind: 'pid' or 'droop') if it exists on
    disk, else None. Resolved by globbing locations/*/<location>/gains/
    rather than looking up the location's country folder, so this needs no
    import of sim_config."""
    tag = scenario_tag(ev, pv, seed)
    matches = list(LOCATIONS_DIR.glob(f"*/{location}/gains/{kind}_{tag}.csv"))
    return matches[0] if matches else None


def gains_exist(kind, location, ev, pv, seed=NETWORK_SEED):
    return gains_path(kind, location, ev, pv, seed) is not None
