"""
compare_control_strategies.py -- aggregate and compare the four voltage-control
strategies (PID, Droop, Centralized, No Control) over a completed yearly batch.

Reads the directory tree written by yearly_tests.py / network_setup.py:

    <root>/                                   (a sim_results_combined/ folder)
        CentralizedControl/<scenario>/week_0 .. week_52/
        Droopcontrol/<scenario>/week_0 .. week_52/
        NoControl/<scenario>/week_0 .. week_52/
        PIDcontrol/<scenario>/week_0 .. week_52/

Each week_N/ holds:
    simulation_results_losses.csv                    -- Total_Losses_kW, time-of-day index
    global_voltages/simulation_results_phase_{A,B,C}.csv -- vm_{ph}_pu_min/max per step
    inverter_profiles/global_q_phase_{A,B,C}.csv      -- one column per active inverter, kVAr
    voltage_quality.csv                               -- viol_count_{a,b,c}, viol_mag_{a,b,c},
                                                          sq_dev_{a,b,c}, vuf_max, vuf_mean,
                                                          n_lv_buses per step (all LV buses)
    transformer_loading.csv                           -- loading_percent (+ per-phase), s_kva,
                                                          p_kw, q_kvar, sn_kva at the MV/LV
                                                          substation transformer per step

DATA CAVEAT: voltage_quality.csv is the exact all-bus figure and is used
whenever present. Older result trees generated before network_setup.py started
persisting it only have the per-phase MIN/MAX voltage files, so
voltage_violation_percentage() falls back to a (step x phase) proxy for those
-- it flags a step-phase as a violation if the worst bus on that phase
breached the band, which is not a true per-bus percentage -- and the
excursion/RMS-deviation/VUF metrics come back NaN entirely (there is no proxy
for those). The summary table records which method (exact/proxy) was used
for the violation percentage per strategy.

Usage:
    python compare_control_strategies.py --root sim_results/2026-07-14_02.18/sim_results_combined
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sim_config as cfg
from compute_curtailment import RESULTS_BASIS_NOTE  # shared 53-day time-basis note

# ---------------------------------------------------------------------------
# Fixed categorical palette (dataviz skill, references/palette.md, slots 1-4).
# Order is fixed and never re-derived from the data, so a strategy always
# gets the same color across every figure. Chosen for CVD-safe adjacent
# contrast, not aesthetics -- do not reorder without re-validating.
# ---------------------------------------------------------------------------
STRATEGY_COLORS = {
    "NoControl": "#2a78d6",         # slot 1, blue
    "PIDcontrol": "#1baf7a",        # slot 2, aqua
    "Droopcontrol": "#eda100",      # slot 3, yellow
    "CentralizedControl": "#008300",  # slot 4, green
}
STRATEGY_LABELS = {
    "NoControl": "No Control",
    "PIDcontrol": "PID",
    "Droopcontrol": "Droop",
    "CentralizedControl": "Centralized",
}
STRATEGY_ORDER = ["NoControl", "PIDcontrol", "Droopcontrol", "CentralizedControl"]

V_LOW, V_HIGH = 0.97, 1.05  # thesis Ch. 1/3 voltage safety band, p.u.

WEEK_DIR_RE = re.compile(r"week_(\d+)$")


# ---------------------------------------------------------------------------
# 1. Directory discovery
# ---------------------------------------------------------------------------
def discover_scenario_dir(control_dir: Path, scenario: str | None) -> Path:
    """Resolve <control_dir>/<scenario>, auto-picking the only child if
    scenario is None. Raises with the available choices if ambiguous."""
    if scenario is not None:
        return control_dir / scenario
    children = sorted(p for p in control_dir.iterdir() if p.is_dir())
    if len(children) != 1:
        names = ", ".join(p.name for p in children)
        raise ValueError(
            f"{control_dir} has {len(children)} scenario folders ({names}); "
            "pass --scenario to disambiguate.")
    return children[0]


def discover_week_dirs(scenario_dir: Path) -> list[tuple[int, Path]]:
    """(week_index, path) pairs sorted numerically (week_0, week_1, ... week_52,
    not the lexicographic week_0, week_1, week_10, ...)."""
    weeks = []
    for child in scenario_dir.iterdir():
        m = WEEK_DIR_RE.match(child.name)
        if child.is_dir() and m:
            weeks.append((int(m.group(1)), child))
    return sorted(weeks, key=lambda pair: pair[0])


# ---------------------------------------------------------------------------
# 2. Per-week loaders
#
# Every week folder only carries a time-of-day index ("00:00:00", ...), so
# concatenating weeks naively would produce duplicate, non-monotonic
# timestamps. Each loader stamps its rows onto a synthetic full-year
# timeline instead (base_date + 7 * week_index days), which is the "common
# time index" the weeks are aligned onto before concatenation.
# ---------------------------------------------------------------------------
def _week_datetime_index(time_of_day_index: pd.Index, week_index: int,
                         base_date: pd.Timestamp) -> pd.DatetimeIndex:
    week_start = base_date + pd.Timedelta(weeks=week_index)
    deltas = pd.to_timedelta(time_of_day_index.astype(str))
    return pd.DatetimeIndex(week_start + deltas)


def load_week_losses(week_dir: Path, week_index: int,
                     base_date: pd.Timestamp) -> pd.Series:
    df = pd.read_csv(week_dir / "simulation_results_losses.csv", index_col=0)
    series = df["Total_Losses_kW"]
    series.index = _week_datetime_index(df.index, week_index, base_date)
    return series


def load_week_voltage_bounds(week_dir: Path, week_index: int,
                             base_date: pd.Timestamp) -> pd.DataFrame:
    voltages_dir = week_dir / "global_voltages"
    phase_frames = []
    for phase in ("A", "B", "C"):
        path = voltages_dir / f"simulation_results_phase_{phase}.csv"
        if not path.exists():
            continue
        phase_frames.append(pd.read_csv(path, index_col=0))
    combined = pd.concat(phase_frames, axis=1)
    combined.index = _week_datetime_index(combined.index, week_index, base_date)
    return combined


def load_week_reactive_power(week_dir: Path, week_index: int,
                             base_date: pd.Timestamp) -> pd.DataFrame:
    """One column per (inverter, phase); columns are suffixed with the phase
    so an inverter id can never collide across the three per-phase files."""
    inverter_dir = week_dir / "inverter_profiles"
    phase_frames = []
    for phase in ("A", "B", "C"):
        path = inverter_dir / f"global_q_phase_{phase}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        df = df.add_suffix(f"_phase_{phase}")
        phase_frames.append(df)
    if not phase_frames:
        return pd.DataFrame()
    combined = pd.concat(phase_frames, axis=1)
    combined.index = _week_datetime_index(combined.index, week_index, base_date)
    return combined


def load_week_voltage_quality(week_dir: Path, week_index: int,
                              base_date: pd.Timestamp) -> pd.DataFrame | None:
    """voltage_quality.csv: per-step, all-LV-bus violation counts/magnitudes,
    squared deviations from 1.0 p.u., and VUF (viol_count_{a,b,c},
    viol_mag_{a,b,c}, sq_dev_{a,b,c}, vuf_max, vuf_mean, n_lv_buses). Only
    present in runs produced after network_setup.py started persisting it --
    returns None for older result trees so callers can fall back to the
    coarser min/max-based proxy (violation % only; excursion/RMS-deviation/VUF
    have no proxy and come back NaN)."""
    path = week_dir / "voltage_quality.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = _week_datetime_index(df.index, week_index, base_date)
    return df


def load_week_timing(week_dir: Path) -> dict | None:
    """timing.csv: single-row (total_seconds, n_steps, seconds_per_step) wall-clock
    timing of one day's control+power-flow loop (network_setup.py). Only present
    in runs produced after that instrumentation was added -- returns None for
    older result trees, so callers should treat missing timing as "unknown",
    not zero."""
    path = week_dir / "timing.csv"
    if not path.exists():
        return None
    row = pd.read_csv(path).iloc[0]
    return {"total_seconds": float(row["total_seconds"]),
           "n_steps": int(row["n_steps"]),
           "seconds_per_step": float(row["seconds_per_step"])}


def load_week_transformer_loading(week_dir: Path, week_index: int,
                                  base_date: pd.Timestamp) -> pd.DataFrame | None:
    """transformer_loading.csv: loading_percent (+ per-phase), s_kva, p_kw,
    q_kvar, sn_kva at the MV/LV substation transformer per step. Only present
    in runs produced after network_setup.py started persisting it -- returns
    None for older result trees, mirroring load_week_voltage_quality."""
    path = week_dir / "transformer_loading.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = _week_datetime_index(df.index, week_index, base_date)
    return df


# ---------------------------------------------------------------------------
# 3. Aggregation across the whole scenario (all weeks) for one strategy
# ---------------------------------------------------------------------------
def aggregate_control_strategy(control_dir: Path, scenario: str | None,
                               base_date: pd.Timestamp) -> dict:
    """Concatenate every week's losses / voltage-bounds / reactive-power /
    voltage-quality tables into one continuous, time-ordered series/frame per
    signal. Returns a dict with keys 'losses', 'voltages', 'reactive_power',
    'voltage_quality' (None if any week is missing voltage_quality.csv), plus
    'week_of' -- a parallel Series mapping each timestamp back to its
    original week_N index, so a specific week can be sliced out later
    without depending on the synthetic base_date.
    """
    scenario_dir = discover_scenario_dir(control_dir, scenario)
    week_dirs = discover_week_dirs(scenario_dir)
    if not week_dirs:
        raise ValueError(f"No week_* folders found under {scenario_dir}")

    losses_parts, voltage_parts, reactive_parts, week_of_parts = [], [], [], []
    quality_parts = []
    trafo_parts = []
    timing_parts = []
    for week_index, week_dir in week_dirs:
        losses = load_week_losses(week_dir, week_index, base_date)
        losses_parts.append(losses)
        voltage_parts.append(load_week_voltage_bounds(week_dir, week_index, base_date))
        reactive_parts.append(load_week_reactive_power(week_dir, week_index, base_date))
        week_of_parts.append(pd.Series(week_index, index=losses.index))

        week_quality = load_week_voltage_quality(week_dir, week_index, base_date)
        if quality_parts is not None:
            if week_quality is None:
                quality_parts = None  # one gap invalidates the exact metrics for this strategy
            else:
                quality_parts.append(week_quality)

        week_trafo = load_week_transformer_loading(week_dir, week_index, base_date)
        if trafo_parts is not None:
            if week_trafo is None:
                trafo_parts = None  # one gap invalidates the transformer metrics for this strategy
            else:
                trafo_parts.append(week_trafo)

        week_timing = load_week_timing(week_dir)
        if week_timing is not None:
            timing_parts.append(week_timing)  # partial coverage is fine, unlike voltage_quality

    return {
        "losses": pd.concat(losses_parts).sort_index(),
        "voltages": pd.concat(voltage_parts).sort_index(),
        "reactive_power": pd.concat(reactive_parts).sort_index(),
        "voltage_quality": pd.concat(quality_parts).sort_index() if quality_parts else None,
        "transformer_loading": pd.concat(trafo_parts).sort_index() if trafo_parts else None,
        "timing": pd.DataFrame(timing_parts) if timing_parts else None,
        "week_of": pd.concat(week_of_parts).sort_index(),
    }


def load_all_strategies(root: Path, scenario: str | None,
                        base_date: pd.Timestamp) -> dict[str, dict]:
    """Aggregate every strategy folder present under root. Missing folders
    are skipped with a warning rather than failing the whole comparison."""
    strategies = {}
    for name in STRATEGY_ORDER:
        control_dir = root / name
        if not control_dir.is_dir():
            print(f"[warn] {control_dir} not found, skipping {name}")
            continue
        strategies[name] = aggregate_control_strategy(control_dir, scenario, base_date)
    return strategies


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def _dt_hours(index: pd.DatetimeIndex) -> float:
    """Median step size of a time index, in hours (robust to the last row of
    each week landing on a slightly different spacing)."""
    diffs = index.to_series().diff().dropna()
    return diffs.median().total_seconds() / 3600.0


def total_active_losses_kwh(losses: pd.Series) -> float:
    """Integrate instantaneous line-loss power (kW) into energy (kWh) via a
    rectangular rule: sum(kW) * step_hours. Matches the kWh convention
    yearly_tests.py/network_setup.py already report their per-day figures in."""
    return float(losses.sum() * _dt_hours(losses.index))


def _voltage_violation_percentage_exact(quality: pd.DataFrame) -> float:
    """True percentage of (time step, phase, LV bus) triples outside
    [V_LOW, V_HIGH], from the exact all-bus viol_count_{a,b,c} columns
    persisted by network_setup.py (voltage_quality.csv)."""
    count_cols = [c for c in ("viol_count_a", "viol_count_b", "viol_count_c")
                 if c in quality.columns]
    if not count_cols or "n_lv_buses" not in quality.columns:
        return float("nan")
    n_lv_buses = float(quality["n_lv_buses"].iloc[0])
    total_violations = float(quality[count_cols].to_numpy().sum())
    total_measurements = len(quality) * len(count_cols) * n_lv_buses
    return float(100.0 * total_violations / total_measurements)


def _voltage_violation_percentage_proxy(voltages: pd.DataFrame,
                                        v_low: float = V_LOW, v_high: float = V_HIGH) -> float:
    """Percentage of (time step, phase) pairs where the worst bus on that
    phase breached [v_low, v_high].

    CAVEAT: only min/max per phase are on disk here, so this is a proxy for
    -- not equal to -- the true per-bus violation rate: one flagged step
    could mean a single bus barely out of band or dozens of buses deep in
    violation. Used only when voltage_violations.csv isn't available (older
    result trees); see _voltage_violation_percentage_exact for the real
    all-bus figure.
    """
    flags = []
    for phase in ("a", "b", "c"):
        min_col, max_col = f"vm_{phase}_pu_min", f"vm_{phase}_pu_max"
        if min_col not in voltages.columns or max_col not in voltages.columns:
            continue
        flags.append((voltages[min_col] < v_low) | (voltages[max_col] > v_high))
    if not flags:
        return float("nan")
    flagged = pd.concat(flags)
    return float(100.0 * flagged.sum() / flagged.size)


def voltage_violation_percentage(data: dict,
                                 v_low: float = V_LOW, v_high: float = V_HIGH) -> tuple[float, str]:
    """Percentage of voltage measurements outside [v_low, v_high], plus which
    method produced it ('exact' = true all-LV-bus count from
    voltage_quality.csv, 'proxy' = inferred from the per-phase min/max
    only). Exact is used whenever available."""
    if data["voltage_quality"] is not None:
        return _voltage_violation_percentage_exact(data["voltage_quality"]), "exact"
    return _voltage_violation_percentage_proxy(data["voltages"], v_low, v_high), "proxy"


def total_excursion_puh(data: dict) -> float:
    """Total voltage excursion beyond the band, in p.u.-hours: sum of
    viol_mag_{a,b,c} (p.u. excursion summed over every violating LV bus at
    each step) integrated over time. A severity-weighted complement to the
    violation percentage -- two strategies can violate equally often but
    differ hugely in how far outside the band they sit. NaN if
    voltage_quality.csv isn't available (no proxy for this one)."""
    quality = data["voltage_quality"]
    if quality is None:
        return float("nan")
    mag_cols = [c for c in ("viol_mag_a", "viol_mag_b", "viol_mag_c") if c in quality.columns]
    if not mag_cols:
        return float("nan")
    return float(quality[mag_cols].sum(axis=1).sum() * _dt_hours(quality.index))


def rms_voltage_deviation_pu(data: dict) -> float:
    """RMS deviation of every LV bus's voltage from 1.0 p.u., over every
    phase and step of the scenario: sqrt(mean((V - 1)^2)). A continuous
    quality measure that doesn't require crossing the safety band to
    register, unlike the violation percentage. NaN if voltage_quality.csv
    isn't available."""
    quality = data["voltage_quality"]
    if quality is None:
        return float("nan")
    sq_cols = [c for c in ("sq_dev_a", "sq_dev_b", "sq_dev_c") if c in quality.columns]
    if not sq_cols or "n_lv_buses" not in quality.columns:
        return float("nan")
    n_lv_buses = float(quality["n_lv_buses"].iloc[0])
    total_sq_dev = float(quality[sq_cols].to_numpy().sum())
    total_measurements = len(quality) * len(sq_cols) * n_lv_buses
    return float(np.sqrt(total_sq_dev / total_measurements))


def vuf_summary(data: dict) -> tuple[float, float]:
    """(worst-case, mean) EN 50160 voltage unbalance factor [%] over the
    scenario: the worst is the max of the per-step max-over-buses, the mean
    is the mean of the per-step mean-over-buses. (NaN, NaN) if
    voltage_quality.csv isn't available."""
    quality = data["voltage_quality"]
    if quality is None or "vuf_max" not in quality.columns or "vuf_mean" not in quality.columns:
        return float("nan"), float("nan")
    return float(quality["vuf_max"].max()), float(quality["vuf_mean"].mean())


def total_reactive_effort_kvarh_samples(reactive_power: pd.DataFrame) -> float:
    """Sum of |Q| (kVAr) across every inverter, phase and time step -- a
    literal effort tally, not a time integral (multiply by _dt_hours() first
    if an energy-like kvarh figure is wanted instead)."""
    if reactive_power.empty:
        return 0.0
    return float(reactive_power.abs().to_numpy().sum())


def per_inverter_reactive_effort(reactive_power: pd.DataFrame) -> pd.Series:
    """Total |Q| effort per inverter column, summed over the whole scenario.
    One value per (inverter, phase) column -- the distribution the boxplot
    visualizes, since the raw per-step samples are mostly zero (Q is only
    nonzero while a controller is actively correcting voltage) and would not
    show a meaningful spread."""
    if reactive_power.empty:
        return pd.Series(dtype=float)
    return reactive_power.abs().sum(axis=0)


def peak_instantaneous_q_kvar(reactive_power: pd.DataFrame) -> float:
    """Largest single (inverter, step) |Q| in the scenario -- a sizing/stress
    figure distinct from the cumulative effort totals above."""
    if reactive_power.empty:
        return 0.0
    return float(reactive_power.abs().to_numpy().max())


def total_q_variation_kvar(data: dict) -> float:
    """Sum of |delta Q| between consecutive steps (every inverter/phase
    column), a "smoothness" measure distinct from total effort: two
    strategies can move the same total |Q| while one chatters step-to-step
    and the other settles smoothly, and this is what tells them apart.

    Diffs are computed within each week_N separately and then summed, since
    consecutive weeks are not really consecutive in time (the synthetic
    full-year timeline in aggregate_control_strategy only keeps them from
    colliding when concatenated) -- a naive diff across the whole
    concatenated series would also count one spurious jump per week
    boundary.
    """
    reactive_power = data["reactive_power"]
    if reactive_power.empty:
        return 0.0
    total = 0.0
    for _, week_df in reactive_power.groupby(data["week_of"]):
        total += week_df.diff().abs().sum().sum()  # .sum() skips the NaN diff() leaves on row 1
    return float(total)


def peak_instantaneous_loss_kw(losses: pd.Series) -> float:
    """Single worst-step network loss -- a thermal/rating figure, distinct
    from the total energy figure."""
    return float(losses.max())


def mean_seconds_per_step(data: dict) -> float:
    """Mean wall-clock seconds per simulated step (thesis Ch. 5 computational-
    cost metric), averaged over whichever days persisted timing.csv
    (network_setup.py). Single-process wall time, NOT normalized for the
    across-week parallelism yearly_tests.py/run_full_year.bat use -- a real
    per-step control+power-flow cost, not a wall-clock-to-solution estimate
    for the whole batch. NaN if no day in the scenario has timing data."""
    timing = data["timing"]
    if timing is None or timing.empty:
        return float("nan")
    return float(timing["seconds_per_step"].mean())


def transformer_loading_summary(data: dict) -> dict:
    """Peak/mean loading_percent and peak apparent power S at the MV/LV
    substation transformer, plus the percentage of steps where loading
    exceeds its nameplate rating (100%) -- a congestion indicator distinct
    from the LV-bus voltage-quality metrics above: reactive power that keeps
    every bus voltage in-band can still push more apparent power through the
    transformer than it is rated for. NaN if transformer_loading.csv isn't
    available (older result trees; see load_week_transformer_loading)."""
    trafo = data["transformer_loading"]
    if trafo is None or trafo.empty:
        return {"max_pct": float("nan"), "mean_pct": float("nan"),
                "pct_time_over_100pct": float("nan"), "peak_s_kva": float("nan")}
    over_100 = float(100.0 * (trafo["loading_percent"] > 100.0).sum() / len(trafo))
    return {
        "max_pct": float(trafo["loading_percent"].max()),
        "mean_pct": float(trafo["loading_percent"].mean()),
        "pct_time_over_100pct": over_100,
        "peak_s_kva": float(trafo["s_kva"].max()) if "s_kva" in trafo.columns else float("nan"),
    }


def gini_coefficient(values: np.ndarray) -> float:
    """Standard Gini coefficient (0 = perfectly even, 1 = maximally
    concentrated in one value) via the sorted mean-absolute-difference
    formula. Values are assumed non-negative (reactive-power effort
    magnitudes here). Returns 0.0 -- rather than the 0/0 NaN the raw formula
    would produce -- for an empty array or when every value is zero (e.g.
    No Control, which never sets Q: "perfectly even" because there is
    nothing to be uneven about)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0 or np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


def reactive_power_fairness(reactive_power: pd.DataFrame) -> dict:
    """How evenly the total reactive-power effort (summed |Q| over the whole
    scenario, per inverter -- see per_inverter_reactive_effort) is
    distributed across the network's chargers: a Gini index (0 = every
    inverter contributes equally, 1 = one inverter does all the work) and
    the coefficient of variation (std/mean) as a second, unit-bearing view
    of the same spread. Most useful for the centralized controller, which
    can in principle concentrate reactive-power effort on whichever
    inverters its sensitivity matrix judges most effective -- an uneven
    controller can look efficient in the aggregate totals above while
    over-stressing a handful of chargers (see sign_flip_frequency)."""
    effort = per_inverter_reactive_effort(reactive_power)
    if effort.empty:
        return {"gini": 0.0, "coefficient_of_variation": float("nan")}
    values = effort.to_numpy()
    mean = float(values.mean())
    cv = float(values.std() / mean) if mean > 0 else float("nan")
    return {"gini": gini_coefficient(values), "coefficient_of_variation": cv}


def estimated_additional_inverter_losses_kwh(data: dict) -> float:
    """Estimated extra energy consumed by the chargers' inverters
    specifically because they supply reactive power, from the I^2R
    conduction-loss model in sim_config.py (Loss(S) = k*S^2, so the loss
    attributable to Q alone is k*Q^2 regardless of P -- see
    sim_config.inverter_conduction_loss_coeff_kw_per_kvar2 for the
    derivation and the rated-loss-percentage assumption behind k). Summed
    over every inverter and time step, integrated to kWh the same way
    total_active_losses_kwh integrates the line losses. 0.0 if no reactive
    power was ever supplied (e.g. No Control)."""
    reactive_power = data["reactive_power"]
    if reactive_power.empty:
        return 0.0
    k = cfg.inverter_conduction_loss_coeff_kw_per_kvar2()
    loss_kw_sum = float((k * (reactive_power ** 2)).to_numpy().sum())
    return float(loss_kw_sum * _dt_hours(reactive_power.index))


def _sign_flip_count(q: np.ndarray, deadband: float) -> int:
    """Number of times one inverter's Q crosses from capacitive (> deadband)
    to inductive (< -deadband) or back. Samples inside the deadband are
    treated as "not delivering Q" (carried forward from the last real sign)
    rather than as a sign of their own, so controller/measurement dither
    around zero isn't counted as repeated switching; any leading run before
    the first non-deadband sample is dropped (nothing to flip from yet)."""
    sign = np.where(q > deadband, 1.0, np.where(q < -deadband, -1.0, np.nan))
    held = pd.Series(sign).ffill().dropna().to_numpy()
    if held.size < 2:
        return 0
    return int(np.sum(np.diff(held) != 0))


def sign_flip_frequency(data: dict, deadband_kvar: float = cfg.SIGN_FLIP_DEADBAND_KVAR) -> dict:
    """Per-inverter count of switches between capacitive and inductive
    operation -- a proxy for DC-link capacitor stress, distinct from the
    total |Q| effort or its step-to-step variation (total_q_variation_kvar
    counts every change, including two steps that are both, say,
    capacitive). Counted within each week_N separately and summed (like
    total_q_variation_kvar), since consecutive weeks are not really
    consecutive in time. 'mean_per_inverter_per_day' normalizes by the
    number of weeks, since each week_N folder is one representative
    simulated day."""
    reactive_power = data["reactive_power"]
    if reactive_power.empty:
        return {"per_inverter": pd.Series(dtype=int), "total": 0,
               "mean_per_inverter": 0.0, "mean_per_inverter_per_day": 0.0}

    n_weeks = int(data["week_of"].nunique())
    counts = pd.Series(0, index=reactive_power.columns, dtype=int)
    for _, week_df in reactive_power.groupby(data["week_of"]):
        for col in week_df.columns:
            counts[col] += _sign_flip_count(week_df[col].to_numpy(), deadband_kvar)

    total = int(counts.sum())
    mean_per_inverter = float(counts.mean()) if len(counts) else 0.0
    mean_per_inverter_per_day = mean_per_inverter / n_weeks if n_weeks else 0.0
    return {"per_inverter": counts, "total": total,
           "mean_per_inverter": mean_per_inverter,
           "mean_per_inverter_per_day": mean_per_inverter_per_day}


def summarize_strategy(name: str, data: dict) -> dict:
    violation_pct, violation_method = voltage_violation_percentage(data)
    vuf_max_pct, vuf_mean_pct = vuf_summary(data)
    trafo = transformer_loading_summary(data)
    fairness = reactive_power_fairness(data["reactive_power"])
    flips = sign_flip_frequency(data)
    return {
        "strategy": STRATEGY_LABELS[name],
        "total_active_losses_kwh": total_active_losses_kwh(data["losses"]),
        "peak_instantaneous_loss_kw": peak_instantaneous_loss_kw(data["losses"]),
        "voltage_violation_pct": violation_pct,
        "voltage_violation_pct_method": violation_method,
        "total_excursion_puh": total_excursion_puh(data),
        "rms_voltage_deviation_pu": rms_voltage_deviation_pu(data),
        "vuf_max_pct": vuf_max_pct,
        "vuf_mean_pct": vuf_mean_pct,
        "total_reactive_effort_kvar": total_reactive_effort_kvarh_samples(data["reactive_power"]),
        "total_q_variation_kvar": total_q_variation_kvar(data),
        "peak_instantaneous_q_kvar": peak_instantaneous_q_kvar(data["reactive_power"]),
        "transformer_loading_max_pct": trafo["max_pct"],
        "transformer_loading_mean_pct": trafo["mean_pct"],
        "transformer_pct_time_over_100pct": trafo["pct_time_over_100pct"],
        "transformer_peak_s_kva": trafo["peak_s_kva"],
        "reactive_fairness_gini": fairness["gini"],
        "reactive_fairness_cv": fairness["coefficient_of_variation"],
        "estimated_additional_inverter_losses_kwh": estimated_additional_inverter_losses_kwh(data),
        "sign_flips_total": flips["total"],
        "sign_flips_mean_per_inverter_per_day": flips["mean_per_inverter_per_day"],
        "mean_seconds_per_step": mean_seconds_per_step(data),
        "n_weeks": int(data["week_of"].nunique()),
    }


def build_summary_table(strategies: dict[str, dict]) -> pd.DataFrame:
    rows = [summarize_strategy(name, data) for name, data in strategies.items()]
    return pd.DataFrame(rows).set_index("strategy")


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
def _extreme_voltage(voltages: pd.DataFrame) -> pd.Series:
    """Per step, whichever of (min across phases, max across phases) is
    furthest from 1.0 p.u. -- a single trace that captures both under- and
    over-voltage excursions instead of six overlapping min/max lines."""
    mins = voltages[[c for c in voltages.columns if c.endswith("_min")]].min(axis=1)
    maxs = voltages[[c for c in voltages.columns if c.endswith("_max")]].max(axis=1)
    under_is_worse = (1.0 - mins) >= (maxs - 1.0)
    return mins.where(under_is_worse, maxs)


def select_worst_week(strategies: dict[str, dict], reference: str = "NoControl") -> int:
    """Week index with the largest voltage excursion from 1.0 p.u. in the
    reference strategy (No Control, the unmitigated baseline) -- used as the
    default week for the line-plot comparison when --week is not given."""
    ref = strategies.get(reference) or next(iter(strategies.values()))
    extreme = _extreme_voltage(ref["voltages"])
    excursion = (extreme - 1.0).abs()
    worst_ts = excursion.idxmax()
    return int(ref["week_of"].loc[worst_ts])


def _finish(fig, output_path: Path | None):
    """Shared tail of every plot_* function: write the PNG and release the
    figure when a path was given (the CLI's behavior), or hand the live figure
    back to the caller when it was not.

    The `output_path=None` mode exists for the desktop app, which embeds these
    same figures in a FigureCanvasQTAgg instead of writing a file
    (../desktop_app_plan.md, Features #3) -- one plotting implementation,
    two consumers, so a GUI chart can never drift from the PNG in the thesis.
    A returned figure is the caller's to close.
    """
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    return fig


def plot_worst_case_voltage(strategies: dict[str, dict], week_index: int,
                            output_path: Path | None = None):
    """Line plot: worst-case (extreme) node voltage over one week, one line
    per control strategy, with the [V_LOW, V_HIGH] safety band overlaid."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in STRATEGY_ORDER:
        if name not in strategies:
            continue
        data = strategies[name]
        week_mask = data["week_of"] == week_index
        week_voltages = data["voltages"].loc[week_mask]
        extreme = _extreme_voltage(week_voltages)
        # Re-base the x-axis onto hours-of-day so all strategies overlay
        # regardless of the synthetic calendar date they were stamped with.
        # Each week_N folder is one representative simulated day for that
        # week (the weekly load/PV/EV multiplier applied to a single day),
        # not seven days of data.
        hours = (extreme.index - extreme.index[0]).total_seconds() / 3600.0
        ax.plot(hours, extreme.values, label=STRATEGY_LABELS[name],
                color=STRATEGY_COLORS[name], linewidth=2)

    ax.axhline(V_LOW, color="#898781", linestyle="--", linewidth=1,
              label=f"Safety band [{V_LOW}, {V_HIGH}] p.u.")
    ax.axhline(V_HIGH, color="#898781", linestyle="--", linewidth=1)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Worst-case node voltage [p.u.]")
    ax.set_title(f"Worst-case node voltage by control strategy -- week {week_index}'s representative day")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    return _finish(fig, output_path)


def plot_total_losses_bar(summary: pd.DataFrame, output_path: Path | None = None):
    """Grouped bar chart: total annual active-power losses per strategy."""
    ordered = summary.reindex([STRATEGY_LABELS[n] for n in STRATEGY_ORDER
                               if STRATEGY_LABELS[n] in summary.index])
    colors = [STRATEGY_COLORS[n] for n in STRATEGY_ORDER
             if STRATEGY_LABELS[n] in summary.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = sns.barplot(x=ordered.index, y=ordered["total_active_losses_kwh"],
                      hue=ordered.index, palette=colors, legend=False, ax=ax)
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(f"{height:,.0f}", (bar.get_x() + bar.get_width() / 2, height),
                   ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Control strategy")
    ax.set_ylabel("Total active-power losses [kWh]")
    ax.set_title("Annual line losses by control strategy")
    return _finish(fig, output_path)


def plot_reactive_effort_box(strategies: dict[str, dict], output_path: Path | None = None):
    """Boxplot: distribution of per-inverter total reactive-power effort for
    the three active controllers (No Control never sets Q, so it is
    excluded -- every value would be zero)."""
    controlled = [n for n in STRATEGY_ORDER if n != "NoControl" and n in strategies]
    records = []
    for name in controlled:
        effort = per_inverter_reactive_effort(strategies[name]["reactive_power"])
        for value in effort.values:
            records.append({"strategy": STRATEGY_LABELS[name], "effort_kvar": value})
    # Named columns even when empty: pd.DataFrame([]) has no columns at all,
    # and seaborn then rejects x="strategy" as a name that does not exist.
    # A run of only No Control is a legitimate baseline -- there is simply no
    # reactive effort to show -- so it must produce an empty figure, not an
    # exception after the batch has already been simulated.
    long_df = pd.DataFrame(records, columns=["strategy", "effort_kvar"])

    fig, ax = plt.subplots(figsize=(7, 5))
    order = [STRATEGY_LABELS[n] for n in controlled]
    palette = {STRATEGY_LABELS[n]: STRATEGY_COLORS[n] for n in controlled}
    if long_df.empty:
        ax.text(0.5, 0.5,
                "No reactive-power effort to show\n"
                "(no controlling strategy in this run)",
                ha="center", va="center", transform=ax.transAxes, color="#555555")
        ax.set_xticks([])
    else:
        sns.boxplot(data=long_df, x="strategy", y="effort_kvar", order=order,
                   hue="strategy", palette=palette, legend=False, ax=ax)
        sns.stripplot(data=long_df, x="strategy", y="effort_kvar", order=order,
                     color="#0b0b0b", alpha=0.4, size=3, ax=ax)

    # Per-inverter effort is heavily right-skewed (most inverters barely
    # engage their controller, a handful dominate the total) with genuine
    # zeros mixed in -- a linear axis squashes every box but the largest
    # outliers into an invisible sliver, so use symlog (linear near 0,
    # logarithmic beyond linthresh) rather than a plain log axis that
    # cannot plot the zeros at all.
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("Control strategy")
    ax.set_ylabel("Per-inverter total |Q| effort [kVAr, symlog scale]")
    ax.set_title("Reactive-power effort distribution across inverters\n"
                "(summed per inverter over the whole scenario)")
    return _finish(fig, output_path)


def plot_transformer_loading_bar(summary: pd.DataFrame, output_path: Path | None = None):
    """Grouped bar chart: peak MV/LV substation transformer loading per
    strategy, with a dashed line at 100% (nameplate rating -- congestion
    above this, independent of whether LV-bus voltages stayed in band)."""
    ordered = summary.reindex([STRATEGY_LABELS[n] for n in STRATEGY_ORDER
                               if STRATEGY_LABELS[n] in summary.index])
    colors = [STRATEGY_COLORS[n] for n in STRATEGY_ORDER
             if STRATEGY_LABELS[n] in summary.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = sns.barplot(x=ordered.index, y=ordered["transformer_loading_max_pct"],
                      hue=ordered.index, palette=colors, legend=False, ax=ax)
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(f"{height:,.1f}", (bar.get_x() + bar.get_width() / 2, height),
                   ha="center", va="bottom", fontsize=10)

    ax.axhline(100.0, color="#898781", linestyle="--", linewidth=1,
              label="Nameplate rating (100%)")
    ax.set_xlabel("Control strategy")
    ax.set_ylabel("Peak transformer loading [%]")
    ax.set_title("Peak MV/LV substation transformer loading by control strategy")
    ax.legend(loc="best", frameon=True)
    return _finish(fig, output_path)


def plot_fairness_and_signflips_bar(summary: pd.DataFrame, output_path: Path | None = None):
    """Two-panel bar chart: reactive-power fairness (Gini index of
    per-inverter total effort, lower = more evenly shared) and mean
    sign-flip frequency per inverter per day (DC-link stress proxy), side by
    side per strategy. No Control is included for reference (both read as
    zero -- it never sets Q)."""
    ordered = summary.reindex([STRATEGY_LABELS[n] for n in STRATEGY_ORDER
                               if STRATEGY_LABELS[n] in summary.index])
    colors = [STRATEGY_COLORS[n] for n in STRATEGY_ORDER
             if STRATEGY_LABELS[n] in summary.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=ordered.index, y=ordered["reactive_fairness_gini"],
               hue=ordered.index, palette=colors, legend=False, ax=ax1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Control strategy")
    ax1.set_ylabel("Gini index of per-inverter Q effort")
    ax1.set_title("Reactive-power fairness\n(lower = more evenly shared)")

    sns.barplot(x=ordered.index, y=ordered["sign_flips_mean_per_inverter_per_day"],
               hue=ordered.index, palette=colors, legend=False, ax=ax2)
    ax2.set_xlabel("Control strategy")
    ax2.set_ylabel("Mean sign flips / inverter / day")
    ax2.set_title("Capacitive <-> inductive switching\n(DC-link stress proxy)")
    return _finish(fig, output_path)


# ---------------------------------------------------------------------------
# 6. CLI / orchestration
# ---------------------------------------------------------------------------
RUN_META_FILENAME = "run_meta.json"


def write_run_meta(output_dir: Path, root: Path, scenario: str | None,
                   week_index: int, strategies: list[str]) -> Path:
    """Record which results tree this analysis folder was built from, as
    <output_dir>/run_meta.json.

    `root` is a .../sim_results/<timestamp>/sim_results_combined path, so the
    run timestamp is its grandparent's name -- stored explicitly rather than
    left to be re-parsed by every reader. When --scenario was omitted the
    scenario actually used is resolved here, so the file always names one.
    """
    if scenario is None and strategies:
        scenario = discover_scenario_dir(root / strategies[0], None).name
    meta = {
        "root": str(root.resolve()),
        "run_timestamp": root.resolve().parent.name,
        "scenario": scenario,
        "week_index": week_index,
        "strategies": strategies,
        "written": datetime.now().isoformat(timespec="seconds"),
    }
    path = output_dir / RUN_META_FILENAME
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and compare control strategies from a completed "
                    "yearly_tests.py --yearly batch (a sim_results_combined/ tree).")
    parser.add_argument("--root", type=Path, required=True,
                       help="Path to the sim_results_combined/ folder for one run, "
                            "e.g. sim_results/<timestamp>/sim_results_combined")
    parser.add_argument("--scenario", type=str, default=None,
                       help="Scenario folder name under each strategy, e.g. "
                            "pv_5_ev_40_files_max. Auto-detected if omitted and "
                            "only one scenario is present.")
    parser.add_argument("--week", type=int, default=None,
                       help="Week index (0-52) for the voltage line plot. "
                            "Defaults to the week with the worst voltage "
                            "excursion under No Control.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"),
                       help="Where to write the PNGs and summary CSV.")
    parser.add_argument("--base-date", type=str, default="2024-01-01",
                       help="Arbitrary Monday used to build the synthetic "
                            "full-year time index weeks are aligned onto.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_date = pd.Timestamp(args.base_date)

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    print(f"Loading strategies from {args.root} ...")
    strategies = load_all_strategies(args.root, args.scenario, base_date)
    if not strategies:
        raise SystemExit(f"No control-strategy folders found under {args.root}")

    week_index = args.week if args.week is not None else select_worst_week(strategies)
    print(f"Using week_{week_index} for the voltage comparison plot.")

    summary = build_summary_table(strategies)
    summary_path = args.output_dir / "strategy_summary.csv"
    summary.to_csv(summary_path)
    print(f"Wrote {summary_path}")
    print(summary)

    # IMPORTANT TIME BASIS: a --yearly run simulates ONE representative day per
    # ISO week (n_weeks, = 53), so this table aggregates 53 DAYS, not a 365-day
    # year. Extensive totals (total_active_losses_kwh, total_excursion_puh,
    # total_reactive_effort_kvar, total_q_variation_kvar, sign_flips_total,
    # estimated_additional_inverter_losses_kwh) must be multiplied by ~7 (365/53)
    # for an annual figure; intensive metrics (all _pct, _max/peak_, _mean_,
    # per_day, gini/cv, rms) are period-independent and must NOT be scaled. The
    # RESULTS_BASIS.md note written beside this CSV spells out which is which.
    (args.output_dir / "RESULTS_BASIS.md").write_text(RESULTS_BASIS_NOTE, encoding="utf-8")

    # Provenance: which results tree this summary came from. Output-folder
    # names carry the location and EV/PV scenario but not the run timestamp,
    # so without this an analysis_output/ folder cannot be traced back to its
    # sim_results/<timestamp>/ run -- which is exactly what the desktop app's
    # History tab lists (desktop_app/backend/results_index.py reads this file;
    # folders written before it existed fall back to matching folder names and
    # scenarios, which cannot tell two runs of the same scenario apart).
    write_run_meta(args.output_dir, args.root, args.scenario, week_index,
                   sorted(strategies))

    plot_worst_case_voltage(strategies, week_index,
                           args.output_dir / "worst_case_voltage.png")
    plot_total_losses_bar(summary, args.output_dir / "total_losses_bar.png")
    plot_reactive_effort_box(strategies, args.output_dir / "reactive_effort_box.png")
    plot_transformer_loading_bar(summary, args.output_dir / "transformer_loading_bar.png")
    plot_fairness_and_signflips_bar(summary, args.output_dir / "fairness_and_signflips_bar.png")
    print(f"Wrote PNGs to {args.output_dir}")


if __name__ == "__main__":
    main()
