"""
plot_thesis_figures.py -- cross-cutting figures for thesis Chapter 5.

The per-run PNGs that compare_control_strategies.py writes each show ONE
(location, scenario, load-extreme). Chapter 5 argues across those axes -- four
controllers x three locations x three penetration levels -- so it needs figures
the per-run plots cannot produce. This script builds them.

Reads simulation/analysis_output/phase{2,3}_*/ directly (not the packaged
snapshot in results_package/), so re-running it after new batches finish picks
them up with no other step. Combinations that have not been simulated yet are
skipped, and the figure is drawn from whatever exists -- so the same command
works before and after the medium/high yearly runs land.

Outputs (working_doc/Latex_thesis/figures/, prefix ch5_):
  ch5_voltage_compliance.png   violation % and excursion depth, per location
  ch5_penetration.png          violation % vs penetration level, per location
  ch5_cost_panel.png           line losses vs baseline, curtailment, Q effort
  ch5_fairness_signflips.png   Gini and sign-flip frequency
  ch5_cost_benefit.png         Q effort against voltage benefit, per location

Usage (from simulation/):
    python plot_thesis_figures.py
    python plot_thesis_figures.py --phase3-scenario base   # which yearly set to plot
    python plot_thesis_figures.py --outdir /some/other/figures

Design notes (see the dataviz method): the four controller colours are the same
entity->colour mapping compare_control_strategies.py uses, so a controller keeps
its colour across every figure in the thesis. That 4-slot palette was validated
for the adjacent-pair list used by grouped bars and, for the 3-slot subset the
scatter uses, for the all-pairs list. Two of the four sit below 3:1 contrast on
white, which obliges a table view -- every figure here is paired with a table
carrying the same numbers in Chapter 5.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = THIS_DIR / "analysis_output"
DEFAULT_OUTDIR = THIS_DIR.parent / "working_doc" / "Latex_thesis" / "figures"

LOCATIONS = ["oslo_no1", "athens", "berlin_50hertz"]
SCENARIOS = ["base", "medium", "high"]
SCENARIO_EL = {"base": "Βασικό", "medium": "Μεσαίο", "high": "Υψηλό"}
EXTREME_EL = {"max": "περιβάλλουσα αιχμής", "min": "περιβάλλουσα ελαφρού φορτίου"}

# Chapter 5 numbers the control scenarios Σ0..Σ3, and Σ1 is DROOP while Σ2 is
# PID -- not the order the CSVs list them in. Keep this mapping in one place.
STRATEGY_TO_SIGMA = {
    "No Control": "Σ0",
    "Droop": "Σ1",
    "PID": "Σ2",
    "Centralized": "Σ3",
}
SIGMA_ORDER = ["Σ0", "Σ1", "Σ2", "Σ3"]
ACTIVE_SIGMAS = ["Σ1", "Σ2", "Σ3"]

# Same entity->colour mapping as compare_control_strategies.STRATEGY_COLORS, so
# a controller is the same colour in every thesis figure.
SIGMA_COLORS = {
    "Σ0": "#2a78d6",  # blue   -- No Control (reference)
    "Σ1": "#eda100",  # yellow -- Droop
    "Σ2": "#1baf7a",  # aqua   -- PID
    "Σ3": "#008300",  # green  -- Centralized
}
SIGMA_LEGEND = {
    "Σ0": "Σ0 — χωρίς έλεγχο",
    "Σ1": "Σ1 — στατισμός",
    "Σ2": "Σ2 — PID",
    "Σ3": "Σ3 — κεντρικός",
}
# Penetration level -> marker, so the scatter encodes level by shape and
# controller by colour (identity never rests on colour alone).
SCENARIO_MARKERS = {"base": "o", "medium": "s", "high": "^"}

# Recessive ink/grid, per the method: text and axes never wear a series colour.
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d2"

CURTAIL_STRATEGY_MAP = {
    "NoControl": "No Control",
    "PIDcontrol": "PID",
    "Droopcontrol": "Droop",
    "CentralizedControl": "Centralized",
}


def _style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _grid(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
PHASE2_RE = re.compile(r"^phase2_(?P<loc>.+)_(?P<scen>base|medium|high)$")
PHASE3_RE = re.compile(r"^phase3_(?P<loc>.+)_(?P<scen>base|medium|high)_(?P<ext>max|min)$")


def load_results(analysis_root=ANALYSIS_ROOT):
    """One tidy row per (phase, location, scenario, extreme, sigma).

    Folders that do not exist yet are simply absent from the result -- callers
    plot what they get. curtailment_summary.csv is optional (it is a separate
    task and may lag the compare step), and its strategy labels differ from
    strategy_summary.csv's, so they are normalised here.
    """
    rows = []
    if not analysis_root.is_dir():
        return pd.DataFrame()

    for folder in sorted(analysis_root.iterdir()):
        if not folder.is_dir():
            continue
        m2, m3 = PHASE2_RE.match(folder.name), PHASE3_RE.match(folder.name)
        if m2:
            phase, loc, scen, ext = 2, m2["loc"], m2["scen"], "max"
        elif m3:
            phase, loc, scen, ext = 3, m3["loc"], m3["scen"], m3["ext"]
        else:
            continue
        if loc not in LOCATIONS:
            continue

        summary = folder / "strategy_summary.csv"
        if not summary.is_file():
            continue
        df = pd.read_csv(summary)

        curt = folder / "curtailment_summary.csv"
        if curt.is_file():
            c = pd.read_csv(curt)
            c["strategy"] = c["strategy"].map(CURTAIL_STRATEGY_MAP)
            df = df.merge(c[["strategy", "curtailed_pct", "total_curtailed_kwh"]],
                          on="strategy", how="left")
        else:
            df["curtailed_pct"] = np.nan
            df["total_curtailed_kwh"] = np.nan

        df["sigma"] = df["strategy"].map(STRATEGY_TO_SIGMA)
        df = df.dropna(subset=["sigma"])
        df["phase"], df["location"], df["scenario"], df["extreme"] = phase, loc, scen, ext
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _pivot(data, value, active_only=False):
    """location x sigma table, rows/cols in canonical order, missing = NaN.

    active_only drops Σ0. Use it for any quantity that is zero *by definition*
    under Σ0 -- reactive effort, curtailment, sign flips, losses-relative-to-Σ0,
    and Gini. Plotting those would draw an invisible zero bar and, worse, a Gini
    of 0 for Σ0 reads as "perfectly fair" when it actually means no reactive
    power was supplied at all.
    """
    t = data.pivot_table(index="location", columns="sigma", values=value)
    cols = ACTIVE_SIGMAS if active_only else SIGMA_ORDER
    return t.reindex(index=[l for l in LOCATIONS if l in t.index],
                     columns=[s for s in cols if s in t.columns])


def _grouped_bars(ax, table, ylabel, title):
    """Grouped bars: one group per location, one bar per controller.

    Bars carry a visible surface gap so the two low-contrast slots stay
    separable (the secondary encoding the palette's CVD floor band requires).
    """
    locs = list(table.index)
    sigmas = list(table.columns)
    n = len(sigmas)
    x = np.arange(len(locs))
    slot = 0.8 / max(n, 1)
    width = slot * 0.86  # remainder is the gap between adjacent bars

    for i, s in enumerate(sigmas):
        ax.bar(x - 0.4 + slot * (i + 0.5), table[s].to_numpy(), width,
               color=SIGMA_COLORS[s], label=SIGMA_LEGEND[s],
               edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(locs)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK)
    _grid(ax)


def _note_missing(fig, missing):
    if missing:
        fig.text(0.5, 0.005,
                 "Εκκρεμούν: " + ", ".join(missing),
                 ha="center", va="bottom", fontsize=7.5, color=INK_SOFT, style="italic")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_voltage_compliance(data, outdir, p3_scen="base"):
    """The benefit: violation percentage and excursion depth, both load extremes."""
    d = data[(data.phase == 3) & (data.scenario == p3_scen)]
    if d.empty:
        print("  [skip] ch5_voltage_compliance: no Phase 3 data")
        return
    # sharey per row: both panels in a row are the same quantity under two load
    # extremes, so an unshared axis would invite reading a taller bar as a larger
    # value across panels.
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharey="row")
    for r, (value, ylabel) in enumerate([
            ("voltage_violation_pct", "Παραβίαση τάσης [%]"),
            ("total_excursion_puh", "Υπέρβαση ορίων [p.u.·h]")]):
        for c, ext in enumerate(["max", "min"]):
            ax = axes[r][c]
            sub = d[d.extreme == ext]
            if sub.empty:
                ax.set_axis_off()
                continue
            # y is shared across the row, so only the left panel is labelled --
            # a label above blank ticks reads as a second, different scale.
            _grouped_bars(ax, _pivot(sub, value), ylabel if c == 0 else "",
                          EXTREME_EL[ext] if r == 0 else "")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = outdir / "ch5_voltage_compliance.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_unbalance_rms(data, outdir, p3_scen="base"):
    """Voltage quality beyond the compliance percentage: unbalance and RMS depth.

    Chapter 5 defines the VUF and the RMS deviation among its metrics but never
    reports them. They are worth reporting precisely because they disagree with
    the headline: the centralized controller lowers the WORST unbalance while
    raising the AVERAGE one, which no single compliance number can express.
    """
    d = data[(data.phase == 3) & (data.scenario == p3_scen) & (data.extreme == "max")]
    if d.empty:
        print("  [skip] ch5_unbalance_rms: no Phase 3 data")
        return
    panels = [("vuf_max_pct", "VUF μέγιστο [%]", "Μέγιστη ασυμμετρία"),
              ("vuf_mean_pct", "VUF μέσο [%]", "Μέση ασυμμετρία"),
              ("rms_voltage_deviation_pu", "Ενεργός απόκλιση [p.u.]",
               "Ενεργός (RMS) απόκλιση τάσης")]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    for ax, (value, ylabel, title) in zip(axes, panels):
        _grouped_bars(ax, _pivot(d, value), ylabel, title)
        # Bars start at zero, deliberately. Zooming the axis to the data range
        # would make these differences look far larger than they are -- and
        # "these metrics barely move" is the actual finding, so inflating them
        # would misreport it. Bar length encodes magnitude; a cropped baseline
        # breaks that.
        ax.set_ylim(bottom=0)
    # The EN 50160 limit is 2 %: worth drawing, because the headline is that
    # every scenario sits comfortably under it and unbalance is NOT the binding
    # constraint here.
    axes[0].axhline(2.0, color=INK_SOFT, linewidth=1.0, linestyle="--")
    axes[0].text(0.02, 2.0, " όριο EN 50160 (2 %)", transform=axes[0].get_yaxis_transform(),
                 va="bottom", fontsize=7.5, color=INK_SOFT)
    axes[0].set_ylim(top=max(axes[0].get_ylim()[1], 2.15))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    out = outdir / "ch5_unbalance_rms.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_inverter_stress(data, outdir, p3_scen="base"):
    """What the reactive service costs the inverter itself.

    Three quantities Chapter 5 defines and does not report: the I^2R conduction
    losses the service adds, how smoothly Q is delivered, and the peak |Q| each
    inverter is asked for. Σ0 is excluded throughout -- it supplies no reactive
    power, so all three are zero by definition.
    """
    d = data[(data.phase == 3) & (data.scenario == p3_scen) & (data.extreme == "max")]
    if d.empty:
        print("  [skip] ch5_inverter_stress: no Phase 3 data")
        return
    panels = [
        ("estimated_additional_inverter_losses_kwh",
         "Πρόσθετες απώλειες [kWh]", "Απώλειες αγωγής αντιστροφέων"),
        ("total_q_variation_kvar",
         r"$\sum|\Delta Q|$ [kVAr]", "Μεταβλητότητα άεργου ισχύος"),
        ("peak_instantaneous_q_kvar",
         "Αιχμή $|Q|$ [kVAr]", "Μέγιστη στιγμιαία άεργος ισχύς"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    for ax, (value, ylabel, title) in zip(axes, panels):
        _grouped_bars(ax, _pivot(d, value, active_only=True), ylabel, title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    out = outdir / "ch5_inverter_stress.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_transformer(data, outdir, p3_scen="base"):
    """Substation loading -- the congestion axis, independent of bus voltage.

    Reported because the answer is a negative one worth stating: the MV/LV
    transformer never approaches its rating in any scenario, so nothing here is
    trading voltage compliance against congestion.
    """
    d = data[(data.phase == 3) & (data.scenario == p3_scen) & (data.extreme == "max")]
    if d.empty:
        print("  [skip] ch5_transformer: no Phase 3 data")
        return
    panels = [("transformer_loading_max_pct", "Μέγιστη φόρτιση [%]", "Αιχμή φόρτισης"),
              ("transformer_loading_mean_pct", "Μέση φόρτιση [%]", "Μέση φόρτιση"),
              ("transformer_peak_s_kva", "Αιχμή $S$ [kVA]", "Αιχμή φαινόμενης ισχύος")]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    for ax, (value, ylabel, title) in zip(axes, panels):
        _grouped_bars(ax, _pivot(d, value), ylabel, title)
        ax.set_ylim(bottom=0)   # see fig_unbalance_rms: bars start at zero
    # Drawn to scale against the nameplate rating, which is the whole point:
    # the bars are supposed to look small. Zooming in on a 15 % peak would
    # dramatise a quantity whose finding is that it never comes close.
    axes[0].axhline(100.0, color=INK_SOFT, linewidth=1.0, linestyle="--")
    axes[0].text(0.02, 100.0, " ονομαστική ισχύς (100 %)",
                 transform=axes[0].get_yaxis_transform(),
                 va="bottom", fontsize=7.5, color=INK_SOFT)
    axes[0].set_ylim(0, 112)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.text(0.5, 0.055, "Σε κανένα σενάριο δεν καταγράφεται βήμα άνω της ονομαστικής "
                         "ισχύος του μετασχηματιστή (0 % του χρόνου).",
             ha="center", fontsize=7.5, color=INK_SOFT, style="italic")
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    out = outdir / "ch5_transformer.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_penetration_annual(data, outdir):
    """Penetration sensitivity on the ANNUAL basis, both load envelopes.

    The companion to fig_penetration, which reads a single worst day. One day
    turns out to be idiosyncratic: it makes the penetration response look
    location-dependent, and the 53-day average shows the same shape everywhere.
    Reporting only the day would overstate how much the geography matters.
    """
    d = data[data.phase == 3]
    if d.empty:
        print("  [skip] ch5_penetration_annual: no Phase 3 data")
        return
    fig, axes = plt.subplots(2, len(LOCATIONS), figsize=(9.6, 6.0), sharey="row")
    missing = []
    for r, ext in enumerate(["max", "min"]):
        for c, loc in enumerate(LOCATIONS):
            ax = axes[r][c]
            sub = d[(d.location == loc) & (d.extreme == ext)]
            for s in SIGMA_ORDER:
                ss = sub[sub.sigma == s].set_index("scenario")
                ys = [ss["voltage_violation_pct"].get(sc, np.nan) for sc in SCENARIOS]
                ax.plot(range(len(SCENARIOS)), ys, marker="o", markersize=5,
                        linewidth=2, color=SIGMA_COLORS[s], label=SIGMA_LEGEND[s])
            for sc in SCENARIOS:
                if sub[sub.scenario == sc].empty:
                    missing.append(f"{loc}/{SCENARIO_EL[sc]}/{ext}")
            ax.set_xticks(range(len(SCENARIOS)))
            ax.set_xticklabels([SCENARIO_EL[s] for s in SCENARIOS])
            if r == 0:
                ax.set_title(loc, color=INK)
            if c == 0:
                ax.set_ylabel(f"Παραβίαση τάσης [%]\n({EXTREME_EL[ext]})")
            _grid(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    _note_missing(fig, sorted(set(missing)))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = outdir / "ch5_penetration_annual.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# Every metric the summary carries, grouped into families that share a story.
# Together these cover the full set Chapter 5 defines -- the point of the
# sweeps is that no metric is reported at one penetration level only.
METRIC_FAMILIES = {
    "voltage": ("Ποιότητα τάσης", [
        ("voltage_violation_pct", "Παραβίαση [%]"),
        ("total_excursion_puh", "Υπέρβαση [p.u.·h]"),
        ("rms_voltage_deviation_pu", "RMS απόκλιση [p.u.]"),
        ("vuf_max_pct", "VUF μέγ. [%]"),
        ("vuf_mean_pct", "VUF μέσο [%]"),
    ]),
    "cost": ("Κόστος", [
        ("total_active_losses_kwh", "Απώλειες γραμμών [kWh]"),
        ("peak_instantaneous_loss_kw", "Αιχμή απωλειών [kW]"),
        ("estimated_additional_inverter_losses_kwh", "Απώλειες αντιστρ. [kWh]"),
        ("curtailed_pct", "Περικοπή [%]"),
        ("mean_seconds_per_step", "Χρόνος/βήμα [s]"),
    ]),
    "reactive": ("Άεργος ισχύς", [
        ("total_reactive_effort_kvar", "Προσπάθεια $\\sum|Q|$ [kVAr]"),
        ("total_q_variation_kvar", "Μεταβλητότητα $\\sum|\\Delta Q|$ [kVAr]"),
        ("peak_instantaneous_q_kvar", "Αιχμή $|Q|$ [kVAr]"),
        ("sign_flips_mean_per_inverter_per_day", "Εναλλαγές [ημ.$^{-1}$]"),
    ]),
    "network": ("Δικαιοσύνη και δίκτυο", [
        ("reactive_fairness_gini", "Gini"),
        ("reactive_fairness_cv", "Συντ. μεταβλητότητας"),
        ("transformer_loading_max_pct", "Μ/Σ μέγ. φόρτιση [%]"),
        ("transformer_loading_mean_pct", "Μ/Σ μέση φόρτιση [%]"),
        ("transformer_peak_s_kva", "Μ/Σ αιχμή $S$ [kVA]"),
    ]),
}

# Quantities that are zero by definition under Σ0 (see _pivot's docstring):
# plotting Σ0 would draw a flat zero line and, for Gini, one that reads as
# "perfectly fair" when it means "supplied no reactive power at all".
ACTIVE_ONLY_METRICS = {
    "total_reactive_effort_kvar", "total_q_variation_kvar",
    "peak_instantaneous_q_kvar", "sign_flips_mean_per_inverter_per_day",
    "reactive_fairness_gini", "reactive_fairness_cv",
    "estimated_additional_inverter_losses_kwh", "curtailed_pct",
}


def fig_metric_sweep(data, outdir, family, extreme="max"):
    """Every metric in one family against penetration level, per location.

    Penetration is on the x-axis rather than faceted, because the question
    these answer is "how does this metric move as penetration rises" -- a
    trend is read along an axis, not across panels. One row per metric, one
    column per location, so a row can be compared left-to-right for
    geographic robustness and read across for the penetration response.
    """
    title, metrics = METRIC_FAMILIES[family]
    d = data[(data.phase == 3) & (data.extreme == extreme)]
    if d.empty:
        print(f"  [skip] ch5_sweep_{family}: no Phase 3 data")
        return

    nrow, ncol = len(metrics), len(LOCATIONS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.15 * ncol, 1.95 * nrow),
                             sharex=True)
    axes = np.atleast_2d(axes)
    for r, (metric, ylabel) in enumerate(metrics):
        if metric not in d.columns:
            for c in range(ncol):
                axes[r][c].set_axis_off()
            continue
        sigmas = ACTIVE_SIGMAS if metric in ACTIVE_ONLY_METRICS else SIGMA_ORDER
        # One shared y-scale per row: the same quantity in three locations, so
        # an unshared axis would invite reading a taller line as a larger value.
        vals = d[metric].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        for c, loc in enumerate(LOCATIONS):
            ax = axes[r][c]
            sub = d[d.location == loc]
            for s in sigmas:
                ss = sub[sub.sigma == s].set_index("scenario")
                ys = [ss[metric].get(sc, np.nan) for sc in SCENARIOS]
                ax.plot(range(len(SCENARIOS)), ys, marker="o", markersize=4.5,
                        linewidth=1.8, color=SIGMA_COLORS[s], label=SIGMA_LEGEND[s])
            if finite.size:
                lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
                pad = (hi - lo) * 0.12 or max(abs(hi) * 0.05, 1e-9)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks(range(len(SCENARIOS)))
            ax.set_xticklabels([SCENARIO_EL[s] for s in SCENARIOS])
            if r == 0:
                ax.set_title(loc, color=INK)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            _grid(ax)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.004))
    fig.suptitle(f"{title} — ετήσια βάση, {EXTREME_EL[extreme]}", y=1.005, color=INK)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    out = outdir / f"ch5_sweep_{family}_{extreme}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def write_metric_tables(data, outdir, extreme="max"):
    """Emit one LaTeX longtable per metric family: every metric, every level.

    Generated rather than hand-written because the alternative is transcribing
    ~800 numbers by hand -- the exact operation that put a superseded run's
    figures in the thesis in the first place. \\input these; regenerating the
    figures regenerates the numbers beside them, so the two cannot disagree.
    """
    d = data[(data.phase == 3) & (data.extreme == extreme)]
    if d.empty:
        print("  [skip] metric tables: no Phase 3 data")
        return
    tabdir = outdir.parent / "tables"
    tabdir.mkdir(parents=True, exist_ok=True)

    def fmt(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "---"
        a = abs(v)
        if a >= 1000:
            s = f"{v:,.0f}".replace(",", "\\,")
        elif a >= 10:
            s = f"{v:.1f}"
        elif a >= 1:
            s = f"{v:.2f}"
        else:
            s = f"{v:.4f}"
        return s.replace(".", "{,}")     # Greek decimal comma

    for family, (title, metrics) in METRIC_FAMILIES.items():
        lines = [
            "% Generated by plot_thesis_figures.py -- do not edit by hand.",
            "\\begin{longtable}{@{}l l " + "r" * len(SCENARIOS) + "@{}}",
            f"\\caption{{{title}: όλοι οι δείκτες ανά επίπεδο διείσδυσης "
            f"(ετήσια βάση, {EXTREME_EL[extreme]}).}}"
            f"\\label{{tab:sweep-{family}-{extreme}}}\\\\",
            "\\hline",
            "Δείκτης & Σεν. & " + " & ".join(SCENARIO_EL[s] for s in SCENARIOS) + " \\\\",
            "\\hline\\endfirsthead",
            "\\hline",
            "Δείκτης & Σεν. & " + " & ".join(SCENARIO_EL[s] for s in SCENARIOS) + " \\\\",
            "\\hline\\endhead",
        ]
        for metric, label in metrics:
            if metric not in d.columns:
                continue
            sigmas = ACTIVE_SIGMAS if metric in ACTIVE_ONLY_METRICS else SIGMA_ORDER
            # Averaged across the three locations: the per-location detail is
            # the sweep figures' job, and a table that carried it would be
            # 3x longer without changing any conclusion.
            for i, s in enumerate(sigmas):
                cells = []
                for sc in SCENARIOS:
                    sub = d[(d.sigma == s) & (d.scenario == sc)][metric]
                    cells.append(fmt(float(sub.mean()) if len(sub) else np.nan))
                # Label on the first row only, blank after -- deliberately not
                # \multirow, which would make these tables depend on a package
                # main.tex does not load (and would have to be added to both
                # LaTeX copies). Reads the same; needs nothing.
                head = label if i == 0 else ""
                lines.append(f"{head} & {s} & " + " & ".join(cells) + " \\\\")
            lines.append("\\addlinespace[2pt]")
        lines += [
            "\\hline",
            "\\multicolumn{" + str(2 + len(SCENARIOS)) + "}{@{}p{12cm}@{}}{\\footnotesize "
            "Μέσος όρος των τριών τοποθεσιών· η ανά τοποθεσία ανάλυση δίνεται "
            "στις αντίστοιχες εικόνες. Κενό («---») σημαίνει δείκτη που δεν "
            "ορίζεται για το σενάριο.}\\\\",
            "\\end{longtable}",
        ]
        out = tabdir / f"ch5_sweep_{family}_{extreme}.tex"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote tables/{out.name}")


def fig_penetration(data, outdir):
    """Penetration sensitivity from Phase 2 (peak day), one panel per location."""
    d = data[data.phase == 2]
    if d.empty:
        print("  [skip] ch5_penetration: no Phase 2 data")
        return
    fig, axes = plt.subplots(1, len(LOCATIONS), figsize=(9.6, 3.2), sharey=True)
    missing = []
    for ax, loc in zip(np.atleast_1d(axes), LOCATIONS):
        sub = d[d.location == loc]
        for s in SIGMA_ORDER:
            ss = sub[sub.sigma == s].set_index("scenario")
            ys = [ss["voltage_violation_pct"].get(sc, np.nan) for sc in SCENARIOS]
            ax.plot(range(len(SCENARIOS)), ys, marker="o", markersize=5,
                    linewidth=2, color=SIGMA_COLORS[s], label=SIGMA_LEGEND[s])
        for sc in SCENARIOS:
            if sub[sub.scenario == sc].empty:
                missing.append(f"{loc}/{SCENARIO_EL[sc]}")
        ax.set_xticks(range(len(SCENARIOS)))
        ax.set_xticklabels([SCENARIO_EL[s] for s in SCENARIOS])
        ax.set_title(loc, color=INK)
        _grid(ax)
    np.atleast_1d(axes)[0].set_ylabel("Παραβίαση τάσης [%]")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    _note_missing(fig, sorted(set(missing)))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    out = outdir / "ch5_penetration.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_cost_panel(data, outdir, p3_scen="base"):
    """The cost side: losses relative to Σ0, curtailed energy, total Q effort."""
    d = data[(data.phase == 3) & (data.scenario == p3_scen) & (data.extreme == "max")]
    if d.empty:
        print("  [skip] ch5_cost_panel: no Phase 3 data")
        return

    # All three quantities are zero by definition under Σ0, which is the baseline
    # every one of them is measured against -- so Σ0 is the axis, not a bar.
    losses = _pivot(d, "total_active_losses_kwh")
    rel = (losses.div(losses["Σ0"], axis=0).sub(1.0).mul(100.0)
           .reindex(columns=[s for s in ACTIVE_SIGMAS if s in losses.columns])
           if "Σ0" in losses else losses)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    _grouped_bars(axes[0], rel, "Μεταβολή απωλειών έναντι Σ0 [%]", "Απώλειες γραμμών")
    axes[0].axhline(0, color=INK_SOFT, linewidth=0.9)
    _grouped_bars(axes[1], _pivot(d, "curtailed_pct", active_only=True),
                  "Περικοπή ενεργού ισχύος [%]", "Περικοπή φόρτισης/παραγωγής")
    eq = _pivot(d, "total_reactive_effort_kvar", active_only=True).div(1000.0)
    _grouped_bars(axes[2], eq, "Προσπάθεια $E_Q$ [10³ kVAr]", "Προσπάθεια άεργου ισχύος")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    out = outdir / "ch5_cost_panel.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_fairness_signflips(data, outdir, p3_scen="base"):
    """Distribution fairness and switching stress -- the counter-intuitive pair."""
    d = data[(data.phase == 3) & (data.scenario == p3_scen) & (data.extreme == "max")]
    if d.empty:
        print("  [skip] ch5_fairness_signflips: no Phase 3 data")
        return
    # Σ0 excluded: it supplies no reactive power, so both metrics are trivially
    # zero for it and a Gini of 0 would read as "perfectly fair".
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    _grouped_bars(axes[0], _pivot(d, "reactive_fairness_gini", active_only=True),
                  "Δείκτης Gini", "Δικαιοσύνη κατανομής (0 = ομοιόμορφη)")
    _grouped_bars(axes[1], _pivot(d, "sign_flips_mean_per_inverter_per_day", active_only=True),
                  "Εναλλαγές [ανά αντιστροφέα·ημέρα]", "Συχνότητα εναλλαγής προσήμου $Q$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    out = outdir / "ch5_fairness_signflips.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_cost_benefit(data, outdir):
    """Q effort against the voltage benefit it buys, Phase 2, per location.

    Benefit is the drop in violation percentage against Σ0 in the same
    (location, scenario), so each point is a like-for-like comparison. A point
    lower and further right is a better deal; the dashed rays are equal-cost
    -per-benefit references.
    """
    d = data[data.phase == 2]
    if d.empty:
        print("  [skip] ch5_cost_benefit: no Phase 2 data")
        return
    # sharex as well as sharey: both axes carry the same quantity in every panel,
    # so the three locations must be readable against one common frame.
    fig, axes = plt.subplots(1, len(LOCATIONS), figsize=(9.6, 3.6),
                             sharey=True, sharex=True)
    for ax, loc in zip(np.atleast_1d(axes), LOCATIONS):
        for scen in SCENARIOS:
            sub = d[(d.location == loc) & (d.scenario == scen)]
            if sub.empty:
                continue
            base = sub[sub.sigma == "Σ0"]["voltage_violation_pct"]
            if base.empty:
                continue
            v0 = float(base.iloc[0])
            for s in ACTIVE_SIGMAS:
                r = sub[sub.sigma == s]
                if r.empty:
                    continue
                benefit = v0 - float(r["voltage_violation_pct"].iloc[0])
                cost = float(r["total_reactive_effort_kvar"].iloc[0]) / 1000.0
                ax.scatter(benefit, cost, s=54, color=SIGMA_COLORS[s],
                           marker=SCENARIO_MARKERS[scen], edgecolor="white",
                           linewidth=0.9, zorder=3)
        ax.set_title(loc, color=INK)
        _grid(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.9)
    np.atleast_1d(axes)[0].set_ylabel("Κόστος: $E_Q$ [10³ kVAr]")
    # One shared x label -- repeating it under each panel overflowed the figure.
    fig.supxlabel("Όφελος: μείωση παραβίασης τάσης έναντι Σ0 [μονάδες %]",
                  fontsize=9, color=INK)

    # Two legends: colour = controller, shape = penetration level.
    ctrl = [plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                       markerfacecolor=SIGMA_COLORS[s], markeredgecolor="white",
                       label=SIGMA_LEGEND[s]) for s in ACTIVE_SIGMAS]
    lvl = [plt.Line2D([], [], marker=SCENARIO_MARKERS[sc], linestyle="", markersize=7,
                      markerfacecolor=INK_SOFT, markeredgecolor="white",
                      label=SCENARIO_EL[sc]) for sc in SCENARIOS]
    fig.legend(handles=ctrl + lvl, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    out = outdir / "ch5_cost_benefit.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-root", type=Path, default=ANALYSIS_ROOT,
                    help="Root holding phase2_*/phase3_* folders (default: simulation/analysis_output).")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                    help=f"Where the PNGs go (default: {DEFAULT_OUTDIR}).")
    ap.add_argument("--phase3-scenario", default="base", choices=SCENARIOS,
                    help="Which penetration level the yearly figures use (default: base).")
    args = ap.parse_args()

    data = load_results(args.analysis_root)
    if data.empty:
        print(f"[error] no phase2_*/phase3_* results under {args.analysis_root}.")
        return 1

    have = (data.groupby(["phase", "scenario", "extreme"]).size()
            .rename("rows").reset_index())
    print("Result sets found:")
    for _, r in have.iterrows():
        print(f"  phase {r['phase']}  {r['scenario']:<7s} {r['extreme']:<4s} "
              f"({r['rows'] // 4} location(s))")

    args.outdir.mkdir(parents=True, exist_ok=True)
    _style()
    print(f"\nWriting figures to {args.outdir}")
    # Both load envelopes. The light-load one is not redundant: it is where
    # PV over-voltage dominates instead of evening charging, and several
    # metrics reverse direction between the two.
    for extreme in ("max", "min"):
        for family in METRIC_FAMILIES:
            fig_metric_sweep(data, args.outdir, family, extreme)
        write_metric_tables(data, args.outdir, extreme)
    fig_voltage_compliance(data, args.outdir, args.phase3_scenario)
    fig_penetration(data, args.outdir)
    fig_penetration_annual(data, args.outdir)
    fig_cost_panel(data, args.outdir, args.phase3_scenario)
    fig_fairness_signflips(data, args.outdir, args.phase3_scenario)
    fig_cost_benefit(data, args.outdir)
    fig_unbalance_rms(data, args.outdir, args.phase3_scenario)
    fig_inverter_stress(data, args.outdir, args.phase3_scenario)
    fig_transformer(data, args.outdir, args.phase3_scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
