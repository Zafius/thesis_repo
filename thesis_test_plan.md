# Thesis simulation test plan

Deadline: 2026-07-31 (2 weeks from today). Scope: controller comparison, EV/PV penetration sensitivity, worst-case (full-year) robustness, across three contrasting locations.

## 1. What each result needs

| Result | What it shows | Script | Key flag |
|---|---|---|---|
| Controller comparison | PID vs Droop vs Centralized vs No Control, same day/network | `yearly_tests.py` | `--controls 0 1 2 3` |
| EV/PV penetration sensitivity | How each controller's benefit scales with EV/PV share | `sim_config.py` (`EV_CONTRIBUTION`, `PV_CONTRIBUTION`) + `yearly_tests.py` | edit config, re-tune, re-run |
| Worst-case robustness | Does the peak-day result hold over the full year | `yearly_tests.py` | `--yearly` |

## 2. Locations (contrasting climates)

- `oslo_no1` — Norway, cold climate, low solar yield
- `athens` — Mediterranean, high solar yield
- `berlin_50hertz` — Central Europe, moderate on both axes

These three are already used as examples in `CLI_GUIDE.md`, so their load/PV data will fetch and cache cleanly on first use.

## 3. Penetration scenarios

Current default in `sim_config.py`: `EV_CONTRIBUTION = 0.4`, `PV_CONTRIBUTION = 0.05`. The Base scenario below no longer matches this default (EV share is lower) — re-tune gains for all three scenarios in Phase 1 rather than reusing the sim_config.py default's cached gains.

| Scenario | EV share | PV share | Purpose |
|---|---|---|---|
| Base | 0.20 | 0.05 | Lower bound — does the controller still matter? |
| Medium | 0.40 | 0.15 | Mid-range, matches the default EV share at a higher PV share |
| High | 0.60 | 0.30 | Stress case — does anything saturate or go unstable? |

Each (location x scenario) combination gets its own cached gains file (`gains/pid_ev*_pv*_seed42.csv`), so changing `EV_CONTRIBUTION`/`PV_CONTRIBUTION` and re-running the optimizers never overwrites a previous scenario's tuning.

## 4. Priority order given the 2-week deadline

Prioritize in this order so you have a complete, defensible result set even if time runs out early. Every phase covers the full 3x3 grid (3 locations x 3 scenarios), and within each phase `base` completes for all 3 locations before `medium` starts anywhere, and `medium` for all 3 before `high` — so an early stop leaves a complete lower-penetration set rather than one location finished at all three scenarios and the others untouched.

> **Revised 2026-07-26.** Phase 3 originally covered only the base scenario at all 3 locations plus a single-location high-scenario addendum (16 yearly runs). It now covers all 9 combinations (36 yearly runs), so the robustness claim has the same coverage as the peak-day comparison and penetration sensitivity can be argued on year-round data. The base-scenario third was already complete when this changed; the 6 remaining (location, scenario) combinations are ~4 h of wall clock each, ~24 h total.

**Phase 1 (days 1-3): tune gains.**
For each of the 3 locations, for each of the 3 penetration scenarios, set `EV_CONTRIBUTION`/`PV_CONTRIBUTION` in `sim_config.py` and run both optimizers (Centralized and No Control need no tuning):
```bash
python pid_optimizer_approx.py --location oslo_no1
python droop_optimizer_approx.py --location oslo_no1
# repeat for athens, berlin_50hertz, x3 scenarios each = 9 combos, 18 optimizer runs
```
This is the one step that can't be parallelized away — everything downstream reads these cached gains.

**Phase 2 (days 3-6): single worst-case day, full comparison.**
Fast (~seconds to minutes per run), gives you the controller-comparison and penetration-sensitivity results in one pass:
```bash
python yearly_tests.py --controls 0 1 2 3 --location oslo_no1
python yearly_tests.py --controls 0 1 2 3 --location athens
python yearly_tests.py --controls 0 1 2 3 --location berlin_50hertz
# x3 penetration scenarios each = 9 runs total
```

**Phase 3 (days 6-11): full-year robustness, full 3x3.**
Full 53-week batches are the expensive part — all 3 locations x all 3 scenarios x all 4 controls (36 yearly runs), each producing both a max-load and a min-load representative-day set:
```bash
python yearly_tests.py --yearly --controls 0 1 2 3 --location oslo_no1
# ...repeat per location, x3 scenarios each
```
Run these through `run_test_plan.py`, which handles the scenario switching, ordering and per-controller timing itself and is resumable — a 24 h batch that gets interrupted picks up where it stopped rather than restarting.
> **Not adopted.** This section originally suggested dropping `--step` to `1` (15-min steps) for the yearly runs to cut runtime. The runs were executed at `--step 15` (1-min steps) instead, matching Phase 1/2 — confirmed against the completed output, where all 1272 Phase 3 week folders hold 1441 rows. Keeping one resolution across all phases means the yearly and peak-day figures are directly comparable with no normalisation caveat, at the cost of ~1 h per controller per yearly run. `run_test_plan.py` pins this as `STEP_YEARLY = 15`; change it there, not here, if the trade-off is ever revisited.

**Phase 4 (days 11-14): buffer.**
Post-processing, plots, and writing — see below. Don't schedule Phase 3 runs past day 11; full-year batches are the one place a stuck run can quietly eat your remaining time.

## 5. Post-processing

Three steps, all wired into `run_test_plan.py` as tracked tasks so they run automatically as each batch finishes:

- `compare_control_strategies.py --root sim_results/<ts>/sim_results_combined [--scenario ...] --output-dir analysis_output/<tag>` — aggregates a run folder into `strategy_summary.csv` + plots (see any `analysis_output/phase*/strategy_summary.csv` for the expected shape).
- `compute_curtailment.py --root <same> --location <key> [--scenario ...] --output-dir <same>` — the active-power curtailment metric: how much charging/PV energy each controller gave up to supply reactive power. Writes `curtailment_summary.csv` beside the above.
- `compress_results.py sim_results/<ts>` — gzips the raw tree's `all_bus_voltages_phase_*.csv` files and deletes the originals, taking a Phase 3 tree from ~16 GB to ~3.5 GB. Runs **only after** the two above have both succeeded, since it is the one step that deletes source data. Reversible with `--decompress`; see "Disk footprint" in `simulation/TEST_PLAN_EXECUTION.md`.

The first two take `--scenario` to single out one load extreme, which Phase 3 needs since a yearly run writes both a max-load and a min-load folder under one root. Compression is per tree, so it needs no `--scenario`.

## 6. Coverage checklist

Live status: `python run_test_plan.py --status` (162 tasks total — 144 simulation/analysis tasks plus 18 `compress` tasks added 2026-07-27).

- [x] Gains tuned for all 3 locations x 3 scenarios (PID + Droop) — Phase 1, 18/18
- [x] Single worst-case day, all 4 controllers, 3 locations x 3 scenarios — Phase 2 runs, 36/36
- [x] Full-year, all 4 controllers, 3 locations, **base** scenario — 12/36 of the Phase 3 runs
- [ ] Full-year, all 4 controllers, 3 locations, **medium** scenario — 12 runs, ~12 h
- [ ] Full-year, all 4 controllers, 3 locations, **high** scenario — 12 runs, ~12 h
- [x] `compare_control_strategies.py` summaries for every completed batch — 15/27
- [ ] `compute_curtailment.py` summaries for every completed batch — 0/27 tracked (the 15 completed batches were backfilled by hand and will be regenerated as tracked tasks on the next run; ~40 s total)
- [x] Sanity check: no `--pid-gains` fallback warnings in the logs (means a tuning step was skipped)

- [ ] Raw trees compressed after analysis — 18 tracked `compress` tasks; the already-finished batches were backfilled on 2026-07-27

> **All three post-processing steps are part of the plan.** `run_test_plan.py` runs `compare_control_strategies.py` (`compare` task) *and* `compute_curtailment.py` (`curtail` task) after every batch, one of each per test and per load-extreme in Phase 3 — so `strategy_summary.csv` and `curtailment_summary.csv` always land together. Curtailment used to be a manual pass that was easy to forget; it no longer is. Cost is negligible: ~0.6 s for a single day, ~5.5 s for a 53-week batch. Once both have passed, `compress_results.py` (`compress` task) shrinks the raw tree — ~7 min for a Phase 3 batch, and it reclaims ~12.5 GB of the ~16 GB each one occupies.

## 7. Risks / things that will bite you

- Missing gains at run time trigger an interactive `[y/N]` prompt — if you ever run `yearly_tests.py` unattended (background/piped) before Phase 1 finishes for that combo, it silently falls back to default gains and the result is not what you think it is. Always tune first.
- The high-penetration scenario is the one most likely to hit numerical instability in the PID/Droop optimizers (`simulate_surrogate` aborts on non-finite trajectories) — if the optimizer reports the WARNING about a parameter sitting at a search bound, widen `BOUNDS` in `pid_optimizer_approx.py`/`droop_optimizer_approx.py` before trusting that scenario's gains.
- Full-year runs use multiprocessing across weeks — don't run more than one `--yearly` batch at a time on the same machine or they'll contend for cores and both slow down.
