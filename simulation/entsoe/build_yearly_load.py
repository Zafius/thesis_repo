#!/usr/bin/env python3
"""Build one chronologically-sorted yearly load file per ENTSOE area.

Reads all `YYYY_MM_ActualTotalLoad_*.csv` files in this directory (tab-
separated ENTSOE exports) and, for every year and every distinct area
(AreaDisplayName, e.g. "Austria (AT)", "DE(50Hertz)", "NO1"), writes a single
CSV with that area's rows in chronological order to
entsoe/yearly_load_data/countries/. Each row
also gets a per-unit load value, using that area's own yearly mean MW as the
pu base (1 pu = that area's average load over the year).

Areas are kept as reported by ENTSOE (country totals and bidding
zones/TSO sub-areas are separate series) since some countries only report
at the sub-area level.

For each area, a second file (`..._weekly_peak_min.csv`) is written with the
peak and minimum load (MW and pu) for each ISO week.
"""
import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).parent
OUT_DIR = SRC_DIR / "yearly_load_data" / "countries"

FILENAME_RE = re.compile(r"^(\d{4})_\d{2}_ActualTotalLoad_.*\.csv$")


def sanitize(name: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", name.strip())
    return name.strip("_")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC_DIR.glob("*_ActualTotalLoad_*.csv"))
    if not files:
        raise SystemExit(f"No ActualTotalLoad CSV files found in {SRC_DIR}")

    # year -> area display name -> list of rows
    data = defaultdict(lambda: defaultdict(list))

    for path in files:
        m = FILENAME_RE.match(path.name)
        if not m:
            print(f"Skipping unrecognized filename: {path.name}")
            continue
        year = m.group(1)
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                data[year][row["AreaDisplayName"]].append(row)

    for year, areas in sorted(data.items()):
        for area, rows in sorted(areas.items()):
            rows.sort(key=lambda r: r["DateTime(UTC)"])

            loads = [float(r["TotalLoad[MW]"]) for r in rows if r["TotalLoad[MW]"].strip() != ""]
            mean_load = sum(loads) / len(loads) if loads else 0.0

            out_path = OUT_DIR / f"{year}_{sanitize(area)}.csv"
            with out_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["DateTime(UTC)", "TotalLoad[MW]", "TotalLoad[pu]", "ResolutionCode", "AreaMapCode"]
                )
                for r in rows:
                    raw = r["TotalLoad[MW]"].strip()
                    pu = float(raw) / mean_load if raw and mean_load else ""
                    writer.writerow(
                        [r["DateTime(UTC)"], r["TotalLoad[MW]"], pu, r["ResolutionCode"], r["AreaMapCode"]]
                    )
            print(f"Wrote {out_path.relative_to(SRC_DIR)} ({len(rows)} rows, mean={mean_load:.1f} MW)")

            # weekly peak/min
            weeks = defaultdict(list)
            for r in rows:
                raw = r["TotalLoad[MW]"].strip()
                if not raw:
                    continue
                dt = datetime.strptime(r["DateTime(UTC)"], "%Y-%m-%d %H:%M:%S")
                iso_year, iso_week, _ = dt.isocalendar()
                weeks[(iso_year, iso_week)].append((dt, float(raw)))

            weekly_path = OUT_DIR / f"{year}_{sanitize(area)}_weekly_peak_min.csv"
            with weekly_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "ISOYear",
                        "ISOWeek",
                        "Peak[MW]",
                        "PeakDateTime(UTC)",
                        "Peak[pu]",
                        "Min[MW]",
                        "MinDateTime(UTC)",
                        "Min[pu]",
                    ]
                )
                for (iso_year, iso_week), points in sorted(weeks.items()):
                    peak_dt, peak_mw = max(points, key=lambda p: p[1])
                    min_dt, min_mw = min(points, key=lambda p: p[1])
                    peak_pu = peak_mw / mean_load if mean_load else ""
                    min_pu = min_mw / mean_load if mean_load else ""
                    writer.writerow(
                        [
                            iso_year,
                            iso_week,
                            peak_mw,
                            peak_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            peak_pu,
                            min_mw,
                            min_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            min_pu,
                        ]
                    )
            print(f"Wrote {weekly_path.relative_to(SRC_DIR)} ({len(weeks)} weeks)")


if __name__ == "__main__":
    main()
