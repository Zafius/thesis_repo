"""
build_pv_source_internal.py -- consolidate the raw PVGIS hourly timeseries
(working_doc/research/pv_data/Timeseries_*.csv) into a single 53-column,
15-minute-resolution master table.

Mirrors data_source_internal.csv: one file network_setup.py reads directly,
replacing the old two-stage pipeline of 366 individual daily files
(pv_daily_profiles/) averaged by pv_daily_to_weekly.py into 53 weekly files
(pv_weekly_profiles/). Column week_N holds the reference 5 kWp system's
output in Watts for ISO week N -- same convention (raw system output, not
per-unit-normalized) the old pv_weekly_profile_*.csv files used.

The raw PVGIS export covers 2020-2021, but only PV_YEAR is used here so the
PV curve matches the single-year (2020) span of the ENTSO-E load data
(simulation/entsoe/build_yearly_load.py, thesis Ch. 4).

Re-run this whenever the raw PVGIS export or PV_YEAR changes. Normally you call
it through fetch_pvgis_timeseries.py --location <key> --build, which passes the
per-location output path (locations/<key>/pv_weekly.csv); run standalone it
rebuilds the default location's (Oslo) cache, which the simulation reads.
"""
import os
import re

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'working_doc', 'research', 'pv_data',
                        'Timeseries_59.961_10.868_SA3_5kWp_crystSi_14_46deg_-2deg_2020_2021.csv')
# Default output: the default location's PV cache
# (locations/Norway/oslo_no1/pv_weekly.csv). fetch_pvgis_timeseries.py
# overrides this per location via build(output_path=).
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '..', 'locations', 'Norway', 'oslo_no1', 'pv_weekly.csv')

PV_YEAR = '2020'  # keep in sync with the ENTSO-E load-data year (thesis Ch. 4)

# 00:00, 00:15, ..., 24:00 -- 97 points, the same day-length grid
# data_source_internal.csv and the old pv_weekly_profile_*.csv files use.
MINUTES_GRID = np.arange(0, 24 * 60 + 1, 15)

ROW_RE = re.compile(r'^(\d{8}):(\d{4}),([-\d.]+),')


def load_raw_hourly(raw_path=RAW_PATH, pv_year=PV_YEAR):
    rows = []
    with open(raw_path) as f:
        for line in f:
            m = ROW_RE.match(line)
            if not m:
                continue
            date_str, time_str, p_str = m.groups()
            if date_str[:4] != pv_year:
                continue
            minute = int(time_str[:2]) * 60 + int(time_str[2:])
            rows.append((date_str, minute, float(p_str)))
    return pd.DataFrame(rows, columns=['date', 'minute', 'P'])


def build(raw_path=RAW_PATH, output_path=OUTPUT_PATH, pv_year=PV_YEAR):
    """Collapse a raw PVGIS hourly export into the 15-min weekly master table.

    Kept parametric so fetch_pvgis_timeseries.py can chain a freshly downloaded
    export straight into a location's pv_weekly.csv without editing this file.
    """
    raw = load_raw_hourly(raw_path, pv_year)

    week_days = {}
    for date_str, day in raw.groupby('date'):
        day = day.sort_values('minute')
        day_curve = np.interp(MINUTES_GRID, day['minute'].to_numpy(), day['P'].to_numpy())
        iso_week = pd.Timestamp(
            f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}").isocalendar().week
        week_days.setdefault(iso_week, []).append(day_curve)

    weeks = sorted(week_days)
    out = pd.DataFrame({f'week_{w}': np.mean(week_days[w], axis=0) for w in weeks})
    out.to_csv(output_path)
    print(f"Wrote {output_path} ({out.shape[0]} rows x {out.shape[1]} weeks, "
          f"averaged over {len(raw['date'].unique())} source days)")
    return out


def main():
    build()


if __name__ == '__main__':
    main()
