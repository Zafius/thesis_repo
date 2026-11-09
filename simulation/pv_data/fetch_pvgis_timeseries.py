"""
fetch_pvgis_timeseries.py -- generate the raw PVGIS hourly PV export dynamically
for any location, instead of hand-downloading it from the web tool at
https://re.jrc.ec.europa.eu/pvg_tools/en/tools.html.

The web tool is a front end to PVGIS's non-interactive ``seriescalc`` API; this
script calls that same API directly, so the file it writes is byte-for-byte the
format build_pv_source_internal.py already consumes
(Timeseries_<lat>_<lon>_..._<y1>_<y2>.csv). Reimplementing PVGIS locally is not
possible -- the tool reads the multi-terabyte SARAH3 satellite-radiation and
ERA5 databases server-side -- so hitting the official API is the faithful way to
"replicate the tool" and to move the PV location for a simulation run.

Defaults reproduce the committed dataset (SA3 rooftop near Oslo: 59.961 N,
10.868 E, 5 kWp c-Si, 14 % loss, optimum tilt/azimuth, 2020). Override any of
them on the command line to relocate the PV source.

Examples
--------
    # Reproduce the committed export (raw file only):
    python fetch_pvgis_timeseries.py

    # Build the PV cache for a named location (locations/athens/pv_weekly.csv):
    python fetch_pvgis_timeseries.py --location athens --build

    # Fixed 30 deg south-facing array over 2019-2020:
    python fetch_pvgis_timeseries.py --slope 30 --azimuth 0 \
        --startyear 2019 --endyear 2020

API reference: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en
"""
import argparse
import os

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'working_doc', 'research', 'pv_data')

# PVGIS non-interactive endpoint (v5.3 -> SARAH3, the database the committed
# export used). Bump the version segment only if PVGIS retires v5_3.
API_URL = 'https://re.jrc.ec.europa.eu/api/v5_3/seriescalc'

# Defaults chosen so a bare run reproduces the committed SA3 Oslo dataset.
DEFAULTS = dict(
    lat=59.961,
    lon=10.868,
    peakpower=5.0,       # kWp
    loss=14.0,           # system losses (%)
    pvtechchoice='crystSi',
    mountingplace='free',  # 'free' (free-standing) or 'building' (roof-integrated)
    slope=None,          # deg; None -> let PVGIS pick the optimum
    azimuth=None,        # deg (0 = south, PVGIS convention); None -> optimum
    startyear=2020,
    endyear=2020,
)


def build_params(args):
    """Translate CLI args into PVGIS seriescalc query parameters."""
    params = {
        'lat': args.lat,
        'lon': args.lon,
        'startyear': args.startyear,
        'endyear': args.endyear,
        'pvcalculation': 1,
        'peakpower': args.peakpower,
        'loss': args.loss,
        'pvtechchoice': args.pvtechchoice,
        'mountingplace': args.mountingplace,
        'outputformat': 'csv',
        'browser': 0,
    }
    # PVGIS optimises whichever angle we leave unspecified. optimalangles=1
    # optimises both; optimalinclination=1 optimises tilt only.
    if args.slope is None and args.azimuth is None:
        params['optimalangles'] = 1
    elif args.slope is None:
        params['optimalinclination'] = 1
        params['aspect'] = args.azimuth
    elif args.azimuth is None:
        # No "optimise azimuth only" flag exists; pin a south-facing default.
        params['angle'] = args.slope
        params['aspect'] = 0
    else:
        params['angle'] = args.slope
        params['aspect'] = args.azimuth
    return params


def output_filename(args, slope_used, azimuth_used):
    """Match the committed export's Timeseries_<...>.csv naming convention."""
    span = (f"{args.startyear}_{args.endyear}"
            if args.endyear != args.startyear else str(args.startyear))
    return (f"Timeseries_{args.lat:g}_{args.lon:g}_{args.pvtechchoice}"
            f"_{args.peakpower:g}kWp_{args.loss:g}pct"
            f"_{slope_used}deg_{azimuth_used}deg_{span}.csv")


def parse_used_angles(text):
    """Read back the slope/azimuth PVGIS actually used from the CSV header."""
    slope = azimuth = 'NA'
    for line in text.splitlines():
        low = line.lower()
        if low.startswith('slope:'):
            slope = line.split(':', 1)[1].strip().split()[0]
        elif low.startswith('azimuth:'):
            azimuth = line.split(':', 1)[1].strip().split()[0]
        elif line.startswith('time,'):
            break
    return slope, azimuth


def fetch(args):
    params = build_params(args)
    print(f"Requesting PVGIS seriescalc for lat={args.lat}, lon={args.lon}, "
          f"{args.startyear}-{args.endyear} ...")
    resp = requests.get(API_URL, params=params, timeout=180)
    if resp.status_code != 200:
        raise SystemExit(
            f"PVGIS returned HTTP {resp.status_code}:\n{resp.text[:500]}")
    text = resp.text
    if 'time,P,' not in text:
        raise SystemExit(f"Unexpected PVGIS response (no data block):\n{text[:500]}")

    slope_used, azimuth_used = parse_used_angles(text)
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, output_filename(args, slope_used, azimuth_used))
    # PVGIS emits CRLF line endings; write bytes verbatim so the on-disk file is
    # identical to a manual browser download.
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        f.write(text)
    print(f"Wrote {out_path}")
    print(f"  slope used: {slope_used} deg, azimuth used: {azimuth_used} deg")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description='Fetch a PVGIS hourly PV timeseries and (optionally) build '
                    'the weekly PV table a location reads '
                    '(locations/<key>/pv_weekly.csv).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--lat', type=float, default=DEFAULTS['lat'],
                   help='Latitude (decimal degrees)')
    p.add_argument('--lon', type=float, default=DEFAULTS['lon'],
                   help='Longitude (decimal degrees)')
    p.add_argument('--peakpower', type=float, default=DEFAULTS['peakpower'],
                   help='Nominal PV system power (kWp)')
    p.add_argument('--loss', type=float, default=DEFAULTS['loss'],
                   help='System losses (%%)')
    p.add_argument('--pvtechchoice', default=DEFAULTS['pvtechchoice'],
                   choices=['crystSi', 'CIS', 'CdTe', 'Unknown'],
                   help='PV module technology')
    p.add_argument('--mountingplace', default=DEFAULTS['mountingplace'],
                   choices=['free', 'building'],
                   help="Mounting: 'free' (free-standing) or 'building' (roof)")
    p.add_argument('--slope', type=float, default=DEFAULTS['slope'],
                   help='Tilt angle (deg); omit to let PVGIS optimise it')
    p.add_argument('--azimuth', type=float, default=DEFAULTS['azimuth'],
                   help='Azimuth (deg, 0=south); omit to let PVGIS optimise it')
    p.add_argument('--startyear', type=int, default=DEFAULTS['startyear'])
    p.add_argument('--endyear', type=int, default=DEFAULTS['endyear'])
    p.add_argument('--location', default=None,
                   help='Named location key from sim_config.LOCATIONS. Sets '
                        '--lat/--lon from the catalog and (with --build) writes '
                        'the location cache locations/<key>/pv_weekly.csv that '
                        'the simulation reads.')
    p.add_argument('--build', action='store_true',
                   help='After fetching, build the weekly PV table '
                        '(locations/<key>/pv_weekly.csv when --location is given, '
                        'otherwise the default location). Uses --pv-year '
                        '(default = --startyear).')
    p.add_argument('--pv-year', type=str, default=None,
                   help='Year to keep when rebuilding (default: --startyear)')
    args = p.parse_args()

    # A named location pins lat/lon (and the output cache filename) from the
    # shared sim_config catalog, so PV and load stay bound to the same place.
    location_out = None
    if args.location:
        import sys
        sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
        import sim_config as cfg
        loc = cfg.get_location(args.location)
        args.lat, args.lon = loc['lat'], loc['lon']
        location_out = cfg.pv_source_path(args.location)
        print(f"Location '{args.location}': lat={args.lat}, lon={args.lon} "
              f"(region {loc['region']})")

    raw_path = fetch(args)

    if args.build:
        # Imported lazily so a plain fetch has no pandas/numpy dependency.
        import build_pv_source_internal as builder
        pv_year = args.pv_year or str(args.startyear)
        out_path = location_out or builder.OUTPUT_PATH
        print(f"Rebuilding {os.path.basename(out_path)} (PV_YEAR={pv_year}) ...")
        builder.build(raw_path=raw_path, output_path=out_path, pv_year=pv_year)


if __name__ == '__main__':
    main()
