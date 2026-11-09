"""
build_location.py -- materialise one location's simulation inputs.

sim_config.LOCATIONS configures 63 locations; only a handful have data on
disk. The rest need two files built:

    locations/<Country>/<key>/load_weekly_peak_min.csv   copied from the
        ENTSO-E zone library in locations/load_zones/ -- instant and offline
    locations/<Country>/<key>/pv_weekly.csv              fetched from the
        PVGIS API at the location's optimum tilt/azimuth -- needs the network

sim_config.ensure_location() already does both, and stays the single
implementation of them; this is a thin front end that exists for two reasons.

First, a stable argv the desktop app can launch as a normal job phase. The GUI
cannot call ensure_location() in-process (importing sim_config pulls numpy into
a Qt process that deliberately avoids it), and `python -c` gives no --help, no
argument contract, and a raw traceback on failure.

Second -- and the actual point -- the two steps fail for reasons that need
different fixes, and a caller has to be able to tell them apart:

    exit 0   the location is ready (re-verified on disk, not merely assumed)
    exit 2   PVGIS could not be reached, or refused -- a network problem;
             retrying later is the fix
    exit 3   there is no ENTSO-E zone file for this location's region -- no
             amount of retrying helps; the file has to be added
    exit 4   --check only: not built (nothing was changed)

Usage:
    python build_location.py --location lisbon
    python build_location.py --location lisbon --check
"""
from __future__ import annotations

import argparse
import os
import sys

import sim_config as cfg

EXIT_OK = 0
EXIT_NETWORK = 2
EXIT_NO_ZONE_FILE = 3
EXIT_NOT_BUILT = 4


def report(key: str) -> tuple[bool, bool]:
    """(load file present, PV file present) for a location."""
    return (os.path.exists(cfg.region_load_path(key)),
            os.path.exists(cfg.pv_source_path(key)))


def build(key: str) -> int:
    loc = cfg.get_location(key)
    have_load, have_pv = report(key)
    if have_load and have_pv:
        print(f"{key}: already built ({cfg.location_dir(key)})")
        return EXIT_OK

    # Check the zone file BEFORE any network call: a location nothing can
    # build should not first spend three minutes talking to PVGIS.
    zone = cfg.zone_library_path(loc['region'])
    if not have_load and not os.path.exists(zone):
        print(f"ERROR: no ENTSO-E load-zone file for {key!r} (region "
              f"{loc['region']!r}).\n"
              f"  Expected: {zone}\n"
              f"  This is not a network problem -- the file has to be added to "
              f"the zone library. See entsoe/build_yearly_load.py, which is "
              f"what produced the others.", file=sys.stderr)
        return EXIT_NO_ZONE_FILE

    print(f"Building {key} ({loc['country']}, region {loc['region']}, "
          f"lat {loc['lat']}, lon {loc['lon']}) ...")
    if not have_pv:
        print("  PV profile is missing and will be fetched from PVGIS "
              "(this needs an internet connection and can take a minute).")

    try:
        cfg.ensure_location(key)
    except FileNotFoundError as exc:
        # ensure_location raises this for a missing zone file; the check above
        # covers the normal path, this covers a race or a half-built folder.
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_NO_ZONE_FILE
    except Exception as exc:
        # The PVGIS fetch is a subprocess with check=True, so anything that
        # goes wrong out there -- no route to host, DNS failure, a 5xx, a
        # truncated response -- surfaces as CalledProcessError here. Report it
        # as the network problem it almost always is, but print the underlying
        # error rather than guessing on the user's behalf.
        print(f"ERROR: could not build {key!r}: {type(exc).__name__}: {exc}\n"
              f"  The PV profile comes from the PVGIS web API "
              f"(re.jrc.ec.europa.eu). Check the internet connection and try "
              f"again; the load data, if it was copied, is kept.", file=sys.stderr)
        return EXIT_NETWORK

    # Trust the disk, not the return: a killed or truncated fetch can leave
    # ensure_location() looking like it succeeded.
    have_load, have_pv = report(key)
    if not (have_load and have_pv):
        missing = [name for name, ok in
                   (("load_weekly_peak_min.csv", have_load), ("pv_weekly.csv", have_pv))
                   if not ok]
        print(f"ERROR: {key} still missing {', '.join(missing)} after the build "
              f"reported success.", file=sys.stderr)
        return EXIT_NETWORK

    print(f"{key}: ready ({cfg.location_dir(key)})")
    return EXIT_OK


def main() -> int:
    p = argparse.ArgumentParser(
        description="Materialise a location's load and PV inputs "
                    "(locations/<Country>/<key>/).")
    p.add_argument('--location', required=True, choices=sorted(cfg.LOCATIONS),
                   metavar='KEY',
                   help='Location key from sim_config.LOCATIONS')
    p.add_argument('--check', action='store_true',
                   help='Report whether it is built and change nothing '
                        '(exit 0 = ready, 4 = not built)')
    args = p.parse_args()

    have_load, have_pv = report(args.location)
    if args.check:
        state = 'ready' if (have_load and have_pv) else 'not built'
        print(f"{args.location}: {state} "
              f"(load={'yes' if have_load else 'no'}, "
              f"pv={'yes' if have_pv else 'no'})")
        return EXIT_OK if (have_load and have_pv) else EXIT_NOT_BUILT

    return build(args.location)


if __name__ == '__main__':
    sys.exit(main())
