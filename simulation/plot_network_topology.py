"""
plot_network_topology.py -- render the IEEE European LV feeder's bus/line
layout (net.bus['geo'] / net.line, the network's built-in geodata), encoding
two independent things so neither is lost to the other:

  - phase (A/B/C, the single-phase tap a load/EV/PV connects to) -> color
  - role (substation / household load / EV charger / PV generator / other)
    -> marker shape

Placement is not hardcoded: it comes from network_setup.setup_simulation(),
the exact function every real simulation run calls, so the figure always
matches whatever EV_CONTRIBUTION/PV_CONTRIBUTION/seed the rest of the
pipeline is actually using (sim_config.py defaults, or the CLI overrides
below) instead of drifting out of sync with it.

build_topology_figure() also hands back a TopologyIndex: where every marker
was actually drawn and which single-phase device sits under it. That exists
for interactive consumers (the desktop app's Network Map tab) which have to
answer "which bus did the user just click, and where is its data on disk"
without re-deriving any of the placement or sizing done here.
"""
import argparse
import colorsys
import json
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandapower.networks as nw
import pandapower.plotting as pp_plot

import network_setup as ns
import sim_config as cfg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'working_doc', 'Latex_thesis', 'figures')

plt.rcParams['font.family'] = ['Segoe UI', 'DejaVu Sans', 'sans-serif']

# Phase -> color (red/green/blue reads as the conventional A/B/C mapping in
# power-systems diagrams) and role -> marker shape, so a reader never has to
# hold both encodings in one channel. Validated CVD-distinct as a trio
# (scripts/validate_palette.js "#e34948,#008300,#2a78d6" --pairs all).
PHASE_COLOR = {'a': '#e34948', 'b': '#008300', 'c': '#2a78d6'}
PHASE_LABEL = {'a': 'Phase A', 'b': 'Phase B', 'c': 'Phase C'}
SUBSTATION_COLOR = '#333333'
PLAIN_COLOR = '#c9c8c2'
LINE_COLOR = '#d5d3cb'

CHART_SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'

SHAPE = {'substation': 'rect', 'load': 'circle', 'ev': 'poly3', 'pv': 'poly4', 'plain': 'circle'}
# matplotlib marker equivalents, for legend proxies only (shape identity, not fill).
LEGEND_MARKER = {'substation': 's', 'load': 'o', 'ev': '^', 'pv': 'D', 'plain': 'o'}

# Each role's halo circumradius as a multiple of base_size -- the largest
# layer drawn per marker, so this is what _max_overlap_free_base_size needs
# to keep any two marked buses from touching. Fill/rim sizes stay a fixed
# fraction of the halo (see _add_role), so rescaling base_size scales the
# whole marker as one unit.
ROLE_HALO_MULT = {'load': 1.25, 'pv': 1.85, 'ev': 1.9, 'substation': 2.4}
# Plain buses are drawn at base_size * 0.6 with no halo (see the collection
# built for plain_buses below); this is their click target, kept equal to what
# is actually on screen so a click counts if and only if it landed on the dot.
PLAIN_HALO_MULT = 0.6

# Base legend text sizes. Named because the display layout scales them
# (_fit_right_column) and has to know what 1.0 means.
LEGEND_FONTSIZE = 9.5
LEGEND_TITLE_FONTSIZE = 10


@dataclass(frozen=True)
class BusDevice:
    """One single-phase tap at a bus.

    Nothing in this feeder is a three-phase device: every household load, EV
    charger and PV generator connects to exactly ONE of a/b/c. So `phase` is
    not "the phase we happen to be looking at", it is the only phase on which
    this device exists at all, and a consumer must not synthesise the other
    two -- there is no data behind them.

    `sgen_index` is the net.asymmetric_sgen row index, which is the identity
    the per-inverter result files are keyed on: network_setup.export_results()
    writes one `inverter_<sgen index>` column per active generator into
    inverter_profiles/global_{p,q,v}_phase_<PHASE>.csv. A household load has
    no per-bus record anywhere in a results tree (only the network-wide
    per-phase min/max envelope), so its element_index is the asymmetric_load
    row and is good for labelling and nothing else -- hence sgen_index is
    None for it, rather than an index that would look loadable and isn't.
    """
    kind: str            # 'load' | 'ev' | 'pv'
    phase: str           # 'a' | 'b' | 'c'
    element_index: int   # asymmetric_sgen row for ev/pv, asymmetric_load row for load

    @property
    def sgen_index(self):
        return self.element_index if self.kind in ('ev', 'pv') else None


@dataclass(frozen=True)
class MarkedBus:
    """One bus, and where its marker landed.

    Named for what it originally held -- only the buses carrying a device --
    but it now covers every bus in the network, role 'plain' included, because
    those have a recorded voltage too (all_bus_voltages_phase_*.csv) and an
    unindexed bus is an unclickable one.

    A bus can host several devices at once: the EV and PV placements run off
    independent seed streams and neither excludes the other, so one bus may
    carry a load, an EV and a PV -- or two EVs -- each a logically separate
    single-phase device with its own column on disk. In *this* feeder they
    always end up on the same phase, because _place_single_phase_sgens() picks
    from the phases that bus's loads energize and every bus here has exactly
    one (single-phase) load; nothing in this structure assumes that, though,
    so read each device's own `phase` rather than the bus's.

    `role` is only which marker ended up on top (the zorder order load -> pv
    -> ev -> substation), i.e. what a click actually lands on; `devices` is
    the full truth.
    """
    bus: int
    x: float
    y: float
    role: str            # 'substation' | 'ev' | 'pv' | 'load'
    devices: tuple       # BusDevice, one per single-phase tap at this bus
    hit_radius: float    # data-coordinate radius of the drawn marker's halo

    @property
    def inverters(self):
        """The devices that have a per-inverter trace on disk, EV/PV only."""
        return tuple(d for d in self.devices if d.sgen_index is not None)


@dataclass(frozen=True)
class TopologyIndex:
    """Enough to hit-test a click on the figure without recomputing any
    pandapower internals.

    Holds every bus in the network, not just the ones with a device on them.

    Coordinates are read AFTER _dejitter_coincident_buses has run, so they are
    where the markers were actually drawn rather than where the model's
    geodata originally put them -- for this feeder's ~40 coincident pairs
    those are different points, and the pre-dejitter value would mis-locate
    both members of every pair. That pass only moves *marked* buses, which is
    why plain ones are read straight from the model: they were drawn there.
    """
    buses: tuple
    base_size: float

    def nearest(self, x, y):
        """The bus whose drawn marker covers (x, y) in data coordinates, or
        None when the click landed on bare canvas.

        Proximity rather than matplotlib's picker: the markers are
        PatchCollections built inside pandapower's create_bus_collection, so
        picker=True would mean threading an argument through someone else's
        API and then de-duplicating one pick event per layer (halo, rim, fill)
        per role. The radius here is the marker's own drawn circumradius --
        base_size * ROLE_HALO_MULT[role], or PLAIN_HALO_MULT for the small
        undecorated dots -- so a click counts if and only if it landed on ink.

        A device marker wins over a plain dot whose target overlaps it, mirroring
        the zorder they were drawn in: the role markers sit on top, so that is
        the one the click visually landed on. Among equals, nearest wins.
        """
        best, best_key = None, None
        for marked in self.buses:
            distance2 = (marked.x - x) ** 2 + (marked.y - y) ** 2
            if distance2 > marked.hit_radius ** 2:
                continue
            key = (marked.role == 'plain', distance2)
            if best_key is None or key < best_key:
                best, best_key = marked, key
        return best

    def by_bus(self, bus):
        return next((m for m in self.buses if m.bus == bus), None)


def _darken(hex_color, factor=0.72):
    """A darker step of the same hue, for the thin rim drawn just inside each
    marker's white halo -- polish, not the separation mechanism (the halo
    already does that)."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, l * factor, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _lighten(hex_color, amount=0.82):
    """Mix toward white -- a soft wash for the count table's header cells, so
    the phase color reads as identity without the cell competing with the
    data ink (text stays in ink, per the color-formula rule)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _load_taps(net):
    """(load index, bus, phase) for every asymmetric load. Each load in this
    network is non-zero on exactly one of p_a/p_b/p_c (55 genuinely
    single-phase loads) -- the phase with the largest |P| is that load's tap.

    The load index is carried along because a marked bus has to be able to say
    *which* load sits on it, not merely that one does."""
    p = net.asymmetric_load[['p_a_mw', 'p_b_mw', 'p_c_mw']].abs().to_numpy()
    phase_of_load = np.array(['a', 'b', 'c'])[p.argmax(axis=1)]
    return list(zip(net.asymmetric_load.index, net.asymmetric_load['bus'], phase_of_load))


def _loads_by_phase(load_taps):
    buses_by_phase = {ph: [] for ph in 'abc'}
    for _index, bus, phase in load_taps:
        buses_by_phase[phase].append(bus)
    return buses_by_phase


def _sgens_by_phase(net, sgen_indices, phases):
    buses_by_phase = {ph: [] for ph in 'abc'}
    for idx, phase in zip(sgen_indices, phases):
        buses_by_phase[phase].append(net.asymmetric_sgen.at[idx, 'bus'])
    return buses_by_phase


def _bus_xy(net, buses):
    coords = net.bus.loc[buses, 'geo'].apply(lambda s: json.loads(s)['coordinates'])
    return np.array(coords.tolist())


def _dejitter_coincident_buses(net, buses, offset_frac=0.9):
    """This feeder model has ~40 pairs of buses sitting at literally
    identical geo coordinates (a characteristic of the underlying OpenDSS
    model -- stacked connection points, not a data error). No amount of
    marker shrinking separates two markers centered on the same point, so
    where that coincidence lands on two *marked* buses, nudge each onto a
    small ring around the shared point instead. Mutates net.bus['geo']
    in place -- safe here since this script owns a throwaway net purely for
    visualization, never fed back into a simulation."""
    coords = net.bus.loc[list(buses), 'geo'].apply(lambda s: tuple(json.loads(s)['coordinates']))
    groups = {}
    for b, pt in coords.items():
        groups.setdefault(pt, []).append(b)
    base = pp_plot.get_collection_sizes(net, bus_size=1.0)['bus']
    r = base * offset_frac
    for pt, group in groups.items():
        if len(group) < 2:
            continue
        for i, b in enumerate(group):
            angle = 2 * np.pi * i / len(group)
            x, y = pt[0] + r * np.cos(angle), pt[1] + r * np.sin(angle)
            net.bus.at[b, 'geo'] = json.dumps({'coordinates': [x, y], 'type': 'Point'})


def _max_overlap_free_base_size(net, halo_mult_of_bus, margin=0.88):
    """The largest base_size for which no two marked buses' halos can touch,
    given each bus's own halo-circumradius multiple of base_size (role ->
    ROLE_HALO_MULT). Exact per-pair, not a single worst-case-role guess: for
    every pair (i, j) the limit is dist(i, j) / (mult_i + mult_j); the
    smallest such limit over all pairs is the binding one. `margin` < 1
    leaves a visible gap rather than merely stopping at first contact."""
    buses = list(halo_mult_of_bus)
    if len(buses) < 2:
        return None
    xy = _bus_xy(net, buses)
    mult = np.array([halo_mult_of_bus[b] for b in buses])
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    mult_sum = mult[:, None] + mult[None, :]
    np.fill_diagonal(dist, np.inf)
    return float((dist / mult_sum).min()) * margin


def _add_role(collections, net, buses_by_phase, shape, size, halo_size, zorder):
    """White halo (drawn first, keeps adjacent/overlapping buses from blurring
    into each other) -> a thin darker-shade rim (polish, not the separation
    mechanism -- the halo already does that) -> the true-color fill on top,
    one rim+fill pair per phase present."""
    any_buses = [b for buses in buses_by_phase.values() for b in buses]
    if not any_buses:
        return
    collections.append(pp_plot.create_bus_collection(
        net, buses=any_buses, size=halo_size, patch_type=shape, color='white', zorder=zorder))
    for phase, buses in buses_by_phase.items():
        if buses:
            collections.append(pp_plot.create_bus_collection(
                net, buses=buses, size=size * 1.12, patch_type=shape, color=_darken(PHASE_COLOR[phase]),
                zorder=zorder + 0.1))
            collections.append(pp_plot.create_bus_collection(
                net, buses=buses, size=size, patch_type=shape, color=PHASE_COLOR[phase], zorder=zorder + 0.2))


def _set_legend_font_scale(legend, scale):
    """Resize one legend's text, title included, as a multiple of the base
    sizes. Idempotent, because _fit_right_column re-runs on every resize and
    must not compound its own shrinking."""
    legend.get_title().set_fontsize(LEGEND_TITLE_FONTSIZE * scale)
    for text in legend.get_texts():
        text.set_fontsize(LEGEND_FONTSIZE * scale)


def _fit_right_column(fig, phase_legend, role_legend, x_left, y_top,
                      gap=0.03, right_limit=0.985, min_scale=0.66):
    """Keep the two guides inside the figure, at whatever size it is *now*.

    Only the embedded (GUI) figure needs this. The saved figure goes through
    bbox_inches='tight' + bbox_extra_artists, which grows the canvas around
    anything the legends overhang, so the thesis PNG never clips however wide
    they get. A canvas in a Qt splitter has no such freedom: it is exactly as
    wide as the user last dragged it.

    The root cause is that legend text is a fixed point size while the figure
    is not, so a legend taking 14 % of the thesis figure's width takes 28 % of
    a canvas half as wide. The side-by-side anchor (RIGHT_COL_X + phase width
    + gap) is computed once, in fractions, and every resize invalidates it.

    Both guides stay on the top row in every arrangement below, which is the
    point: the obvious fix -- stacking the role legend under the phase legend
    -- buys width by spending height the column does not have, and drops the
    role legend straight onto the count table underneath. Whatever happens
    here happens within that one row.

      side-by-side           the thesis figure's arrangement, while it fits
      side-by-side compact   same, text scaled down until it fits
      side-by-side shifted   also moved left, into the map's whitespace flank

    Returns the arrangement chosen, which is what the tests assert on.
    """
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()

    def extent(legend):
        """(x0, y0, x1, y1) of a legend in figure fractions."""
        box = legend.get_window_extent(renderer)
        x0, y0 = inverse.transform((box.x0, box.y0))
        x1, y1 = inverse.transform((box.x1, box.y1))
        return x0, y0, x1, y1

    def place(left):
        """Anchor both guides along the top row starting at `left`; return the
        right edge the pair reaches."""
        phase_legend.set_bbox_to_anchor((left, y_top), transform=fig.transFigure)
        phase_right = extent(phase_legend)[2]
        role_legend.set_bbox_to_anchor((phase_right + gap, y_top), transform=fig.transFigure)
        return extent(role_legend)[2]

    for legend in (phase_legend, role_legend):
        _set_legend_font_scale(legend, 1.0)
    role_right = place(x_left)
    if role_right <= right_limit:
        return 'side-by-side'

    # Shrink by the ratio the overflow implies. Legend width is dominated by
    # its text but not only text -- the gap between the guides, and the marker
    # handles, do not scale with font size -- so the first ratio lands short
    # and the width bottoms out entirely somewhere below scale 0.72. Hence a
    # bounded corrective loop that gives up rather than spins: converges in
    # two or three passes when it can converge at all.
    scale = 1.0
    for _ in range(4):
        used = role_right - x_left
        if used <= 0:
            break
        scale = max(min_scale, scale * (right_limit - x_left) / used)
        for legend in (phase_legend, role_legend):
            _set_legend_font_scale(legend, scale)
        role_right = place(x_left)
        if role_right <= right_limit:
            return 'side-by-side compact'
        if scale <= min_scale:
            break

    # The pair is wider than the column even at the smallest text it will
    # take. Move it left by the residual excess: overlapping the network
    # drawing is recoverable (that flank is mostly whitespace, and the toolbar
    # pans), a legend cut off mid-word is not.
    place(max(0.0, x_left - (role_right - right_limit)))
    return 'side-by-side shifted'


# Figure-fraction layout: the network gets a fixed rectangle of its own
# (matplotlib's aspect='equal' then centers the data within it on whichever
# axis has slack) instead of sharing the full figure with side content that
# only exists at axes-fraction x > 1 -- which left the network looking
# shoved into the left third once bbox_inches='tight' cropped around
# whatever the legends/table extended out to on the right.
NETWORK_RECT = [0.02, 0.03, 0.60, 0.84]  # left, bottom, width, height
RIGHT_COL_X = 0.66


def build_topology_figure(contribution=cfg.EV_CONTRIBUTION, pv_contribution=cfg.PV_CONTRIBUTION,
                           seed=cfg.NETWORK_SEED, figsize=(15, 10.5), for_display=False):
    """Place EVs/PVs the same way a real run would and return
    (fig, ax, net, TopologyIndex).

    `for_display` re-lays-out the right column for a figure that will be shown
    in a resizable canvas instead of saved -- see _fit_right_column(). It
    changes nothing about the network, the placement or the counts, and the
    default (False) path is byte-for-byte the figure the thesis uses.
    """
    net = nw.ieee_european_lv_asymmetric()
    net, ev_sgen_indices, ev_phases, pv_sgen_indices, pv_phases, *_ = ns.setup_simulation(
        net, seed=seed, contribution=contribution, pv_contribution=pv_contribution)

    load_taps = _load_taps(net)
    load_by_phase = _loads_by_phase(load_taps)
    ev_by_phase = _sgens_by_phase(net, ev_sgen_indices, ev_phases)
    pv_by_phase = _sgens_by_phase(net, pv_sgen_indices, pv_phases)

    load_buses = {b for buses in load_by_phase.values() for b in buses}
    ev_buses = {b for buses in ev_by_phase.values() for b in buses}
    pv_buses = {b for buses in pv_by_phase.values() for b in buses}
    source_buses = set(net.trafo['hv_bus']) | set(net.ext_grid['bus'])
    plain_buses = sorted(set(net.bus.index) - load_buses - ev_buses - pv_buses - source_buses)

    # Every EV/PV bus in this network is also a load bus (chargers/generators
    # are only ever placed on buses_with_loads), so drawing the load's circle
    # there too just sits a round halo under a triangle/diamond -- the circle
    # silhouette doesn't fit inside the triangle/diamond silhouette even when
    # its nominal size is smaller, so an arc of it always peeks out past the
    # straight edges. The table below still counts every load regardless;
    # this only thins what actually gets drawn on the map.
    load_draw_by_phase = {ph: [b for b in buses if b not in ev_buses and b not in pv_buses]
                          for ph, buses in load_by_phase.items()}

    load_draw_buses = {b for buses in load_draw_by_phase.values() for b in buses}
    marked_buses = load_draw_buses | ev_buses | pv_buses | source_buses

    # A handful of buses in this feeder model sit at literally identical geo
    # coordinates (see _dejitter_coincident_buses); resolve those before any
    # size math, since no marker size avoids two markers centered on the same
    # point, and an unresolved coincidence would otherwise pull the whole
    # overlap-free cap below down to the pair's true distance of zero.
    _dejitter_coincident_buses(net, marked_buses)

    # base_size: pandapower's own network-extent heuristic, capped by the
    # largest size that still guarantees no two marked buses' halos touch
    # (computed exactly from real bus geodata, not eyeballed) -- so shrinking
    # never goes further than the actual closest marked pair requires.
    halo_mult_of_bus = {b: ROLE_HALO_MULT['load'] for b in load_draw_buses}
    for b in pv_buses:
        halo_mult_of_bus[b] = ROLE_HALO_MULT['pv']
    for b in ev_buses:
        halo_mult_of_bus[b] = ROLE_HALO_MULT['ev']
    for b in source_buses:
        halo_mult_of_bus[b] = ROLE_HALO_MULT['substation']
    pp_default_size = pp_plot.get_collection_sizes(net, bus_size=1.0)['bus']
    overlap_free_size = _max_overlap_free_base_size(net, halo_mult_of_bus)
    base_size = min(pp_default_size, overlap_free_size) if overlap_free_size is not None else pp_default_size

    # The click-hit-testing index, built here rather than earlier because both
    # preceding steps change its answers: the dejitter moved every coincident
    # bus off its model coordinate, and base_size fixes how large a target each
    # marker really presents. Roles are assigned in the same order the markers
    # are drawn below, so role_of_bus ends up naming whichever marker is on
    # top -- the one a click visually lands on.
    # Plain int/str throughout, never the numpy scalars pandas hands back:
    # this structure crosses into a GUI process that json-serializes and
    # string-formats it, and a np.str_ phase or np.int64 index survives both
    # just well enough to go unnoticed until something downstream refuses it.
    devices_by_bus = {}
    for index, bus, phase in load_taps:
        devices_by_bus.setdefault(int(bus), []).append(
            BusDevice('load', str(phase), int(index)))
    for kind, sgen_indices, phases in (('pv', pv_sgen_indices, pv_phases),
                                       ('ev', ev_sgen_indices, ev_phases)):
        for index, phase in zip(sgen_indices, phases):
            bus = int(net.asymmetric_sgen.at[index, 'bus'])
            devices_by_bus.setdefault(bus, []).append(
                BusDevice(kind, str(phase), int(index)))

    role_of_bus = {int(b): 'load' for b in load_draw_buses}
    role_of_bus.update({int(b): 'pv' for b in pv_buses})
    role_of_bus.update({int(b): 'ev' for b in ev_buses})
    role_of_bus.update({int(b): 'substation' for b in source_buses})
    # Every bus in the network, not only the ones carrying a device. The plain
    # ones are drawn (small grey dots, PLAIN_COLOR) and they do have a recorded
    # voltage -- export_results writes global_voltages/all_bus_voltages_phase_*
    # with one column per bus -- so leaving them out of the index would make
    # 851 of this feeder's 907 buses unclickable for no reason other than that
    # nothing is plugged into them.
    role_of_bus.update({int(b): 'plain' for b in plain_buses
                        if int(b) not in role_of_bus})
    indexed = []
    for bus in sorted(int(b) for b in set(net.bus.index)):
        role = role_of_bus.get(bus, 'plain')
        x, y = json.loads(net.bus.at[bus, 'geo'])['coordinates']
        indexed.append(MarkedBus(
            bus=bus, x=float(x), y=float(y), role=role,
            devices=tuple(devices_by_bus.get(bus, ())),
            hit_radius=base_size * ROLE_HALO_MULT.get(role, PLAIN_HALO_MULT)))
    topology_index = TopologyIndex(buses=tuple(indexed), base_size=base_size)

    # Lines first (background), then roles load -> pv -> ev -> substation,
    # each zorder block strictly above the last, so a bus that is both a load
    # and an EV/PV site still shows its (more specific) EV/PV marker on top.
    collections = [
        pp_plot.create_line_collection(net, lines=net.line.index, color=LINE_COLOR, linewidths=1.0, zorder=1),
    ]
    if plain_buses:
        collections.append(pp_plot.create_bus_collection(
            net, buses=plain_buses, size=base_size * 0.6, color=PLAIN_COLOR, zorder=2))

    _add_role(collections, net, load_draw_by_phase, SHAPE['load'], base_size, base_size * ROLE_HALO_MULT['load'], zorder=3)
    _add_role(collections, net, pv_by_phase, SHAPE['pv'], base_size * 1.5, base_size * ROLE_HALO_MULT['pv'], zorder=4)
    _add_role(collections, net, ev_by_phase, SHAPE['ev'], base_size * 1.5, base_size * ROLE_HALO_MULT['ev'], zorder=5)
    if source_buses:
        # No separate white halo here: create_trafo_collection's circles are
        # unfilled outlines (the chart surface already shows through), so
        # they need no legibility backing the way a solid-fill role marker does.
        # create_trafo_collection draws the classic two-winding-transformer
        # symbol (two overlapping circles) *on the hv_bus-lv_bus connecting
        # line itself* -- both the icon and the link into the LV network in
        # one call, styled in our ink instead of pandapower's default black.
        # Sized off pp_default_size, not the (possibly much smaller)
        # overlap-safe base_size -- there is exactly one of these, nothing
        # else nearby to collide with, and at the shrunk size the two circles
        # overlapped into what read as a single ring.
        collections.append(pp_plot.create_trafo_collection(
            net, trafos=net.trafo.index, color=SUBSTATION_COLOR, linewidth=2.0,
            size=pp_default_size * 1.3, zorder=6.1))
        # The external grid symbol (hatched bar + stub) marks that the MV
        # side continues beyond this substation into a larger grid, instead
        # of implying this transformer is the network's true, unconnected
        # root.
        collections.append(pp_plot.create_ext_grid_collection(
            net, ext_grid_buses=net.ext_grid.bus.values, size=pp_default_size * 2.0, zorder=6.2))

    fig = plt.figure(figsize=figsize)
    fig.set_facecolor(CHART_SURFACE)
    ax = fig.add_axes(NETWORK_RECT)
    ax.set_facecolor(CHART_SURFACE)
    pp_plot.draw_collections(collections, ax=ax, plot_colorbars=False)
    ax.set_aspect('equal')
    ax.axis('off')

    n_load, n_ev, n_pv = len(net.asymmetric_load), len(ev_sgen_indices), len(pv_sgen_indices)
    neutral_fill, neutral_edge = '#a8a7a1', _darken('#a8a7a1', 0.75)
    role_handles = [
        Line2D([], [], marker='o', linestyle='', markersize=11,
               markerfacecolor='none', markeredgecolor=SUBSTATION_COLOR, markeredgewidth=1.6,
               label='Substation (MV/LV transformer)'),
        Patch(facecolor=CHART_SURFACE, edgecolor=INK_PRIMARY, hatch='xxxx', linewidth=1.2,
             label='External grid (MV network continues upstream)'),
        Line2D([], [], marker=LEGEND_MARKER['load'], linestyle='', markersize=9,
               markerfacecolor=neutral_fill, markeredgecolor=neutral_edge, markeredgewidth=1.1,
               label=f'Household load ({n_load})'),
        Line2D([], [], marker=LEGEND_MARKER['ev'], linestyle='', markersize=11,
               markerfacecolor=neutral_fill, markeredgecolor=neutral_edge, markeredgewidth=1.1,
               label=f'EV charger ({n_ev})'),
        Line2D([], [], marker=LEGEND_MARKER['pv'], linestyle='', markersize=10,
               markerfacecolor=neutral_fill, markeredgecolor=neutral_edge, markeredgewidth=1.1,
               label=f'PV generator ({n_pv})'),
        Line2D([], [], marker=LEGEND_MARKER['plain'], linestyle='', markersize=6,
               markerfacecolor=PLAIN_COLOR, markeredgecolor='none', label='Other LV bus'),
    ]
    phase_handles = [
        Line2D([], [], marker='o', linestyle='', markersize=10, markerfacecolor=PHASE_COLOR[ph],
               markeredgecolor=_darken(PHASE_COLOR[ph]), markeredgewidth=1.1, label=PHASE_LABEL[ph])
        for ph in 'abc'
    ]

    legend_style = dict(loc='upper left', frameon=False, fontsize=LEGEND_FONTSIZE,
                        title_fontsize=LEGEND_TITLE_FONTSIZE,
                        labelspacing=0.9, handletextpad=0.7, borderaxespad=0, bbox_transform=fig.transFigure)
    # Both guides side by side along the same top row of the right column,
    # not stacked, so the rest of that column is free for the count table.
    # Figure-fraction anchors (not axes-fraction) keep them at a fixed canvas
    # position regardless of where the network's own data happens to sit.
    phase_legend = ax.legend(handles=phase_handles, title='Phase  ·  color',
                             bbox_to_anchor=(RIGHT_COL_X, 0.90), **legend_style)
    ax.add_artist(phase_legend)
    if for_display:
        # Anchored provisionally; _fit_right_column() below decides where it
        # actually goes, and re-decides every time the canvas is resized.
        role_legend = ax.legend(handles=role_handles, title='Role  ·  shape',
                                bbox_to_anchor=(RIGHT_COL_X, 0.90), **legend_style)
    else:
        fig.canvas.draw()
        phase_legend_width_fig = fig.transFigure.inverted().transform(
            (phase_legend.get_window_extent().x1, 0))[0] - RIGHT_COL_X
        role_legend = ax.legend(handles=role_handles, title='Role  ·  shape',
                                bbox_to_anchor=(RIGHT_COL_X + phase_legend_width_fig + 0.03, 0.90), **legend_style)
    for legend in (role_legend, phase_legend):
        legend.get_title().set_color(INK_SECONDARY)
        legend.get_title().set_fontweight('bold')
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    # Devices-per-phase count table -- one row per role that actually has a
    # phase (the substation ties all three and the plain buses have none, so
    # neither gets a row), plus a row for buses where an EV charger and a PV
    # generator share the same bus AND phase (the two are placed independently,
    # so this is a real overlap count, not derived from the rows above), one
    # column per phase. Pinned to the bottom-right corner -- both guides now
    # share the top row, so the table doesn't need to dodge them.
    both_by_phase = {ph: set(ev_by_phase[ph]) & set(pv_by_phase[ph]) for ph in 'abc'}
    table_rows = [
        ('Household load', load_by_phase),
        ('EV charger', ev_by_phase),
        ('PV generator', pv_by_phase),
        ('EV + PV, same bus', both_by_phase),
    ]
    cell_text = [[str(len(counts[ph])) for ph in 'abc'] for _, counts in table_rows]

    table_rect = [RIGHT_COL_X, 0.06, 0.30, 0.30]  # left, bottom, width, height, figure fraction
    table_caption = fig.text(table_rect[0], table_rect[1] + table_rect[3] + 0.025, 'Devices per phase',
                             fontsize=10, fontweight='bold', color=INK_SECONDARY, ha='left', va='bottom')
    table_ax = fig.add_axes(table_rect)
    table_ax.axis('off')
    count_table = table_ax.table(
        cellText=cell_text, rowLabels=[label for label, _ in table_rows],
        colLabels=[PHASE_LABEL[ph] for ph in 'abc'], cellLoc='center', bbox=(0, 0, 1, 1))
    count_table.auto_set_font_size(False)
    count_table.set_fontsize(9.5)
    for (row, col), cell in count_table.get_celld().items():
        cell.set_edgecolor(INK_MUTED)
        cell.set_linewidth(0.6)
        if row == 0:  # phase header row
            phase = 'abc'[col]
            cell.set_facecolor(_lighten(PHASE_COLOR[phase]))
            cell.get_text().set_color(INK_PRIMARY)
            cell.get_text().set_fontweight('bold')
        elif col == -1:  # row-label column
            cell.set_facecolor(CHART_SURFACE)
            cell.get_text().set_color(INK_SECONDARY)
            cell.get_text().set_ha('left')
        else:
            cell.set_facecolor(CHART_SURFACE)
            cell.get_text().set_color(INK_PRIMARY)

    # Figure-level (not axes-level) title + subtitle, sitting in the margin
    # above NETWORK_RECT's fixed top (0.87) -- pinned by figure fraction, not
    # by the axes' data box, so they can never collide with the network
    # drawing or with each other regardless of what the data looks like.
    n_bus = len(net.bus)
    title_artist = fig.suptitle("IEEE European LV feeder", x=0.02, y=0.97, ha='left', fontsize=18,
                                fontweight='bold', color=INK_PRIMARY)
    subtitle_artist = fig.text(
        0.02, 0.925, f"{n_bus} buses  ·  EV {contribution * 100:.0f}% / PV {pv_contribution * 100:.0f}% "
        f"of loads  ·  placement seed {seed}", ha='left', va='top', fontsize=11, color=INK_SECONDARY)
    # Kept for savefig's tight-bbox pass as a safety margin in case any text
    # still nudges past the figure edge (e.g. a very long legend label).
    fig.legend_artists_for_save = [role_legend, phase_legend, title_artist, subtitle_artist,
                                   table_caption, count_table]

    if for_display:
        fig.canvas.draw()          # a renderer has to exist before anything can be measured
        fig.display_arrangement = _fit_right_column(fig, phase_legend, role_legend,
                                                    RIGHT_COL_X, 0.90)

        def _refit(_event, fig=fig, phase_legend=phase_legend, role_legend=role_legend):
            fig.display_arrangement = _fit_right_column(fig, phase_legend, role_legend,
                                                        RIGHT_COL_X, 0.90)

        # Held on the figure, not dropped as a local: mpl_connect stores only a
        # weak reference to a bound method, and a callback nothing else owns is
        # collected at the first GC -- the resize handler would then simply
        # stop firing, silently, some arbitrary time later.
        fig.display_refit_callback = _refit
        fig.display_refit_cid = fig.canvas.mpl_connect('resize_event', _refit)

    return fig, ax, net, topology_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the LV network topology with EV/PV placement, "
                    "using the same setup_simulation() placement the "
                    "simulations themselves use.")
    parser.add_argument('--ev-contribution', type=float, default=cfg.EV_CONTRIBUTION,
                        help=f'Fraction of loads fitted with an EV charger (default from sim_config.py: {cfg.EV_CONTRIBUTION})')
    parser.add_argument('--pv-contribution', type=float, default=cfg.PV_CONTRIBUTION,
                        help=f'Fraction of loads fitted with a PV generator (default from sim_config.py: {cfg.PV_CONTRIBUTION})')
    parser.add_argument('--seed', type=int, default=cfg.NETWORK_SEED,
                        help=f'Placement seed, the same one setup_simulation() uses for a real run (default: {cfg.NETWORK_SEED})')
    parser.add_argument('--output', default=None,
                        help='Output image path (default: working_doc/Latex_thesis/figures/'
                             'network_topology_ev<PCT>_pv<PCT>.png)')
    parser.add_argument('--show', action='store_true', help='Also open an interactive window')
    args = parser.parse_args()

    fig, ax, net, topology = build_topology_figure(
        args.ev_contribution, args.pv_contribution, args.seed)

    output = args.output or os.path.join(
        THESIS_FIGURES_DIR,
        f"network_topology_ev{int(args.ev_contribution * 100)}_pv{int(args.pv_contribution * 100)}.png")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight', bbox_extra_artists=fig.legend_artists_for_save)
    print(f"Saved {output}")
    print(f"{len(topology.buses)} marked buses, "
          f"{sum(len(m.inverters) for m in topology.buses)} EV/PV inverters")

    if args.show:
        plt.show()
