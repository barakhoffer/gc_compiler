"""Vertical vss/n-well strap column.

A bitcell's own ``vss`` rail and a row-pair's n-well merge *horizontally* across the
columns of one row by abutment, but nothing ties a row to its neighbours, or to the tap
rows that carry the actual p+/n+ contacts, *vertically*. This column closes that loop: it
runs two vertical metal2 straps the full height of the array, with a metal1 bridge at
every row's ``vss`` rail, one new n-well tap per row-pair (the row-pairs' n-wells have
never had a contact of their own), and a bridge into every tap row's own rails.

Word lines are horizontal poly stripes spanning the whole array, so the column also
carries plain pass-through poly at every row's ``wwl``/``rwl`` height. With word-line straps
it also ties every word line to its metal3 strap: a contacted poly landing, stacked up to
metal2, a short metal2 jog to the strap's track, and via2.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

import gdsfactory as gf

from gc_compiler.cache import cached_cell
from gc_compiler.primitives.wells import fix_min_area
from gc_compiler.primitives.contacts import min_pad, place_cuts, via_stack
from gc_compiler.primitives.devices import gate_pad_size
from gc_compiler.primitives.geometry import Box, draw, add_port
from gc_compiler.tech.rules import DeviceKind, LayerRole, Technology


@dataclass(frozen=True)
class StrapColumnGeometry:
    width: float
    vss_track_x: float
    vnw_track_x: float
    tap_width: float
    poly_landing_x: tuple[float, float]
    """Two x slots, left of the n-well tap, for word-line landing pads (alternating, so landings
    on neighbouring stripes keep their spacing)."""
    right_landing_x: float
    """A landing slot right of the tap, for the one word line a tap may cut (see
    :func:`strap_column_segment`)."""


def _stack_pads(tech: Technology, top: LayerRole) -> dict[LayerRole, float]:
    """Each layer's pad width in a via stack from the local interconnect up to ``top`` (the bottom
    pad only where the stack draws one: see :func:`~gc_compiler.primitives.contacts.via_stack`)."""
    rules, stack = tech.rules, tech.stack
    layers = list(stack.routing[: stack.routing.index(top) + 1])
    pads = {}
    for i, role in enumerate(layers):
        cuts = [stack.cut_between(a, b) for a, b in zip(layers, layers[1:])][max(0, i - 1) : i + 1]
        if role is stack.local_interconnect and rules.cut(cuts[-1]) <= rules.cut(stack.contact) + 1e-9:
            continue  # sits on the contact's own shape
        pads[role] = max(max(rules.via_pad(role, c) for c in cuts), rules.w(role))
    return pads


def _stack_clearance(
    tech: Technology, top_a: LayerRole, top_b: LayerRole, below: LayerRole | None = None
) -> float:
    """How far apart two via stacks from the local interconnect (up to ``top_a`` and ``top_b``) must
    be centred: each shared layer's two pads, and each shared cut, at their spacing (only the layers
    under ``below``, if given: what runs off at different heights above does not face)."""
    rules = tech.rules
    a, b = _stack_pads(tech, top_a), _stack_pads(tech, top_b)
    if below is not None:
        routing = tech.stack.routing
        a = {r: v for r, v in a.items() if routing.index(r) < routing.index(below)}
        b = {r: v for r, v in b.items() if routing.index(r) < routing.index(below)}
    need = [a[r] / 2 + b[r] / 2 + rules.s(r) for r in a.keys() & b.keys()]
    stack = tech.stack
    for low, high in zip(stack.routing, stack.routing[1:]):
        if high in a and high in b:
            cut = stack.cut_between(low, high)
            need.append(rules.cut(cut) + rules.cut_space[cut])
    return max(need)


def strap_column_geometry(tech: Technology) -> StrapColumnGeometry:
    rules = tech.rules
    # A little slack over the bare minimum keeps this off the floating-point edge of the
    # enclosure rule (width - 2*enc would otherwise round down to just under the cut size).
    tap_w = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON) + 2 * rules.grid
    via_pad = rules.cut(LayerRole.VIA1) + 2 * rules.enc(LayerRole.M2, LayerRole.VIA1)
    pad_size = gate_pad_size(tech) + 2 * rules.grid

    vss_x = via_pad / 2 + rules.grid
    # npc (poly-cut marker, where the process has one) grows past the pad itself, so the two
    # landing slots must clear by npc's own width/spacing, not the narrower poly pad; without it,
    # by the pad's and poly's
    if tech.has(LayerRole.POLY_CUT_MARKER):
        npc_half = (
            max(
                rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.POLY_CUT_MARKER, LayerRole.LICON),
                rules.w(LayerRole.POLY_CUT_MARKER),
            )
            / 2
        )
        npc_space = rules.s(LayerRole.POLY_CUT_MARKER)
    else:
        npc_half, npc_space = pad_size / 2, rules.s(LayerRole.POLY)
    # from a landing's centre to diffusion -- the tap, or the neighbouring bitcell's, which can
    # sit right at the column's edge: its contact and its pad keep their spacing to diffusion
    to_diff = rules.snap(
        max(rules.cut(LayerRole.LICON) / 2 + rules.poly_contact_to_diff, pad_size / 2 + rules.poly_to_diff)
        + rules.grid
    )
    land_a = vss_x + via_pad / 2 + rules.s(LayerRole.M2) + pad_size / 2
    # two landings side by side, each tied up to its own strap on metal3 (the straps run at
    # different heights; the stacks' pads below them face each other)
    land_b = land_a + max(
        2 * npc_half + npc_space, _stack_clearance(tech, LayerRole.M3, LayerRole.M3, below=LayerRole.M3)
    )
    # the n-tap's via stack (li up to metal2) sits beside the landing's (up to the strap on metal3)
    vnw_x = land_b + max(to_diff + tap_w / 2, _stack_clearance(tech, LayerRole.M3, LayerRole.M2))
    # the right landing's li (a contact and its enclosure) beside the tap's (its via pad, if drawn)
    li = tech.stack.local_interconnect
    landing_li = max(
        rules.cut(LayerRole.LICON) + 2 * rules.enc(li, LayerRole.LICON),
        _stack_pads(tech, LayerRole.M3).get(li, 0.0),  # a strap tie's via pad, where it is drawn
    )
    tap_li = max(_stack_pads(tech, LayerRole.M2).get(li, 0.0), rules.cut(LayerRole.LICON), rules.w(li))
    land_r = vnw_x + max(tap_w / 2 + to_diff, landing_li / 2 + tap_li / 2 + rules.s(li))
    # past the right landing: diffusion spacing, and a whole npc spacing -- the column can end
    # against a word-line driver, whose own gate contacts' npc sits near its edge
    width = land_r + max(to_diff, npc_half + npc_space)
    return StrapColumnGeometry(
        width=rules.snap(width),
        vss_track_x=rules.snap(vss_x),
        vnw_track_x=rules.snap(vnw_x),
        tap_width=tap_w,
        poly_landing_x=(rules.snap(land_a), rules.snap(land_b)),
        right_landing_x=rules.snap(land_r),
    )


def strap_column_segment(
    tech: Technology,
    geo: StrapColumnGeometry,
    height: float,
    *,
    vss_bridge_y: tuple[float, ...] = (),
    nwell_taps: tuple[tuple[float, float], ...] = (),
    tap_row_bridges: tuple[tuple[float, float], ...] = (),
    tap_row_nwell: tuple[float, float] | None = None,
    poly_y: tuple[float, ...] = (),
    poly_width: float = 0.15,
    poly_widths: tuple[float, ...] = (),
    strap_y: tuple[float, ...] = (),
    name: str | None = None,
) -> gf.Component:
    """One vertical slice of the strap column, spanning ``height``.

    Args:
        vss_bridge_y: absolute-in-segment y of every ``vss`` rail to bridge (pure metal).
        nwell_taps: (y0, y1) n-well band per row-pair; a fresh n+ tap is added at its
            centre, since no bitcell n-well has a contact of its own.
        tap_row_bridges: (y, "vss"|"vnw") bridges into a tap row's own rail.
        tap_row_nwell: (y0, y1) of a tap row's own n-well band, bridged through so the
            tap row's n-well does not get a gap where this column crosses it.
        poly_y: y of every word-line pass-through stripe.
        poly_widths: gate length of each ``poly_y`` stripe (default ``poly_width`` for all), so the
            pass-through is as wide as the word line it continues.
        strap_y: the metal3 strap's y for each ``poly_y`` stripe; every stripe is then landed
            and tied to its strap.
    """
    rules = tech.rules
    seg_name = name or f"gc_strap_seg_{height:g}"

    def build(cell_name: str) -> gf.Component:
        return _build_strap_segment(
            tech,
            geo,
            height,
            cell_name,
            vss_bridge_y=vss_bridge_y,
            nwell_taps=nwell_taps,
            tap_row_bridges=tap_row_bridges,
            tap_row_nwell=tap_row_nwell,
            poly_y=poly_y,
            poly_width=poly_width,
            poly_widths=poly_widths,
            strap_y=strap_y,
        )

    return cached_cell(seg_name, build, tech)


def _build_strap_segment(
    tech: Technology,
    geo: StrapColumnGeometry,
    height: float,
    name: str,
    *,
    vss_bridge_y: tuple[float, ...],
    nwell_taps: tuple[tuple[float, float], ...],
    tap_row_bridges: tuple[tuple[float, float], ...],
    tap_row_nwell: tuple[float, float] | None,
    poly_y: tuple[float, ...],
    poly_width: float,
    poly_widths: tuple[float, ...] = (),
    strap_y: tuple[float, ...] = (),
) -> gf.Component:
    rules = tech.rules
    component = gf.Component(name)
    m2 = tech.layer(LayerRole.M2)

    nwell_ranges = list(nwell_taps) + ([tap_row_nwell] if tap_row_nwell else [])
    widths = [poly_widths[i] if poly_widths else poly_width for i in range(len(poly_y))]
    taps = [_place_tap(tech, band, poly_y, widths) for band in nwell_taps]
    cut = {index for _cy, index in taps if index is not None}
    tap_x0, tap_x1 = geo.vnw_track_x - geo.tap_width / 2, geo.vnw_track_x + geo.tap_width / 2
    landing_index = 0
    for index, y in enumerate(poly_y):
        width = widths[index]
        # A bare poly gate over n-well is a real sky130 varactor primitive; one contact
        # anywhere along the run turns it back into plain poly wiring. Word lines that
        # never cross a n-well band do not need one -- unless they are strapped.
        landed = strap_y or any(y0 <= y <= y1 for y0, y1 in nwell_ranges) or index in cut
        if index in cut:
            # poly over the tap would make it a varactor's plate: the stripe stops short of the
            # tap on both sides, each piece is landed, and li joins them round the tap
            space = rules.poly_to_diff + rules.grid
            draw(component, tech.layer(LayerRole.POLY), Box(0.0, y - width / 2, tap_x0 - space, y + width / 2))
            draw(component, tech.layer(LayerRole.POLY), Box(tap_x1 + space, y - width / 2, geo.width, y + width / 2))
        else:
            draw(component, tech.layer(LayerRole.POLY), Box(0.0, y - width / 2, geo.width, y + width / 2))
        if landed:
            cx = geo.poly_landing_x[landing_index % 2]
            landing_index += 1
            others = tuple(zip(poly_y, widths))
            pad_y = _poly_landing(component, tech, cx, y, width, height, others)
            if index in cut:
                tap_cy = next(cy for cy, i in taps if i == index)
                right_y = _poly_landing(component, tech, geo.right_landing_x, y, width, height, others)
                _li_bridge(component, tech, (cx, pad_y), (geo.right_landing_x, right_y), tap_cy)
            if strap_y:
                _strap_tie(component, tech, cx, pad_y, strap_y[index])

    for y in vss_bridge_y:
        _bridge(component, tech, geo.vss_track_x, y, geo.width)
        # a p-tap on every row's vss: the row pairs' n-well bands run the whole array's width, so
        # nothing else reaches the p-substrate strips between them within latch-up distance (LU.2)
        _ptap(component, tech, geo.vnw_track_x, y, geo.tap_width, poly_y, widths, nwell_ranges)
    for y, net in tap_row_bridges:
        track_x = geo.vss_track_x if net == "vss" else geo.vnw_track_x
        _bridge(component, tech, track_x, y, geo.width)
    if tap_row_nwell is not None:
        draw(component, tech.layer(LayerRole.NWELL), Box(0.0, tap_row_nwell[0], geo.width, tap_row_nwell[1]))
    for (y0, y1), (cy, _index) in zip(nwell_taps, taps):
        draw(component, tech.layer(LayerRole.NWELL), Box(0.0, y0, geo.width, y1))
        _tap(component, tech, geo.vnw_track_x, cy, geo.tap_width, DeviceKind.PMOS)

    draw(component, m2, Box(geo.vss_track_x - rules.w(LayerRole.M2) / 2, 0.0, geo.vss_track_x + rules.w(LayerRole.M2) / 2, height))
    draw(component, m2, Box(geo.vnw_track_x - rules.w(LayerRole.M2) / 2, 0.0, geo.vnw_track_x + rules.w(LayerRole.M2) / 2, height))

    for net, x in (("vss", geo.vss_track_x), ("vnw", geo.vnw_track_x)):
        for y, orientation in ((0.0, 270), (height, 90)):
            add_port(component,
                name=f"{net}_{'s' if orientation == 270 else 'n'}",
                center=(x, y),
                orientation=orientation,
                width=rules.w(LayerRole.M2),
                layer=m2,
                port_type="electrical",
            )
        label = tech.label_layer(LayerRole.M2)
        if label is not None:
            component.add_label(text=net, position=(x, height / 2), layer=label)

    component.info["width"] = geo.width
    component.info["height"] = height
    fix_min_area(component, tech)
    return component


def _tap_height(tech: Technology) -> float:
    """A tap's height: a contact and its enclosure, and tall enough (at the tap's width) for the
    tap layer's minimum area."""
    rules = tech.rules
    height = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON)
    width = strap_column_geometry(tech).tap_width
    step = 2 * rules.grid
    return max(height, math.ceil(rules.area(LayerRole.TAP) / width / step - 1e-9) * step)


def _place_tap(
    tech: Technology, band: tuple[float, float], poly_y: tuple[float, ...], widths: list[float]
) -> tuple[float, int | None]:
    """Where a row pair's n-well tap goes in its band ``(y0, y1)``, and which poly stripe (if
    any) has to be cut for it: the tap keeps the well's enclosure, and poly spacing from every
    stripe but at most one -- the stripes are the row pair's word lines, and two write word
    lines leave no room for a tap between or beside them. The centre-most place that needs the
    fewest cuts."""
    rules = tech.rules
    half = _tap_height(tech) / 2
    enc = rules.enc(LayerRole.NWELL, LayerRole.TAP)
    lo, hi = band[0] + enc + half, band[1] - enc - half
    if lo > hi:
        raise ValueError(f"the n-well band {band} is too short for a tap")
    best: tuple[int, float, float, int | None] | None = None
    steps = max(1, round((hi - lo) / rules.grid))
    for k in range(steps + 1):
        cy = rules.snap(lo + (hi - lo) * k / steps)
        crossing = [
            i for i, (y, w) in enumerate(zip(poly_y, widths))
            if y - w / 2 < cy + half + rules.poly_to_diff and y + w / 2 > cy - half - rules.poly_to_diff
        ]
        if len(crossing) > 1:
            continue
        score = (len(crossing), abs(cy - (band[0] + band[1]) / 2))
        if best is None or score < best[:2]:
            best = (*score, cy, crossing[0] if crossing else None)
    if best is None:
        raise ValueError(f"no place in the n-well band {band} where a tap crosses at most one word line")
    return best[2], best[3]


def _ptap(
    component: gf.Component,
    tech: Technology,
    cx: float,
    cy: float,
    width: float,
    poly_y: tuple[float, ...],
    widths: list[float],
    nwell_ranges: list[tuple[float, float]],
) -> None:
    """A p+ tap on the vss bridge at ``cy``, contacted up to the bridge's metal1 only (the vnw
    strap's metal2 runs over it). It keeps poly spacing from every word line and its distance
    from the n-well bands."""
    rules = tech.rules
    half = _tap_height(tech) / 2
    for y, w in zip(poly_y, widths):
        if abs(y - cy) < half + w / 2 + rules.poly_to_diff:
            raise ValueError(f"no room for a p-tap on the vss rail at y={cy}: word line at {y}")
    for y0, y1 in nwell_ranges:
        if y0 - rules.nwell_to_ptap_space < cy + half and cy - half < y1 + rules.nwell_to_ptap_space:
            raise ValueError(f"no room for a p-tap on the vss rail at y={cy}: n-well {y0}..{y1}")
    strip = Box(cx - width / 2, cy - half, cx + width / 2, cy + half)
    draw(component, tech.layer(LayerRole.TAP), strip)
    implant = tech.tap_implant_role(DeviceKind.NMOS)
    draw(component, tech.layer(implant), strip.grow(rules.enc(implant, LayerRole.TAP)))
    cut_region = Box(strip.x0, cy - rules.cut(LayerRole.LICON) / 2, strip.x1, cy + rules.cut(LayerRole.LICON) / 2)
    hull = place_cuts(component, tech, LayerRole.LICON, LayerRole.TAP, cut_region, axis="x")
    draw(component, tech.layer(LayerRole.LI), min_pad(tech, hull.grow(0.0, rules.enc(LayerRole.LI, LayerRole.LICON)), LayerRole.LI))
    via_stack(component, tech, LayerRole.LI, LayerRole.M1, (cx, cy), draw_bottom_pad=False)


def _li_bridge(
    component: gf.Component, tech: Technology, left: tuple[float, float], right: tuple[float, float], tap_cy: float
) -> None:
    """li from one landing to the other, round the tap's own li strip (on the side of it the
    landings are on): down or up from each landing, across, and back."""
    rules = tech.rules
    w = rules.w(LayerRole.LI)
    # each landing's li is a contact's width plus li's enclosure of it; the tap's the same, grown
    # to li's minimum width and area (see _tap)
    cut = rules.cut(LayerRole.LICON)
    contact_half = (cut + 2 * rules.enc(LayerRole.LI, LayerRole.LICON)) / 2
    tap_li = min_pad(tech, Box.centered(0.0, 0.0, cut, 2 * contact_half), LayerRole.LI)
    # a landing tied to a strap also carries its via pad on li where that is the wider (SG13G2)
    pad_half = max(contact_half, _stack_pads(tech, LayerRole.M3).get(LayerRole.LI, 0.0) / 2)
    clear = max(pad_half, tap_li.height / 2) + rules.s(LayerRole.LI) + w / 2 + rules.grid
    below = (left[1] + right[1]) / 2 <= tap_cy
    y = rules.snap(tap_cy - clear if below else tap_cy + clear)
    li = tech.layer(LayerRole.LI)
    # legs as wide as the landings they rise from, so no notch is left between a landing and
    # the bridge
    for x, pad_y in (left, right):
        draw(component, li, Box(x - pad_half, min(y, pad_y) - w / 2, x + pad_half, max(y, pad_y) + w / 2))
    draw(component, li, Box(left[0] - pad_half, y - w / 2, right[0] + pad_half, y + w / 2))


def _bridge(component: gf.Component, tech: Technology, track_x: float, y: float, width: float) -> None:
    """Pure-metal bridge from a horizontal row rail into the vertical strap."""
    rules = tech.rules
    pad_h = rules.cut(LayerRole.VIA1) + 2 * rules.enc(LayerRole.M1, LayerRole.VIA1)
    draw(component, tech.layer(LayerRole.M1), Box(0.0, y - pad_h / 2, width, y + pad_h / 2))
    via_stack(component, tech, LayerRole.M1, LayerRole.M2, (track_x, y), draw_bottom_pad=False)


def _poly_landing(
    component: gf.Component, tech: Technology, cx: float, cy: float, poly_width: float, height: float,
    others: tuple[tuple[float, float], ...] = (),
) -> float:
    """A licon-contacted poly pad on the stripe at ``cy``; returns the pad's centre y.

    Unstrapped, its only job is to stop magic from recognising the surrounding stripe as a bare
    gate; strapped, it is where the stripe meets its strap. The pad is widened beyond the stripe
    (matching an ordinary gate contact) so the licon keeps its poly enclosure. Segments are
    mirrored copies of each other, so a stripe near the segment's edge has another one as near
    beyond it (a read line is 0.22 um from the edge): the pad slides toward the segment's middle
    by as much as that one's spacing needs, still covering its own stripe.
    """
    rules = tech.rules
    # Same grid buffer as the n-well tap: keeps width - 2*enc from rounding down to
    # exactly the cut size.
    pad_size = max(gate_pad_size(tech), poly_width) + 2 * rules.grid
    clear = poly_width / 2 + rules.s(LayerRole.POLY) + pad_size / 2
    low, high = -cy + clear, 2 * height - cy - clear  # the mirrored neighbours below 0 and above height
    # the other stripes in the segment: the pad (wider than its own stripe) keeps poly spacing
    # from each, sliding along its stripe to where it does
    slack = (pad_size - poly_width) / 2
    lo, hi = max(cy - slack, low), min(cy + slack, high)
    blocked = [(y - (w / 2 + rules.s(LayerRole.POLY) + pad_size / 2), y + (w / 2 + rules.s(LayerRole.POLY) + pad_size / 2))
               for y, w in others if abs(y - cy) > 1e-9]
    candidates = [cy, lo, hi] + [b for a, b in blocked] + [a for a, b in blocked]
    free = [
        y for y in candidates
        if lo - 1e-9 <= y <= hi + 1e-9 and all(not (a + 1e-9 < y < b - 1e-9) for a, b in blocked)
    ]
    if not free:
        raise ValueError(f"no room for a poly landing on the stripe at y={cy}")
    pad_y = min(free, key=lambda y: abs(y - cy))
    pad_y = rules.snap(pad_y + (rules.grid if pad_y > cy else -rules.grid if pad_y < cy else 0.0))
    if abs(pad_y - cy) > slack + 1e-9:
        raise ValueError(f"no room for a poly landing on the stripe at y={cy}")
    pad = Box.centered(cx, pad_y, pad_size, pad_size)
    draw(component, tech.layer(LayerRole.POLY), pad)
    hull = place_cuts(component, tech, LayerRole.LICON, LayerRole.POLY, pad, axis="x")
    tech.decorate_gate_contact(component, hull.as_tuple())
    li = hull.grow(rules.enc(LayerRole.LI, LayerRole.LICON))
    draw(component, tech.layer(LayerRole.LI), li)
    return pad_y


def _strap_tie(component: gf.Component, tech: Technology, cx: float, cy: float, strap_y: float) -> None:
    """From a poly landing's li up to the metal3 strap at ``strap_y``: stacked to metal2 on the
    landing, a metal2 jog along x = ``cx``, and via2 on the strap."""
    rules = tech.rules
    via_stack(component, tech, LayerRole.LI, LayerRole.M2, (cx, cy), draw_bottom_pad=False)
    pad = rules.cut(LayerRole.VIA1) / 2 + rules.enc(LayerRole.M2, LayerRole.VIA1)
    draw(component, tech.layer(LayerRole.M2), Box(cx - pad, min(cy, strap_y) - pad, cx + pad, max(cy, strap_y) + pad))
    via_stack(component, tech, LayerRole.M2, LayerRole.M3, (cx, strap_y))


def _tap(
    component: gf.Component, tech: Technology, cx: float, cy: float, width: float, body_of: DeviceKind
) -> None:
    """A single contacted p+/n+ tap, stacked all the way up to metal2."""
    rules = tech.rules
    strip_h = _tap_height(tech)
    strip = Box(cx - width / 2, cy - strip_h / 2, cx + width / 2, cy + strip_h / 2)
    draw(component, tech.layer(LayerRole.TAP), strip)

    implant = tech.tap_implant_role(body_of)
    draw(component, tech.layer(implant), strip.grow(rules.enc(implant, LayerRole.TAP)))

    cut_region = Box(strip.x0, cy - rules.cut(LayerRole.LICON) / 2, strip.x1, cy + rules.cut(LayerRole.LICON) / 2)
    hull = place_cuts(component, tech, LayerRole.LICON, LayerRole.TAP, cut_region, axis="x")
    # li over the cut only, enclosing it above and below: a strip the tap's width would come
    # within li spacing of the landings either side of the tap
    draw(component, tech.layer(LayerRole.LI), min_pad(tech, hull.grow(0.0, rules.enc(LayerRole.LI, LayerRole.LICON)), LayerRole.LI))
    via_stack(component, tech, LayerRole.LI, LayerRole.M2, (cx, cy), draw_bottom_pad=False)
