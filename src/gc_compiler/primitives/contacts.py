"""Generic contact / via constructs built from layer roles and rules."""

from __future__ import annotations

import math

import gdsfactory as gf

from gc_compiler.primitives.geometry import Box, cut_positions, draw
from gc_compiler.tech.rules import CUT_BETWEEN, ROUTING_ROLES, LayerRole, Technology


def place_cuts(
    component: gf.Component,
    tech: Technology,
    cut_role: LayerRole,
    encloser_role: LayerRole,
    region: Box,
    axis: str = "x",
) -> Box:
    """Fill ``region`` with the largest cut array that keeps ``encloser_role`` enclosure.

    ``region`` is the shape of the enclosing layer (e.g. the diffusion node area).
    Returns the bounding box of the cuts that were drawn.
    """
    rules = tech.rules
    enc = rules.enc(encloser_role, cut_role)
    size = rules.cut(cut_role)
    # cuts are drawn centred: every centre on the grid needs an even pitch in grid steps, and a
    # centre on the grid (a region an odd number of steps wide is centred half a step off)
    step = 2 * rules.grid
    pitch = math.ceil(rules.cut_pitch(cut_role) / step - 1e-9) * step
    cx, cy = region.center
    along = cx if axis == "x" else cy
    shift = abs(rules.snap(along) - along)
    cx, cy = (rules.snap(cx), cy) if axis == "x" else (cx, rules.snap(cy))

    available = rules.snap_down((region.width if axis == "x" else region.height) - 2 * enc - 2 * shift)
    offsets = cut_positions(available, size, pitch)
    if not offsets:
        raise ValueError(
            f"{region} is too small for a {cut_role.value} cut with {enc} enclosure by "
            f"{encloser_role.value}"
        )
    boxes = [
        Box.centered(cx + o, cy, size, size) if axis == "x" else Box.centered(cx, cy + o, size, size)
        for o in offsets
    ]
    for box in boxes:
        draw(component, tech.layer(cut_role), box)
    return Box.hull(boxes)


def metal_pad(
    tech: Technology,
    cut_hull: Box,
    metal_role: LayerRole,
    cut_role: LayerRole,
    axis: str | None = None,
) -> Box:
    """Grow ``cut_hull`` to a legal pad on ``metal_role``.

    ``axis`` names the direction the pad is elongated in; the two short sides then only
    need the relaxed enclosure.
    """
    rules = tech.rules
    enc = rules.enc(metal_role, cut_role)
    relaxed = rules.enc_relaxed(metal_role, cut_role)
    dx, dy = (enc, enc) if axis is None else ((enc, relaxed) if axis == "x" else (relaxed, enc))
    pad = cut_hull.grow(dx, dy)
    min_w = rules.w(metal_role)
    cx, cy = pad.center
    return Box.centered(cx, cy, max(pad.width, min_w), max(pad.height, min_w))


def via_stack(
    component: gf.Component,
    tech: Technology,
    bottom: LayerRole,
    top: LayerRole,
    center: tuple[float, float],
    draw_bottom_pad: bool = True,
    axis: str | None = None,
) -> dict[LayerRole, Box]:
    """Stack single vias from ``bottom`` to ``top`` routing layer at ``center``.

    Returns the pad box drawn on each routing layer.
    """
    order = list(ROUTING_ROLES)
    i0, i1 = order.index(bottom), order.index(top)
    if i1 <= i0:
        raise ValueError(f"{top.value} must be above {bottom.value}")

    # a cut is drawn centred, so its centre must be on the grid (a centre worked out from a shape an
    # odd number of grid steps wide is half a step off it)
    cx, cy = tech.rules.snap(center[0]), tech.rules.snap(center[1])
    pads: dict[LayerRole, Box] = {}
    for lower, upper in zip(order[i0:i1], order[i0 + 1 : i1 + 1]):
        cut_role = CUT_BETWEEN[(lower, upper)]
        size = tech.rules.cut(cut_role)
        cut = Box.centered(cx, cy, size, size)
        draw(component, tech.layer(cut_role), cut)
        for role in (lower, upper):
            pad = metal_pad(tech, cut, role, cut_role, axis=axis)
            pads[role] = Box.hull([pads[role], pad]) if role in pads else pad

    # A stack starting on the local interconnect usually sits on a contact's own shape there, which
    # the caller drew (``draw_bottom_pad=False``). That shape is sized for the contact: where the
    # via above is the larger cut (SG13G2: Via1 0.19 on Cont 0.16; sky130's are both 0.17) it
    # would hang out of it, so the pad enclosing the via is drawn after all.
    stack = tech.stack
    if not draw_bottom_pad and bottom is stack.local_interconnect and i1 > i0:
        first_cut = CUT_BETWEEN[(order[i0], order[i0 + 1])]
        if tech.rules.cut(first_cut) > tech.rules.cut(stack.contact) + 1e-9:
            draw_bottom_pad = True
    # (a pad between the ends may be under its layer's minimum area: that is the merged shape's
    # rule, fixed on the cell by primitives.wells.fix_min_area)
    for role, pad in pads.items():
        if role is bottom and not draw_bottom_pad:
            continue
        draw(component, tech.layer(role), pad)
    return pads


def min_pad(tech: Technology, box: Box, role: LayerRole, axis: str = "y") -> Box:
    """``box`` grown to ``role``'s minimum width both ways and then along ``axis`` to its minimum
    area, from its own edges (so a box on the grid stays on it, whatever its centre)."""
    rules = tech.rules
    w = rules.w(role)

    def grow(lo: float, hi: float, size: float) -> tuple[float, float]:
        extra = rules.snap_up(size - (hi - lo)) if size > hi - lo + 1e-9 else 0.0
        left = rules.snap_down(extra / 2)
        return lo - left, hi + (extra - left)

    x0, x1 = grow(box.x0, box.x1, w)
    y0, y1 = grow(box.y0, box.y1, w)
    area = rules.area(role)
    if (x1 - x0) * (y1 - y0) < area - 1e-9:
        if axis == "y":
            y0, y1 = grow(y0, y1, area / (x1 - x0))
        else:
            x0, x1 = grow(x0, x1, area / (y1 - y0))
    return Box(x0, y0, x1, y1)
