"""Power pins for a hard-macro flow: metal4 stripes over the macro, one set per supply net, each
tied down to that net's own rails by via stacks where the layers in between leave room.

A place-and-route flow connects its power grid to a macro's pins on an upper metal; this macro's
supplies are rails on metal1 and metal2 under three layers of wiring, so the stripes are placed
where via stacks can actually reach them. Where that is, is computed from the layout: a via
stack's centre is allowed where it lies on the net's rail and its pads on every layer it passes
keep their spacing to everything drawn there (the pads, grown by the spacing, must miss every
shape; equivalently the centre must miss every shape grown by spacing plus half the pad).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gdsfactory as gf
import klayout.db as kdb

from gc_compiler.primitives.geometry import Box, draw, add_port
from gc_compiler.tech.rules import LayerRole, Technology

#: The supplies' label texts, the rails' layers, and the macro's pin names they become.
SUPPLY_LABELS = ("vdd", "vss", "vnw")
@dataclass(frozen=True)
class PowerStripe:
    net: str
    x: float
    sites: tuple[float, ...]
    """y of each via stack under the stripe."""


def _dbu(component: gf.Component) -> float:
    return component.kcl.layout.dbu


def _region(component: gf.Component, layer: tuple[int, int]) -> kdb.Region:
    index = component.kcl.layout.find_layer(*layer)
    if index is None:
        return kdb.Region()
    return kdb.Region(component.begin_shapes_rec(index)).merged()


def _labels(component: gf.Component, tech: Technology, role: LayerRole) -> list[tuple[str, kdb.Point]]:
    layer = tech.label_layer(role)
    index = component.kcl.layout.find_layer(*layer) if layer else None
    if index is None:
        return []
    out = []
    it = component.begin_shapes_rec(index)
    while not it.at_end():
        shape = it.shape()
        if shape.is_text() and shape.text.string in SUPPLY_LABELS:
            out.append((shape.text.string, (it.trans() * shape.text.trans).disp.to_p()))
        it.next()
    return out


def _grow(region: kdb.Region, dx: float, dy: float, dbu: float) -> kdb.Region:
    nx, ny = round(dx / dbu), round(dy / dbu)
    return region.minkowski_sum(kdb.Box(-nx, -ny, nx, ny)).merged()


@dataclass(frozen=True)
class _Stack:
    """A via stack from ``base`` to metal4: its cuts and the pad (width, height) on each metal."""

    base: LayerRole
    axis: str
    cuts: tuple[tuple[LayerRole, float], ...]
    pads: tuple[tuple[LayerRole, float, float], ...]


def _stack(tech: Technology, base: LayerRole, axis: str) -> _Stack:
    """The base pad lies along the rail (``axis``), so its short sides need only the relaxed
    enclosure; the pads above are square, metal3's lengthened along the stripe to its minimum
    area. Metal4's pad is inside the stripe."""
    rules, stack = tech.rules, tech.stack
    order = list(stack.routing)
    layers = order[order.index(base) : order.index(stack.power_pin_layer) + 1]
    cuts = [(stack.cut_between(lo, hi), rules.cut(stack.cut_between(lo, hi))) for lo, hi in zip(layers, layers[1:])]
    pads = []
    for i, role in enumerate(layers[:-1]):
        width = height = 0.0
        for cut, size in cuts[max(0, i - 1) : i + 1]:
            if role is base:
                along, across = rules.enc(role, cut), rules.enc_relaxed(role, cut)
                w, h = (size + 2 * along, size + 2 * across) if axis == "x" else (size + 2 * across, size + 2 * along)
            else:
                w = h = size + 2 * rules.enc(role, cut)
            width, height = max(width, w, rules.w(role)), max(height, h, rules.w(role))
        area = rules.min_area.get(role, 0.0)
        if width * height < area and role is not base:  # the base pad is part of the rail
            # an even number of grid steps, so the pad's edges stay on grid about its centre
            step = 2 * rules.grid
            height = math.ceil(area / width / step - 1e-9) * step
        pads.append((role, width, height))
    return _Stack(base, axis, tuple(cuts), tuple(pads))


def net_rails(component: gf.Component, tech: Technology, bank_x: float | None = None) -> dict[str, dict[LayerRole, kdb.Region]]:
    """Each supply net's rails: the metal1/metal2 polygons its labels sit on. With ``bank_x`` (a
    banked macro) the n-well is one net per bank, ``vnw_0`` left of it and ``vnw_1`` right."""
    dbu = _dbu(component)
    rails: dict[str, dict[LayerRole, kdb.Region]] = {}
    for role in (LayerRole.M1, LayerRole.M2):
        metal = _region(component, tech.layer(role))
        for text, point in _labels(component, tech, role):
            net = text
            if text == "vnw" and bank_x is not None:
                net = f"vnw_{int(point.x * dbu >= bank_x)}"
            probe = kdb.Region(kdb.Box(point.x - 1, point.y - 1, point.x + 1, point.y + 1))
            hit = metal.interacting(probe)
            rails.setdefault(net, {}).setdefault(role, kdb.Region())
            rails[net][role] += hit
    for layers in rails.values():
        for role in layers:
            layers[role] = layers[role].merged()
    return rails


def allowed_sites(
    component: gf.Component, tech: Technology, rails: dict[LayerRole, kdb.Region]
) -> list[tuple[_Stack, kdb.Region]]:
    """Where a via stack from one net's rails up to metal4 may be centred, per stack shape: its
    base pad inside the rail (so it adds no new edge there), and every pad and cut above clear of
    whatever is drawn on its layer."""
    dbu = _dbu(component)
    rules = tech.rules
    out = []
    for base, rail in rails.items():
        for axis in ("x", "y"):
            stack = _stack(tech, base, axis)
            (_role, bw, bh), *above = stack.pads
            ok = rail.sized(-round(bw / 2 / dbu), -round(bh / 2 / dbu), 2)
            for role, w, h in above:
                shapes = _region(component, tech.layer(role))
                ok -= _grow(shapes, rules.s(role) + w / 2, rules.s(role) + h / 2, dbu)
            for cut, size in stack.cuts:
                existing = _region(component, tech.layer(cut))
                d = rules.cut_space.get(cut, 0.0) + size / 2
                ok -= _grow(existing, d, d, dbu)
            out.append((stack, ok.merged()))
    return out


def _draw_stack(component: gf.Component, tech: Technology, stack: _Stack, x: float, y: float) -> None:
    for cut, size in stack.cuts:
        draw(component, tech.layer(cut), Box.centered(x, y, size, size))
    for role, w, h in stack.pads:
        draw(component, tech.layer(role), Box.centered(x, y, w, h))


def plan_stripes(
    component: gf.Component, tech: Technology, bank_x: float | None = None
) -> list[PowerStripe]:
    """Where each supply net's stripes go: greedily, the x with the most via sites under a
    full-height stripe first, then others a stripe pitch away, clear of the other nets'."""
    dbu = _dbu(component)
    bounds = component.dbbox()
    rails = net_rails(component, tech, bank_x)
    policy = tech.policy.power_stripe
    want = max(2, round(bounds.width() / policy.pitch))
    stripes: list[PowerStripe] = []
    # the nets with the fewest sites choose first
    per_stack = {net: allowed_sites(component, tech, layers) for net, layers in rails.items()}
    sites = {net: kdb.Region() for net in per_stack}
    for net, options in per_stack.items():
        for _stack_shape, region in options:
            sites[net] += region
        sites[net] = sites[net].merged()
    for net in sorted(sites, key=lambda n: sites[n].area()):
        region = sites[net]
        if region.is_empty():
            raise ValueError(f"no room for a via stack down to any {net} rail")
        candidates = []
        step = policy.scan_step
        x = bounds.left
        while x <= bounds.right:
            line = region & kdb.Region(kdb.Box(round(x / dbu), region.bbox().bottom, round(x / dbu) + 1, region.bbox().top))
            ys = []
            for polygon in line.each():
                box = polygon.bbox()
                lo, hi = box.bottom * dbu, box.top * dbu
                y = (lo + hi) / 2
                if not ys or y - ys[-1] >= policy.site_pitch:
                    ys.append(y)
            if ys:
                candidates.append((len(ys), x, tuple(sorted(ys))))
            x += step
        candidates.sort(key=lambda c: (-c[0], c[1]))
        mine: list[PowerStripe] = []
        for _count, x, ys in candidates:
            if any(abs(x - s.x) < policy.width + policy.clearance for s in stripes):
                continue
            if any(abs(x - s.x) < policy.pitch for s in mine):
                continue
            mine.append(PowerStripe(net, tech.rules.snap(x), ys))
            if len(mine) == want:
                break
        if not mine:
            raise ValueError(f"no room for a {net} stripe clear of the other supplies'")
        stripes.extend(mine)
    return stripes


def add_power_pins(
    component: gf.Component, tech: Technology, bank_x: float | None = None
) -> list[PowerStripe]:
    """Draw the stripes and their via stacks, and name each stripe a metal4 pin of its net."""
    stripes = plan_stripes(component, tech, bank_x)
    options = {net: allowed_sites(component, tech, layers) for net, layers in net_rails(component, tech, bank_x).items()}
    bounds = component.dbbox()
    dbu = _dbu(component)
    pin_role = tech.stack.power_pin_layer
    width = tech.policy.power_stripe.width
    m4 = tech.layer(pin_role)
    label = tech.label_layer(pin_role)
    for stripe in stripes:
        half = width / 2
        draw(component, m4, Box(stripe.x - half, bounds.bottom, stripe.x + half, bounds.top))
        for y in stripe.sites:
            y = tech.rules.snap(y)
            point = kdb.Region(kdb.Box(round(stripe.x / dbu), round(y / dbu), round(stripe.x / dbu) + 1, round(y / dbu) + 1))
            stack = next(st for st, region in options[stripe.net] if not region.interacting(point).is_empty())
            _draw_stack(component, tech, stack, stripe.x, y)
        centre = (stripe.x, (bounds.bottom + bounds.top) / 2)
        add_port(component,
            name=f"{stripe.net}_m4_{stripes.index(stripe)}", center=centre, orientation=90,
            width=width, layer=m4, port_type="electrical",
        )
        if label is not None:
            component.add_label(text=stripe.net, position=centre, layer=label)
    return stripes
