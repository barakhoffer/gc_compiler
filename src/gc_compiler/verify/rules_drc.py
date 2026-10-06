"""A design-rule check generated from a technology's own :class:`~gc_compiler.tech.rules.DesignRules`.

The PDK's decks check the PDK's rules; this checks the rules the compiler was *given*, whatever
they are. That is what shows the layout follows its technology rather than numbers that happen to
suit sky130: every reference structure built with a second process (IHP SG13G2, :mod:`gc_compiler.tech.sg13g2`)
must pass this checker with that process's rules, as it passes with sky130's -- and, where the
PDKs are installed, both processes' own decks.

Every rule field is either checked here or listed in :data:`NOT_CHECKED` with the reason
(``tests/test_portability.py`` keeps the two lists complete). The checks, all with klayout.db on the
flattened, merged layout:

* ``grid.<layer>``: every vertex on the manufacturing grid;
* ``width.<layer>`` / ``space.<layer>`` / ``area.<layer>``: each layer's minimum width (projection
  metrics), spacing (euclidean, notches included) and area;
* ``enc.<outer>/<inner>``: each enclosure, on the ``inner`` shapes that touch ``outer`` -- they must
  lie inside it, by the full value on all sides or, where a relaxed value exists, by that on all
  sides and the full one on two opposite sides;
* ``cut.<layer>`` / ``cutspace.<layer>``: every cut a square of exactly its size, cuts spaced;
* the n and p implants never overlapping;
* device rules on the gates (poly over diffusion): endcap, diffusion extension, gate-to-contact,
  gate spacing, gate length and width, poly and poly contacts to diffusion, well to diffusion and
  tap, deep n-well to n-wells and p+ diffusion.

``leaf=True`` checks a cell meant to abut others: errors touching the cell's edge (halves of shapes
the neighbour completes: shared contacts, wells that carry on) are not counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import klayout.db as kdb

from gc_compiler.tech.rules import DeviceKind, LayerRole, Technology

#: Rule fields this checker does not check, and why (the rest are all checked).
NOT_CHECKED = {
    "isolation_ring_inner": "the ring's geometry is magic's nwell.5a/nwell.6 (sign-off deck)",
    "isolation_ring_outer": "likewise",
    "latchup_distance": "a distance to the nearest tap along any path: magic's LU rules",
}

@dataclass
class RulesResult:
    by_rule: dict[str, int] = field(default_factory=dict)
    markers: dict[str, list[kdb.Box]] = field(default_factory=dict)
    """A few error locations per rule, in um."""

    @property
    def total(self) -> int:
        return sum(self.by_rule.values())

    @property
    def clean(self) -> bool:
        return self.total == 0

    def __str__(self) -> str:
        if self.clean:
            return "rules clean"
        lines = [f"rule violations: {self.total}"]
        for rule, count in sorted(self.by_rule.items()):
            where = ", ".join(f"({b.center().x:.3f}, {b.center().y:.3f})" for b in self.markers.get(rule, [])[:3])
            lines.append(f"  {rule}: {count}  at {where}")
        return "\n".join(lines)


class _Checker:
    def __init__(self, layout: kdb.Layout, top: kdb.Cell, tech: Technology, leaf: bool, tile=None):
        self.tech, self.rules, self.leaf = tech, tech.rules, leaf
        self.dbu = layout.dbu
        self.bbox = top.bbox() if tile is None else kdb.DBox(*tile).to_itype(layout.dbu)
        self.layers: dict[LayerRole, kdb.Region] = {}
        for role in LayerRole:
            if not tech.has(role):
                continue
            index = layout.find_layer(*tech.layer(role))
            region = kdb.Region(top.begin_shapes_rec(index)) if index is not None else kdb.Region()
            self.layers[role] = region.merged()
        self.result = RulesResult()

    def region(self, role: LayerRole) -> kdb.Region:
        return self.layers.get(role, kdb.Region())

    def d(self, um: float) -> int:
        return round(um / self.dbu)

    def report(self, rule: str, errors, roles: tuple[LayerRole, ...] = ()) -> None:
        boxes = [e.bbox() for e in errors.each()] if hasattr(errors, "each") else []
        if self.leaf:
            # a leaf cell's shapes at its edge are halves its neighbour completes (a shared contact,
            # a well that carries on): what touches the edge is the assembled layout's to check
            inner = self.bbox.enlarged(-self.d(0.001), -self.d(0.001))
            boxes = [b for b in boxes if inner.contains(b.p1) and inner.contains(b.p2)]
        if boxes:
            self.result.by_rule[rule] = self.result.by_rule.get(rule, 0) + len(boxes)
            self.result.markers.setdefault(rule, []).extend(boxes[:5])

    # --- the checks ---

    def layer_rules(self) -> None:
        r = self.rules
        g = self.d(r.grid)
        for role, region in self.layers.items():
            self.report(f"grid.{role.value}", region.grid_check(g, g), (role,))
        for role, width in r.min_width.items():
            if role in self.layers:
                self.report(f"width.{role.value}", self.region(role).width_check(self.d(width), False, kdb.Metrics.Projection), (role,))
        for role, space in r.min_space.items():
            if role in self.layers:
                self.report(f"space.{role.value}", self.region(role).space_check(self.d(space)), (role,))
        for role, area in r.min_area.items():
            if role in self.layers and area > 0:
                small = self.region(role).with_area(0, round(area / self.dbu**2), False)
                self.report(f"area.{role.value}", small, (role,))

    def enclosures(self) -> None:
        r = self.rules
        implants = {self.tech.implant_role(k) for k in DeviceKind}
        for (outer, inner), value in r.enclosure.items():
            if outer not in self.layers or inner not in self.layers:
                continue
            o = self.region(outer)
            if outer in implants:
                # a butted tap or diffusion meets the other implant at its edge: the two implants
                # together enclose it (which implant is right is the deck's and LVS's check)
                o = sum((self.region(role) for role in implants), kdb.Region()).merged()
            i = self.inner_shapes(outer, inner)
            name = f"enc.{outer.value}/{inner.value}"
            self.report(name + ".cover", i - o, (outer, inner))
            relaxed = r.relaxed_enclosure.get((outer, inner))
            if relaxed is None:
                self.report(name, i.enclosed_check(o, self.d(value), metrics=kdb.Metrics.Projection), (outer, inner))
                continue
            if relaxed > 0:
                self.report(
                    name + ".relaxed", i.enclosed_check(o, self.d(relaxed), metrics=kdb.Metrics.Projection), (outer, inner)
                )
            # the full value on two opposite sides: the error is two adjacent short sides, i.e.
            # short edges meeting at a corner (how the PDK decks state "of 2 adjacent edges")
            short = o.enclosing_check(i, self.d(value), metrics=kdb.Metrics.Projection).second_edges()
            corners = short.width_check(1, False, kdb.Metrics.Euclidian, 100.0)
            self.report(name, i.interacting(corners.polygons(1)), (outer, inner))

    def inner_shapes(self, outer: LayerRole, inner: LayerRole) -> kdb.Region:
        """The ``inner`` shapes an ``outer`` enclosure applies to."""
        i = self.region(inner)
        if inner in (LayerRole.DIFF, LayerRole.TAP) and self.tech.layer(LayerRole.DIFF) == self.tech.layer(LayerRole.TAP):
            # one layer for both (SG13G2's Activ): a device's diffusion is what poly crosses, a
            # tap the rest
            poly = self.region(LayerRole.POLY)
            i = i.interacting(poly) if inner is LayerRole.DIFF else i.not_interacting(poly)
        contact = self.tech.stack.contact
        if inner is contact and outer in (LayerRole.POLY, LayerRole.POLY_CUT_MARKER):
            return i.interacting(self.region(LayerRole.POLY)) - self.region(LayerRole.DIFF) - self.region(LayerRole.TAP)
        if inner is contact and outer in (LayerRole.DIFF, LayerRole.TAP):
            return i.interacting(self.region(outer))
        if inner in (LayerRole.DIFF, LayerRole.TAP) and outer is LayerRole.NWELL:
            return i.interacting(self.region(outer))
        # every cut is covered by the layers it joins; a layer shape only where it meets ``outer``
        return i if inner in self.cut_roles() else i.interacting(self.region(outer))

    def cut_roles(self) -> set[LayerRole]:
        return set(self.rules.cut_size)

    def cuts(self) -> None:
        r = self.rules
        for role, size in r.cut_size.items():
            if role not in self.layers:
                continue
            cuts = self.region(role)
            s = self.d(size)
            wrong = kdb.Region()
            for polygon in cuts.each():
                box = polygon.bbox()
                if not polygon.is_box() or box.width() != s or box.height() != s:
                    wrong.insert(polygon)
            self.report(f"cut.{role.value}", wrong, (role,))
            if role in r.cut_space:
                self.report(f"cutspace.{role.value}", cuts.space_check(self.d(r.cut_space[role])), (role,))

    def devices(self) -> None:
        r, tech = self.rules, self.tech
        poly, diff, tap = self.region(LayerRole.POLY), self.region(LayerRole.DIFF), self.region(LayerRole.TAP)
        contact = self.region(tech.stack.contact)
        gate = poly & diff
        on_diff_edge = gate.edges() & diff.edges()  # where the poly leaves the diffusion
        on_poly_edge = gate.edges() & poly.edges()  # the gate's source and drain sides
        self.report("dev.endcap", poly.edges().enclosing_check(on_diff_edge, self.d(r.poly_endcap)))
        self.report("dev.diff_extension", diff.edges().enclosing_check(on_poly_edge, self.d(r.diff_extension)))
        self.report("dev.gate_to_cut", (contact & diff).separation_check(gate, self.d(r.gate_to_cut_space)))
        self.report("dev.gate_space", gate.space_check(self.d(r.gate_to_gate_space)))
        self.report("dev.length", gate.width_check(self.d(r.device_min_length), False, kdb.Metrics.Projection))
        self.report("dev.width", on_poly_edge.with_length(0, self.d(r.device_min_width), False))
        if r.poly_to_diff:
            self.report("dev.poly_to_diff", poly.separation_check(diff + tap, self.d(r.poly_to_diff)))
        poly_contact = contact.interacting(poly) - diff - tap
        if r.poly_contact_to_diff:
            self.report("dev.poly_contact_to_diff", poly_contact.separation_check(diff + tap, self.d(r.poly_contact_to_diff)))
        pimp = self.region(tech.implant_role(DeviceKind.PMOS))
        nimp = self.region(tech.implant_role(DeviceKind.NMOS))
        # the two implants never overlap: each keeps the other its own enclosure away from its
        # diffusion (on SG13G2, where n+ is implied, that is what keeps pSD off n+ Activ)
        implant_roles = (tech.implant_role(DeviceKind.NMOS), tech.implant_role(DeviceKind.PMOS))
        self.report("dev.implant_overlap", nimp & pimp, implant_roles)
        if r.poly_contact_to_pdiff:
            self.report(
                "dev.poly_contact_to_pdiff", poly_contact.separation_check(diff & pimp, self.d(r.poly_contact_to_pdiff)),
            )
        nwell = self.region(LayerRole.NWELL)
        roles = (LayerRole.NWELL,)
        if r.nwell_to_ndiff_space:
            self.report("dev.nwell_to_ndiff", nwell.separation_check(diff & nimp, self.d(r.nwell_to_ndiff_space)), roles)
        if r.nwell_to_ptap_space:
            self.report("dev.nwell_to_ptap", nwell.separation_check(tap & pimp, self.d(r.nwell_to_ptap_space)), roles)
        dnwell = self.region(LayerRole.DNWELL)
        if not dnwell.is_empty():
            roles = (LayerRole.DNWELL,)
            if r.isolation_nwell_space:
                apart = nwell.not_interacting(dnwell)
                self.report("dev.dnwell_to_nwell", dnwell.separation_check(apart, self.d(r.isolation_nwell_space)), roles)
            if r.isolation_to_pdiff_space:
                pdiff = (diff & pimp) - dnwell
                self.report("dev.dnwell_to_pdiff", dnwell.separation_check(pdiff, self.d(r.isolation_to_pdiff_space)), roles)


#: Every rule field and the check names it feeds (the coverage test reads it).
CHECKED = {
    "grid": "grid.*",
    "min_width": "width.*", "min_space": "space.*", "min_area": "area.*", "enclosure": "enc.*",
    "relaxed_enclosure": "enc.*.relaxed", "cut_size": "cut.*", "cut_space": "cutspace.*",
    "poly_endcap": "dev.endcap", "diff_extension": "dev.diff_extension", "gate_to_cut_space": "dev.gate_to_cut",
    "gate_to_gate_space": "dev.gate_space", "device_min_length": "dev.length", "device_min_width": "dev.width",
    "poly_to_diff": "dev.poly_to_diff", "poly_contact_to_diff": "dev.poly_contact_to_diff",
    "poly_contact_to_pdiff": "dev.poly_contact_to_pdiff", "nwell_to_ndiff_space": "dev.nwell_to_ndiff",
    "nwell_to_ptap_space": "dev.nwell_to_ptap", "isolation_nwell_space": "dev.dnwell_to_nwell",
    "isolation_to_pdiff_space": "dev.dnwell_to_pdiff",
}


def check_rules(target, tech: Technology, *, leaf: bool = False, top_cell: str | None = None) -> RulesResult:
    """Check ``target`` (a gdsfactory component or a GDS path) against ``tech.rules``."""
    layout = kdb.Layout()
    if isinstance(target, (str, Path)):
        layout.read(str(target))
        source = layout.cell(top_cell) if top_cell else layout.top_cell()
        top = source
    else:
        layout.dbu = target.kcl.layout.dbu
        top = layout.create_cell("top")
        top.copy_tree(target.kcl.layout.cell(target.name))
    top.flatten(-1, True)
    # a tiled cell's edge is its pitch, not its bounding box (wells reach past it)
    info = {} if isinstance(target, (str, Path)) else dict(target.info)
    tile = (0.0, 0.0, info["pitch_x"], info["pitch_y"]) if leaf and "pitch_x" in info and "pitch_y" in info else None
    checker = _Checker(layout, top, tech, leaf, tile)
    checker.layer_rules()
    checker.enclosures()
    checker.cuts()
    checker.devices()
    result = checker.result
    result.markers = {k: [b.to_dtype(layout.dbu) for b in v] for k, v in result.markers.items()}
    return result
