"""Well fix-ups on assembled layout."""

from __future__ import annotations

import gdsfactory as gf
import klayout.db as kdb

from gc_compiler.tech.rules import LayerRole, Technology


def _region(component: gf.Component, layer: tuple[int, int]) -> kdb.Region:
    index = component.kcl.layout.find_layer(*layer)
    if index is None:
        return kdb.Region()
    return kdb.Region(component.begin_shapes_rec(index)).merged()


def close_well_ends(component: gf.Component, tech: Technology) -> int:
    """Extend the n-well wherever it does not enclose an n-tap or a p+ diffusion in it by the
    rule's margin; returns how many shapes were added.

    Tap and logic strips are drawn so that abutting cells' wells merge: each cell's well ends
    flush with its taps and diffusion at the cell edge, and the next cell's well carries on. At
    the last cell of a row there is no next cell, so the well ends too close (diff/tap.10 and
    diff/tap.8, which KLayout's deck does not check). Moving that end of the well out, on the
    assembled layout, closes every such end at once.

    The well's straight edge is moved out over its whole length where it falls short, rather
    than a bump added: a bump leaves steps and notches that the width and spacing rules catch.
    """
    dbu = component.kcl.layout.dbu
    nwell = _region(component, tech.layer(LayerRole.NWELL))
    if nwell.is_empty():
        return 0
    tap, diff = _region(component, tech.layer(LayerRole.TAP)), _region(component, tech.layer(LayerRole.DIFF))
    nimp, pimp = _region(component, tech.layer(LayerRole.NIMP)), _region(component, tech.layer(LayerRole.PIMP))
    inside = ((tap & nimp) + (diff & pimp)).merged().interacting(nwell)
    missing = kdb.Region()
    for role in (LayerRole.TAP, LayerRole.DIFF):
        margin = round(tech.rules.enc(LayerRole.NWELL, role) / dbu)
        missing += inside.minkowski_sum(kdb.Box(-margin, -margin, margin, margin)) - nwell
    return _move_edges_out(component, tech, LayerRole.NWELL, nwell, missing.merged())


def close_implant_ends(component: gf.Component, tech: Technology) -> int:
    """Extend each source/drain implant wherever it does not enclose its diffusion (or a tap, by
    the relaxed margin) by the rule's margin and no other implant is there -- the same problem as :func:`close_well_ends`, for the
    implants: a column of slices draws each implant as a stripe the full slice height, so at the
    column's ends a diffusion that runs to the slice edge has its implant end flush with it.
    Neither sky130 deck checks drawn implants (magic derives its own); the rules check does."""
    from gc_compiler.tech.rules import DeviceKind

    dbu = component.kcl.layout.dbu
    diff = _region(component, tech.layer(LayerRole.DIFF))
    tap = _region(component, tech.layer(LayerRole.TAP))
    if tech.layer(LayerRole.DIFF) == tech.layer(LayerRole.TAP):
        # one layer for both (SG13G2's Activ): a device's diffusion is what poly crosses
        poly = _region(component, tech.layer(LayerRole.POLY))
        diff, tap = diff.interacting(poly), diff.not_interacting(poly)
    implants = [tech.implant_role(kind) for kind in DeviceKind]
    added = 0
    for role in implants:
        implant = _region(component, tech.layer(role))
        if implant.is_empty():
            continue
        others = kdb.Region()
        for other in implants:
            if other is not role:
                others += _region(component, tech.layer(other))
        margin = round(tech.rules.enc(role, LayerRole.DIFF) / dbu)
        grown = (diff & implant).merged().minkowski_sum(kdb.Box(-margin, -margin, margin, margin))
        # a tap strip's ends need only the relaxed enclosure (its long sides are drawn in full)
        margin = round(tech.rules.enc_relaxed(role, LayerRole.TAP) / dbu)
        if margin:
            grown += (tap & implant).merged().minkowski_sum(kdb.Box(-margin, -margin, margin, margin))
        missing = (grown - implant - others).merged()
        added += _move_edges_out(component, tech, role, implant, missing)
    return added


def _move_edges_out(component: gf.Component, tech: Technology, role: LayerRole, outer, missing) -> int:
    """Grow ``outer`` (the merged ``role`` region) over ``missing`` by moving its straight edges out."""
    if missing.is_empty():
        return 0
    dbu = component.kcl.layout.dbu
    wmin = round(tech.rules.w(role) / dbu)
    edges = list(outer.edges().each())

    def touches(box: kdb.Box) -> bool:
        return not outer.interacting(kdb.Region(box)).is_empty()

    def edge_span(lo: int, hi: int, vertical: bool, at: int) -> tuple[int, int]:
        """The whole straight edge at ``at`` (an x for a vertical edge, a y otherwise) that
        an addition spanning ``lo..hi`` along it lies against."""
        for edge in edges:
            if vertical and edge.dx() == 0 and edge.p1.x == at:
                a, b = sorted((edge.p1.y, edge.p2.y))
            elif not vertical and edge.dy() == 0 and edge.p1.y == at:
                a, b = sorted((edge.p1.x, edge.p2.x))
            else:
                continue
            if a < hi and b > lo:
                lo, hi = min(lo, a), max(hi, b)
        return lo, hi

    added = kdb.Region()
    for polygon in missing.each():
        b = polygon.bbox()
        x0, y0, x1, y1 = b.left, b.bottom, b.right, b.top
        if touches(kdb.Box(x1, y0, x1 + 1, y1)):  # the layer is to its right: move its left edge
            y0, y1 = edge_span(y0, y1, True, x1)
        elif touches(kdb.Box(x0 - 1, y0, x0, y1)):
            y0, y1 = edge_span(y0, y1, True, x0)
        elif touches(kdb.Box(x0, y1, x1, y1 + 1)):  # the layer is above: move its bottom edge
            x0, x1 = edge_span(x0, x1, False, y1)
        elif touches(kdb.Box(x0, y0 - 1, x1, y0)):
            x0, x1 = edge_span(x0, x1, False, y0)
        else:  # a corner: square it off to the minimum width
            x1, y1 = max(x1, x0 + wmin), max(y1, y0 + wmin)
        added.insert(kdb.Box(x0, y0, x1, y1))
    added = (added - outer).merged()
    # two edges moved out apart can leave a notch between them narrower than the layer's spacing:
    # close it (only next to what was added, so separate shapes never merge)
    space = round(tech.rules.s(role) / dbu)
    near = kdb.Region()
    for polygon in added.each():
        near.insert(polygon.bbox().enlarged(space, space))
    whole = (outer + added).merged()
    closed = whole.sized(space // 2 + 1).sized(-(space // 2 + 1))
    added = (added + ((closed - whole) & near)).merged()
    layer = component.kcl.layout.layer(*tech.layer(role))
    for polygon in added.each():
        component.shapes(layer).insert(polygon)
    return added.count()


def fill_notches(component: gf.Component, tech: Technology, role: LayerRole | None = None) -> int:
    """Fill every gap narrower than the layer's spacing between two parts of one merged shape of
    ``role`` (the cell flattened): same-net pieces drawn by different cells, such as two cells'
    stubs off a shared rail. Returns how many were filled. Without a ``role``, every routing
    layer of the stack."""
    if role is None:
        return sum(fill_notches(component, tech, r) for r in tech.stack.routing)
    space = round(tech.rules.s(role) / component.kcl.layout.dbu)
    layer = component.kcl.layout.layer(*tech.layer(role))
    filled = 0
    for _ in range(4):  # a fill can leave a smaller notch next to it
        pairs = _region(component, tech.layer(role)).notch_check(space, False, kdb.Metrics.Projection)
        if pairs.is_empty():
            break
        for pair in pairs.each():
            component.shapes(layer).insert(pair.bbox())
            filled += 1
    return filled


def fix_min_area(component: gf.Component, tech: Technology, role: LayerRole | None = None, tile=None) -> int:
    """Grow every shape of ``role`` (merged, the cell flattened) below the layer's minimum area,
    along whichever axis keeps it at the layer's spacing from everything else; returns how many
    grew. Shapes touching ``tile`` (the cell's abutment box, default its bounding box) are left:
    the neighbour completes them. A rule of the merged shape, so it is fixed on the cell, once its
    pieces are drawn. Without a ``role``, every routing layer of the stack."""
    if role is None:
        return sum(fix_min_area(component, tech, r, tile) for r in tech.stack.routing)
    area = tech.rules.area(role)
    if area <= 0:
        return 0
    dbu = component.kcl.layout.dbu
    region = _region(component, tech.layer(role))
    if region.is_empty():
        return 0
    edge = kdb.DBox(*tile).to_itype(dbu) if tile is not None else component.dbbox().to_itype(dbu)
    inner = edge.enlarged(-1, -1)
    space = round(tech.rules.s(role) / dbu)
    step = 2 * round(tech.rules.grid / dbu)
    need = area / dbu**2
    layer = component.kcl.layout.layer(*tech.layer(role))
    grown = 0
    for polygon in list(region.each()):
        if polygon.area() >= need - 1e-6:
            continue
        box = polygon.bbox()
        if not (inner.contains(box.p1) and inner.contains(box.p2)):
            # on an abutment edge: the neighbour is this cell mirrored across it (the array mirrors
            # its rows), so the shape is completed by its own mirror image -- grow it along the edge
            # if the two together are short
            added = _fix_edge_shape(component, tech, region, polygon, edge, need, space, step, layer)
            if added is not None:
                region = (region + kdb.Region(added)).merged()
            continue
        others = region - kdb.Region(polygon)
        extra = polygon.area()
        add_x = -(-max(0, int(-(-(need - extra) // box.height()))) // step) * step
        add_y = -(-max(0, int(-(-(need - extra) // box.width()))) // step) * step
        # symmetric first (along the longer side), then toward one side only
        candidates = [
            kdb.Box(box.left - add_x // 2, box.bottom, box.right + add_x - add_x // 2, box.top),
            kdb.Box(box.left, box.bottom - add_y // 2, box.right, box.top + add_y - add_y // 2),
            kdb.Box(box.left - add_x, box.bottom, box.right, box.top),
            kdb.Box(box.left, box.bottom, box.right + add_x, box.top),
            kdb.Box(box.left, box.bottom - add_y, box.right, box.top),
            kdb.Box(box.left, box.bottom, box.right, box.top + add_y),
        ]
        if box.height() > box.width():
            candidates[0], candidates[1] = candidates[1], candidates[0]
        for candidate in candidates:
            if tile is not None and not edge.contains(candidate.p1) or tile is not None and not edge.contains(candidate.p2):
                continue  # a tiled cell's shapes stay inside its abutment box
            if others.interacting(kdb.Region(candidate.enlarged(space - 1, space - 1))).is_empty():
                component.shapes(layer).insert(candidate)
                # what the next shape keeps its spacing from includes this growth
                region = (region + kdb.Region(candidate)).merged()
                grown += 1
                break
    return grown


def _fix_edge_shape(component, tech, region, polygon, edge, need, space, step, layer) -> None:
    box = polygon.bbox()
    for at, axis in ((edge.bottom, "y"), (edge.top, "y"), (edge.left, "x"), (edge.right, "x")):
        touches = (box.bottom <= at <= box.top) if axis == "y" else (box.left <= at <= box.right)
        if not touches:
            continue
        mirror = kdb.Trans(kdb.Trans.M0, kdb.Vector(0, 2 * at)) if axis == "y" else kdb.Trans(kdb.Trans.M90, kdb.Vector(2 * at, 0))
        whole = (kdb.Region(polygon) + kdb.Region(polygon).transformed(mirror)).merged()
        area = whole.area()
        if area >= need - 1e-6:
            return None
        # grow along the edge (x for a y edge), symmetric about the shape
        along = box.width() if axis == "y" else box.height()
        across = whole.bbox().height() if axis == "y" else whole.bbox().width()
        add = max(0, int(-(-(need - area) // max(across, 1))))
        add = -(-add // step) * step
        half = add // 2
        grown = (
            kdb.Box(box.left - half, box.bottom, box.right + (add - half), box.top) if axis == "y"
            else kdb.Box(box.left, box.bottom - half, box.right, box.top + (add - half))
        )
        others = region - kdb.Region(polygon)
        others = others + others.transformed(mirror)
        if others.interacting(kdb.Region(grown.enlarged(space - 1, space - 1))).is_empty():
            component.shapes(layer).insert(grown)
            return grown
        return None
    return None
