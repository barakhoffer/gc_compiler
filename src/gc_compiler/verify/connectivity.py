"""Geometric net tracing with KLayout: which shapes are electrically one piece of metal.

This is not LVS -- it knows nothing about devices' *function* -- but it answers the
question routing code most needs answered and DRC cannot: which points are on the same
net, and (just as important) which are not. Conductors are the routing metals and li,
the diffusion outside the channels (``diff - poly``), taps and poly; cuts join the layers
they sit between. Wells and the substrate are deliberately not conductors, so a p-tap and
an n-tap are different nets here exactly as they are in the mask data.

``metals_only`` restricts that to metal1-3 and the vias between them: the wiring between
blocks, without whatever the blocks themselves do underneath.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import klayout.db as kdb

from gc_compiler.tech.rules import LayerRole, Technology

def _graph(tech: Technology, metals_only: bool):
    """Conductors and the (cut, conductor) pairs joining them, from the technology's stack: the
    routing layers and their cuts, and (unless ``metals_only``) the contact from poly and tap up to
    the first routing layer. ``metals_only`` keeps the routing layers between the local
    interconnect and the power-pin layer."""
    stack = tech.stack
    routing = [r for r in stack.routing if r is not stack.power_pin_layer]
    if metals_only:
        routing = [r for r in routing if r is not stack.local_interconnect]
    joins = []
    for lower, upper in zip(routing, routing[1:]):
        cut = stack.cut_between(lower, upper)
        joins += [(cut, lower), (cut, upper)]
    conductors = list(routing)
    if not metals_only:
        conductors += [LayerRole.POLY, LayerRole.TAP]
        joins += [(stack.contact, stack.routing[0]), (stack.contact, LayerRole.POLY), (stack.contact, LayerRole.TAP)]
    return tuple(conductors), tuple(joins)


@dataclass
class Nets:
    """A traced layout: ask which net a point on a layer belongs to."""

    l2n: kdb.LayoutToNetlist
    layout: kdb.Layout
    regions: dict[LayerRole, kdb.Region]

    def net_at(self, role: LayerRole, point: tuple[float, float]) -> int | None:
        """Id of the net under ``point`` (um) on ``role``; ``None`` if there is no shape."""
        net = self.l2n.probe_net(self.regions[role], kdb.DPoint(*point))
        return None if net is None else net.cluster_id

    def same_net(
        self, a: tuple[LayerRole, tuple[float, float]], b: tuple[LayerRole, tuple[float, float]]
    ) -> bool:
        """True if both points hit metal and it is the same net."""
        na, nb = self.net_at(*a), self.net_at(*b)
        return na is not None and na == nb


def trace_nets(
    tech: Technology,
    gds: str | os.PathLike[str],
    top: str | None = None,
    metals_only: bool = False,
) -> Nets:
    """Trace connectivity of ``top`` (default: the GDS's only top cell) flat."""
    layout = kdb.Layout()
    layout.read(str(Path(gds)))
    cell = layout.cell(top) if top else layout.top_cell()
    l2n = kdb.LayoutToNetlist(kdb.RecursiveShapeIterator(layout, cell, []))

    def layer(role: LayerRole) -> kdb.Region:
        return l2n.make_layer(layout.layer(*tech.layer(role)), role.value)

    conductors, joins = _graph(tech, metals_only)
    cuts = {cut for cut, _conductor in joins}
    regions = {role: layer(role) for role in (*conductors, *cuts)}
    for role in conductors:
        l2n.connect(regions[role])
    for cut, conductor in joins:
        l2n.connect(regions[cut], regions[conductor])
    if not metals_only:
        # a gate is not a conductor: diffusion is cut into source/drain pieces by the poly
        regions[LayerRole.DIFF] = layer(LayerRole.DIFF).not_(regions[LayerRole.POLY])
        l2n.connect(regions[LayerRole.DIFF])
        l2n.connect(regions[tech.stack.contact], regions[LayerRole.DIFF])
    l2n.extract_netlist()
    return Nets(l2n, layout, regions)
