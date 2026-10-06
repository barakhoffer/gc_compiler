"""The routing stack: which layers route, which cut joins each pair, and what the flow calls them."""

from __future__ import annotations

from dataclasses import dataclass

from gc_compiler.tech.rules import LayerRole


@dataclass(frozen=True)
class MetalStack:
    """The process's interconnect, from the layer closest to silicon outwards.

    The builders assume a local-interconnect layer between the contacts and metal1 (sky130's li1):
    diffusion and poly contacts land on it, and every via stack starts from it. A process without
    one (most) would need contacts landing on metal1 directly -- ``local_interconnect=None`` states
    that, but the builders do not implement it yet and refuse such a technology.
    """

    routing: tuple[LayerRole, ...]
    """Routing layers, closest to silicon first."""
    cuts: dict[tuple[LayerRole, LayerRole], LayerRole]
    """The cut joining each pair of adjacent routing layers."""
    local_interconnect: LayerRole | None
    """The layer diffusion and poly contacts land on (sky130: li1), or ``None`` for metal1."""
    contact: LayerRole
    """The cut from diffusion, tap and poly up to the first routing layer."""
    lef_names: dict[LayerRole, str]
    """Each routing layer's name in the technology LEF (what a place-and-route flow reads)."""
    power_pin_layer: LayerRole
    """The layer a hard macro's supply pins are on (above everything the cells use)."""

    def cut_between(self, lower: LayerRole, upper: LayerRole) -> LayerRole:
        return self.cuts[(lower, upper)]

    def above(self, role: LayerRole) -> LayerRole:
        """The next routing layer up from ``role``."""
        return self.routing[self.routing.index(role) + 1]

    def cut_above(self, role: LayerRole) -> LayerRole:
        """The cut from ``role`` to the next routing layer up."""
        return self.cuts[(role, self.above(role))]

    def lef_name(self, role: LayerRole) -> str:
        return self.lef_names[role]
