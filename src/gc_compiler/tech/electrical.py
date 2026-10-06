"""The process's electrical facts: supplies, corners and the parasitics the timing models use."""

from __future__ import annotations

from dataclasses import dataclass, field

from gc_compiler.tech.rules import LayerRole


@dataclass(frozen=True)
class Corner:
    """A process corner (a section of the model library) at a supply voltage and temperature."""

    process: str
    vdd: float
    temp: float

    @property
    def name(self) -> str:
        return f"{self.process}_{self.vdd:g}V_{self.temp:g}C"


@dataclass(frozen=True)
class Electrical:
    vdd_nominal: float
    """The core devices' nominal supply, volts."""
    device_voltage_rating: float
    """The most a core device's gate oxide may see, volts (a negative write rail adds to vdd)."""
    corners: tuple[Corner, ...]
    """The corners the benches run by default; the first is the typical one."""
    sheet_resistance: dict[LayerRole, float] = field(default_factory=dict)
    """Ohm/square of the layers the timing model resistances are computed for (poly word lines,
    their metal straps)."""
    strap_contact_resistance: float = 0.0
    """One contact stack from a poly word line up to its strap, ohms."""
    gate_cap_per_um2: float = 0.0
    """Gate-oxide capacitance, farads per square micrometre (a varactor's load)."""

    def corner(self, process: str) -> Corner:
        for corner in self.corners:
            if corner.process == process:
                return corner
        raise KeyError(f"no {process!r} corner (have {', '.join(c.process for c in self.corners)})")

    @property
    def processes(self) -> tuple[str, ...]:
        return tuple(c.process for c in self.corners)
