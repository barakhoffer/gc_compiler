"""Technology-independent description of a process for the gain-cell compiler.

Blocks are written against :class:`LayerRole` and :class:`DesignRules` only; concrete
layer numbers, implant names and device model names live in a PDK adapter
(see :mod:`gc_compiler.tech.sky130`).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gc_compiler.tech.electrical import Electrical
    from gc_compiler.tech.characterised import Characterised
    from gc_compiler.tech.policy import LayoutPolicy, Sizing
    from gc_compiler.tech.stack import MetalStack
    from gc_compiler.tech.tools import ToolSetup

Layer = tuple[int, int]


class LayerRole(str, Enum):
    """Generic role of a mask layer, independent of any PDK."""

    NWELL = "nwell"
    PWELL = "pwell"
    DNWELL = "dnwell"  # deep n-well: isolates a p-well from the substrate
    DIFF = "diff"
    TAP = "tap"
    POLY = "poly"
    NIMP = "nimp"
    PIMP = "pimp"
    HVT = "hvt"
    LVT = "lvt"
    POLY_CUT_MARKER = "poly_cut_marker"  # sky130 npc; absent in many PDKs
    LICON = "licon"
    LI = "li"
    MCON = "mcon"
    M1 = "m1"
    VIA1 = "via1"
    M2 = "m2"
    VIA2 = "via2"
    M3 = "m3"
    VIA3 = "via3"
    M4 = "m4"  # the power pins a hard-macro flow connects to (the cells use m3 and below)
    VIA4 = "via4"
    M5 = "m5"  # a process whose place-and-route grid is higher up (SG13G2's TopMetal1): its pins
    BOUNDARY = "boundary"


#: Routing layers, from closest to silicon outwards.
ROUTING_ROLES = (LayerRole.LI, LayerRole.M1, LayerRole.M2, LayerRole.M3, LayerRole.M4, LayerRole.M5)

#: Cut layer connecting each pair of adjacent routing layers.
CUT_BETWEEN: dict[tuple[LayerRole, LayerRole], LayerRole] = {
    (LayerRole.LI, LayerRole.M1): LayerRole.MCON,
    (LayerRole.M1, LayerRole.M2): LayerRole.VIA1,
    (LayerRole.M2, LayerRole.M3): LayerRole.VIA2,
    (LayerRole.M3, LayerRole.M4): LayerRole.VIA3,
    (LayerRole.M4, LayerRole.M5): LayerRole.VIA4,
}


class DeviceKind(str, Enum):
    NMOS = "nmos"
    PMOS = "pmos"


@dataclass(frozen=True)
class DesignRules:
    """Minimal rule set needed by the generic layout builders.

    Distances are in micrometers. Only rules actually consumed by the builders are
    modelled; a PDK adapter may be stricter than the real deck without breaking anything.
    """

    grid: float
    min_width: dict[LayerRole, float]
    min_space: dict[LayerRole, float]
    #: (outer, inner) -> minimum enclosure of ``inner`` by ``outer``.
    enclosure: dict[tuple[LayerRole, LayerRole], float]
    #: Enclosure allowed on the short sides when the other two sides use ``enclosure``.
    relaxed_enclosure: dict[tuple[LayerRole, LayerRole], float]
    #: Exact (square) size of each cut layer.
    cut_size: dict[LayerRole, float]
    #: Minimum spacing between two cuts of the same layer.
    cut_space: dict[LayerRole, float]

    # --- device-level rules ---
    poly_endcap: float
    """Poly extension beyond the diffusion edge, transverse to current flow."""
    gate_to_cut_space: float
    """Space between a gate edge and a source/drain cut."""
    gate_to_gate_space: float
    """Space between two gates sharing an uncontacted diffusion node."""
    diff_extension: float
    """Diffusion extension beyond a gate when the node carries no contact."""
    device_min_width: float
    device_min_length: float
    nwell_to_ndiff_space: float
    """Space from an n-well edge to n+ diffusion outside the well."""
    #: Minimum drawn area of an isolated shape on a layer; layers without a rule are 0.
    min_area: dict[LayerRole, float] = field(default_factory=dict)
    isolation_ring_inner: float = 0.0
    """How far inside a deep n-well's edge the n-well ring that closes it must reach (0 if the
    process has no deep n-well)."""
    isolation_ring_outer: float = 0.0
    """How far the n-well ring must extend beyond the deep n-well's edge."""
    isolation_nwell_space: float = 0.0
    """Minimum space from a deep n-well's edge to any n-well that does not overlap it."""
    poly_contact_to_diff: float = 0.0
    """Space from a contact on poly to diffusion or tap (a contacted poly pad is not a gate)."""
    poly_contact_to_pdiff: float = 0.0
    """Space from a contact on poly to p+ diffusion (stricter than to n+ where the p+ implant's
    rule adds to it)."""
    poly_to_diff: float = 0.0
    """Space from poly on field to diffusion or tap that it does not cross."""
    nwell_to_ptap_space: float = 0.0
    """Space from an n-well edge to a p+ tap outside it."""
    isolation_to_pdiff_space: float = 0.0
    """Space from a deep n-well's edge to p+ diffusion outside it (0 if no deep n-well)."""
    latchup_distance: float = 0.0
    """The farthest any diffusion may be from a well or substrate tap."""

    def w(self, role: LayerRole) -> float:
        return self.min_width[role]

    def s(self, role: LayerRole) -> float:
        return self.min_space[role]

    def enc(self, outer: LayerRole, inner: LayerRole) -> float:
        return self.enclosure[(outer, inner)]

    def enc_relaxed(self, outer: LayerRole, inner: LayerRole) -> float:
        """Enclosure for the two short sides of an elongated pad."""
        return self.relaxed_enclosure.get((outer, inner), self.enclosure[(outer, inner)])

    def cut(self, role: LayerRole) -> float:
        return self.cut_size[role]

    def area(self, role: LayerRole) -> float:
        return self.min_area.get(role, 0.0)

    def snap(self, value: float) -> float:
        return round(value / self.grid) * self.grid

    def snap_up(self, value: float) -> float:
        return math.ceil(value / self.grid - 1e-6) * self.grid

    def snap_down(self, value: float) -> float:
        return math.floor(value / self.grid + 1e-6) * self.grid

    def via_pad(self, metal: LayerRole, cut: LayerRole, relaxed: bool = False) -> float:
        """The side of a square pad of ``metal`` around one ``cut``."""
        enc = self.enc_relaxed(metal, cut) if relaxed else self.enc(metal, cut)
        return self.cut_size[cut] + 2 * enc

    def tap_width(self, contact: LayerRole = LayerRole.LICON) -> float:
        """The narrowest tap strip that holds a contact."""
        return self.cut_size[contact] + 2 * self.enc(LayerRole.TAP, contact)

    @property
    def isolation_ring_width(self) -> float:
        """Width of the n-well ring around an isolated p-well: it spans the deep n-well's
        ``isolation_ring_inner`` margin and its ``isolation_ring_outer`` overhang."""
        return self.isolation_ring_inner + self.isolation_ring_outer

    def cut_pitch(self, role: LayerRole) -> float:
        return self.cut_size[role] + self.cut_space[role]

    def gate_to_cut_center(self, role: LayerRole = LayerRole.LICON) -> float:
        """Distance from a gate edge to the center of an adjacent source/drain cut."""
        return self.gate_to_cut_space + self.cut_size[role] / 2

    def contacted_node_extent(self, role: LayerRole = LayerRole.LICON) -> float:
        """Diffusion extension beyond a contact center at a contacted node."""
        return self.cut_size[role] / 2 + self.enc(LayerRole.DIFF, role)


@dataclass(frozen=True)
class StdCellLibrary:
    """The process's digital standard-cell library, as the refresh controller's flow needs it
    (see :mod:`gc_compiler.flow`): LibreLane finds everything else from the PDK and library
    names."""

    name: str
    pdk: str
    pdk_root: str
    corners: dict[str, str]
    """The library's timing corners, keyed by the SPICE process corner each stands for."""
    liberty: dict[str, str]
    """Each corner's timing library, keyed the same way."""
    verilog: tuple[str, ...]
    """Functional models of the cells, for gate-level simulation."""
    verilog_defines: tuple[str, ...]
    conditions: dict[str, tuple[float, float]]
    """Each timing corner's (supply volts, temperature C), keyed like ``corners``."""
    site_width: float
    """The placement site's width, um: what block edges and macro positions snap to."""
    row_height: float
    power_pin: str
    ground_pin: str
    pdn: PowerGrid
    rc_corner: str
    """The flow's parasitic-extraction corner name (LibreLane: ``nom``)."""
    area_per_flop_um2: float
    """A flop with its share of the logic around it, measured on a synthesized controller: what
    the controller band is sized from before synthesis."""
    sdc: SdcConstants
    crossing_nets_per_um: float = 2.0
    """Nets running the width of a block that a um of the height below its macro can carry (a banked
    macro's per-column din and dout reach both banks, all crossing its middle): the band is made
    tall enough for them before its cells are counted. Measured on routed blocks."""
    flow_config: dict[str, object] = field(default_factory=dict)
    """Settings for the controller's flow (LibreLane variables: the clock tree's root buffer,
    synthesis options), what a block's own configuration does not set."""
    synth_excluded_cells: tuple[str, ...] | None = None
    """The cells synthesis may not use, where they differ from the PDK's list (``None``: the
    PDK's)."""


@dataclass(frozen=True)
class PowerGrid:
    """The core power grid a place-and-route flow builds over a block."""

    vertical_layer: str
    horizontal_layer: str
    horizontal_pitch: float
    horizontal_offset: float
    vertical_pitch: float
    align_to_macro: bool = False
    """Put a vertical strap of each net right on one of the macro's own stripes: where the grid's
    straps stop at a macro instead of crossing it (SG13G2's TopMetal1), those are what join the
    grid over the macro to the grid over the standard cells."""
    vertical_width: float = 0.0
    """The vertical straps' width (with ``align_to_macro``)."""
    vertical_min_spacing: float = 0.0
    """The closest two vertical straps may be (with ``align_to_macro``)."""
    grid: float = 0.0
    """The manufacturing grid the aligned offset and spacing are snapped to (with ``align_to_macro``)."""


@dataclass(frozen=True)
class SdcConstants:
    """Clock and I/O assumptions of the controller's timing constraints, in the library's units
    (ns, pF)."""

    setup_uncertainty: float
    hold_uncertainty: float
    clock_transition: float
    derate_early: float
    derate_late: float
    input_transition: float
    output_load: float


@dataclass(frozen=True)
class LayerStyle:
    """Display style for a role, used to build a gdsfactory LayerViews for plotting."""

    color: str
    hatch: str = "solid"
    transparent: bool = True


class Technology(ABC):
    """PDK adapter interface.

    Everything PDK-specific -- layer numbers, implant/well policy, device model names,
    extra marker layers -- is confined to implementations of this class.
    """

    name: str
    rules: DesignRules
    styles: dict[LayerRole, LayerStyle] = {}
    stack: MetalStack
    electrical: Electrical
    tools: ToolSetup
    policy: LayoutPolicy
    sizing: Sizing
    #: What was measured on this technology; ``None`` if nothing was (the compiler then refuses
    #: to guess a refresh interval or controller timing).
    characterised: Characterised | None = None
    #: Prepended to every cached cell's name, so two technologies' cells never share a name in
    #: gdsfactory's global layout ("" for the reference technology keeps its names unchanged).
    cell_prefix: str = ""

    @abstractmethod
    def layer(self, role: LayerRole) -> Layer:
        """GDS (layer, datatype) for a role."""

    @abstractmethod
    def label_layer(self, role: LayerRole) -> Layer | None:
        """GDS (layer, datatype) used for text labels drawn on ``role``."""

    def pin_layer(self, role: LayerRole) -> Layer | None:
        """GDS (layer, datatype) used for pin shapes drawn on ``role``."""
        return None

    def has(self, role: LayerRole) -> bool:
        try:
            self.layer(role)
        except KeyError:
            return False
        return True

    # --- device policy ---

    @abstractmethod
    def implant_role(self, kind: DeviceKind) -> LayerRole:
        """Source/drain implant that must cover the diffusion of a device."""

    @abstractmethod
    def well_role(self, kind: DeviceKind) -> LayerRole | None:
        """Well that must enclose the diffusion of a device, if any."""

    @abstractmethod
    def tap_implant_role(self, kind: DeviceKind) -> LayerRole:
        """Implant for a well/substrate tap biasing the body of ``kind`` devices."""

    def vt_role(self, flavor: str | None) -> LayerRole | None:
        """Marker layer for a threshold-voltage flavor (``None`` for the nominal device)."""
        if flavor in (None, "svt", ""):
            return None
        if flavor not in self.flavors:
            have = ", ".join(repr(f) for f in self.flavors if f) or "only the nominal device"
            raise ValueError(f"{self.name} has no {flavor!r} threshold flavor ({have})")
        return {"lvt": LayerRole.LVT, "hvt": LayerRole.HVT}[flavor]

    @abstractmethod
    def spice_model(self, kind: DeviceKind, flavor: str | None = None) -> str:
        """SPICE model name for a device."""

    #: Threshold flavors a model may come in (what :meth:`classify_model` tries).
    flavors: tuple[str | None, ...] = (None, "lvt", "hvt")

    def classify_model(self, model: str) -> DeviceKind | str | None:
        """What an extracted device's model is: a :class:`DeviceKind`, ``"varactor"`` (a gate over
        a well, which extraction reports as a device), or ``None`` for anything else."""
        for kind in DeviceKind:
            for flavor in self.flavors:
                try:
                    if self.spice_model(kind, flavor) == model:
                        return kind
                except KeyError:
                    continue
        return None

    def spice_device(
        self,
        ref: str,
        drain: str,
        gate: str,
        source: str,
        bulk: str,
        kind: DeviceKind,
        width: float,
        length: float,
        flavor: str | None = None,
    ) -> str:
        """One SPICE instance line; the pin order is PDK-specific and lives here."""
        model = self.spice_model(kind, flavor)
        return f"X{ref} {drain} {gate} {source} {bulk} {model} w={width:g} l={length:g}"

    # --- electrical constants for timing (see :attr:`electrical`) ---

    @property
    def poly_sheet_resistance(self) -> float:
        """Poly sheet resistance in ohm/square: a word line is poly all the way across the array."""
        return self.electrical.sheet_resistance.get(LayerRole.POLY, 0.0)

    @property
    def m3_sheet_resistance(self) -> float:
        """Metal3 sheet resistance in ohm/square: word-line straps."""
        return self.electrical.sheet_resistance.get(LayerRole.M3, 0.0)

    @property
    def strap_contact_resistance(self) -> float:
        """One poly-to-strap contact stack, ohms: where a word line meets its strap."""
        return self.electrical.strap_contact_resistance

    def standard_cells(self, root: str | None = None) -> StdCellLibrary:
        """The process's digital standard-cell library (for the refresh controller)."""
        raise NotImplementedError(f"{self.name} has no standard-cell library configured")

    def model_library(self, root: str | None = None) -> str:
        """Path of the SPICE device-model library, with a ``.lib`` corner argument added by the
        caller; the corners it offers are :attr:`model_corners`."""
        raise NotImplementedError

    def model_header(
        self, process: str, root: str | None = None, mismatch: bool = False
    ) -> list[str]:
        """SPICE lines that load the device models of one process corner.

        ``mismatch`` turns on the PDK's local-mismatch model: every device instance draws its own
        random parameter offsets (the draws are seeded by ``.option seed``), on top of the
        corner. Overridden where the library offers a cheaper way to load only the devices a
        transistor-level bench needs; the default loads the whole ``.lib`` section, whose
        mismatch variant is named ``<process>_mm``.
        """
        return [f'.lib "{self.model_library(root)}" {process}{"_mm" if mismatch else ""}']

    @property
    def model_corners(self) -> tuple[str, ...]:
        """Process corners the model library provides (the typical one first)."""
        return self.electrical.processes

    # --- optional decorations ---

    def decorate_gate_contact(self, component, bbox: tuple[float, float, float, float]) -> None:
        """Add PDK-specific markers around a contact landing on poly (sky130: npc)."""

    def decorate_device(self, component, kind: DeviceKind, bbox) -> None:
        """Add PDK-specific markers around a device diffusion."""
