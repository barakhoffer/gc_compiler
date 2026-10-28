"""SkyWater sky130 adapter: every sky130 value the compiler uses.

Layer numbers, marker layers and device models; the design rules (DRM rule ids in the
comments); the routing stack and its LEF names (:data:`STACK`); supplies, corners and parasitics
(:data:`ELECTRICAL`); the KLayout, magic and netgen setups and their waivers (:data:`TOOLS`);
layout choices (:data:`POLICY`); default device sizes (:data:`SIZING`); the ``sky130_fd_sc_hd``
flow facts (:meth:`Sky130.standard_cells`); and what was measured on sky130 -- refresh intervals,
controller timing (:data:`CHARACTERISED`). A guard test keeps sky130 names out of the rest of the
code. Porting means writing another :class:`~gc_compiler.tech.rules.Technology` subclass like this
one, within the structural assumptions the README lists; :mod:`gc_compiler.tech.sg13g2` (IHP
SG13G2) is the second.

Rule values follow the public sky130 design-rule manual; where the manual gives a
"one of two adjacent sides" relaxation the stricter value is used, so generated layout
is conservative by construction.
"""

from __future__ import annotations

from gc_compiler.tech.characterised import Characterised, ControllerDefaults, RefreshInterval
from gc_compiler.tech.electrical import Corner, Electrical
from gc_compiler.tech.policy import LayoutPolicy, PowerStripe, Sizing
from gc_compiler.tech.rules import (
    DesignRules,
    DeviceKind,
    Layer,
    LayerRole,
    LayerStyle,
    PowerGrid,
    SdcConstants,
    StdCellLibrary,
    Technology,
)
from gc_compiler.tech.stack import MetalStack
from gc_compiler.tech.tools import ToolSetup

LAYERS: dict[LayerRole, Layer] = {
    LayerRole.NWELL: (64, 20),
    LayerRole.DNWELL: (64, 18),
    LayerRole.DIFF: (65, 20),
    LayerRole.TAP: (65, 44),
    LayerRole.POLY: (66, 20),
    LayerRole.LICON: (66, 44),
    LayerRole.NIMP: (93, 44),  # nsdm
    LayerRole.PIMP: (94, 20),  # psdm
    LayerRole.HVT: (78, 44),  # hvtp
    LayerRole.LVT: (125, 44),  # lvtn
    LayerRole.POLY_CUT_MARKER: (95, 20),  # npc
    LayerRole.LI: (67, 20),
    LayerRole.MCON: (67, 44),
    LayerRole.M1: (68, 20),
    LayerRole.VIA1: (68, 44),
    LayerRole.M2: (69, 20),
    LayerRole.VIA2: (69, 44),
    LayerRole.M3: (70, 20),
    LayerRole.VIA3: (70, 44),
    LayerRole.M4: (71, 20),
    LayerRole.BOUNDARY: (235, 4),
}

LABEL_LAYERS: dict[LayerRole, Layer] = {
    LayerRole.LI: (67, 5),
    LayerRole.M1: (68, 5),
    LayerRole.M2: (69, 5),
    LayerRole.M3: (70, 5),
    LayerRole.M4: (71, 5),
    LayerRole.POLY: (66, 5),
    LayerRole.DIFF: (65, 6),
}

PIN_LAYERS: dict[LayerRole, Layer] = {
    LayerRole.LI: (67, 16),
    LayerRole.M1: (68, 16),
    LayerRole.M2: (69, 16),
    LayerRole.M3: (70, 16),
    LayerRole.M4: (71, 16),
}

STYLES: dict[LayerRole, LayerStyle] = {
    LayerRole.NWELL: LayerStyle("#c8a2c8", "lightly left-hatched"),
    LayerRole.DNWELL: LayerStyle("#8e6bb0", "dotted"),
    LayerRole.DIFF: LayerStyle("#4caf50", "solid"),
    LayerRole.TAP: LayerStyle("#1b5e20", "coarsely dotted"),
    LayerRole.POLY: LayerStyle("#e53935", "solid"),
    LayerRole.LICON: LayerStyle("#ffffff", "solid"),
    LayerRole.NIMP: LayerStyle("#64b5f6", "lightly right-hatched"),
    LayerRole.PIMP: LayerStyle("#ffb74d", "lightly right-hatched"),
    LayerRole.HVT: LayerStyle("#b39ddb", "dotted"),
    LayerRole.LVT: LayerStyle("#80cbc4", "dotted"),
    LayerRole.POLY_CUT_MARKER: LayerStyle("#9e9e9e", "hollow"),
    LayerRole.LI: LayerStyle("#ab47bc", "cross-hatched"),
    LayerRole.MCON: LayerStyle("#ffffff", "solid"),
    LayerRole.M1: LayerStyle("#1e88e5", "left-hatched"),
    LayerRole.VIA1: LayerStyle("#ffffff", "solid"),
    LayerRole.M2: LayerStyle("#fb8c00", "right-hatched"),
    LayerRole.VIA2: LayerStyle("#ffffff", "solid"),
    LayerRole.M3: LayerStyle("#43a047", "cross-hatched"),
    LayerRole.VIA3: LayerStyle("#ffffff", "solid"),
    LayerRole.M4: LayerStyle("#8d6e63", "left-hatched"),
    LayerRole.BOUNDARY: LayerStyle("#9e9e9e", "hollow"),
}

SKY130_RULES = DesignRules(
    grid=0.005,
    min_width={
        LayerRole.NWELL: 0.84,
        LayerRole.DNWELL: 3.0,
        LayerRole.DIFF: 0.15,
        LayerRole.TAP: 0.15,
        LayerRole.POLY: 0.15,
        LayerRole.NIMP: 0.38,
        LayerRole.PIMP: 0.38,
        LayerRole.HVT: 0.38,
        LayerRole.LVT: 0.38,
        LayerRole.POLY_CUT_MARKER: 0.27,
        LayerRole.LI: 0.17,
        LayerRole.M1: 0.14,
        LayerRole.M2: 0.14,
        LayerRole.M3: 0.30,
        LayerRole.M4: 0.30,  # m4.1
    },
    min_space={
        LayerRole.NWELL: 1.27,
        LayerRole.DNWELL: 6.3,
        LayerRole.DIFF: 0.27,
        LayerRole.TAP: 0.27,
        LayerRole.POLY: 0.21,
        LayerRole.NIMP: 0.38,
        LayerRole.PIMP: 0.38,
        LayerRole.HVT: 0.38,
        LayerRole.LVT: 0.38,
        LayerRole.POLY_CUT_MARKER: 0.27,
        LayerRole.LI: 0.17,
        LayerRole.M1: 0.14,
        LayerRole.M2: 0.14,
        LayerRole.M3: 0.30,
        LayerRole.M4: 0.30,  # m4.2
    },
    enclosure={
        (LayerRole.DIFF, LayerRole.LICON): 0.06,
        (LayerRole.TAP, LayerRole.LICON): 0.12,
        (LayerRole.POLY, LayerRole.LICON): 0.08,
        (LayerRole.LI, LayerRole.LICON): 0.08,
        (LayerRole.LI, LayerRole.MCON): 0.08,
        (LayerRole.M1, LayerRole.MCON): 0.06,
        (LayerRole.M1, LayerRole.VIA1): 0.085,
        (LayerRole.M2, LayerRole.VIA1): 0.085,
        (LayerRole.M2, LayerRole.VIA2): 0.085,
        (LayerRole.M3, LayerRole.VIA2): 0.085,
        (LayerRole.M3, LayerRole.VIA3): 0.09,  # via3.5
        (LayerRole.M4, LayerRole.VIA3): 0.065,  # m4.3
        (LayerRole.NWELL, LayerRole.DIFF): 0.18,
        (LayerRole.NWELL, LayerRole.TAP): 0.18,
        (LayerRole.NIMP, LayerRole.DIFF): 0.125,
        (LayerRole.PIMP, LayerRole.DIFF): 0.125,
        (LayerRole.NIMP, LayerRole.TAP): 0.125,
        (LayerRole.PIMP, LayerRole.TAP): 0.125,
        (LayerRole.HVT, LayerRole.DIFF): 0.18,
        (LayerRole.LVT, LayerRole.DIFF): 0.18,
        (LayerRole.POLY_CUT_MARKER, LayerRole.LICON): 0.10,
    },
    relaxed_enclosure={
        (LayerRole.LI, LayerRole.LICON): 0.0,  # li.5: 0.08 on one of two adjacent sides
        (LayerRole.LI, LayerRole.MCON): 0.0,  # ct.4: li only covers mcon
        (LayerRole.NIMP, LayerRole.TAP): 0.0,  # a tap strip's implant may end flush with it (magic)
        (LayerRole.PIMP, LayerRole.TAP): 0.0,
        (LayerRole.M1, LayerRole.MCON): 0.03,
        (LayerRole.M1, LayerRole.VIA1): 0.055,
        (LayerRole.M2, LayerRole.VIA1): 0.055,
        (LayerRole.M2, LayerRole.VIA2): 0.065,
        (LayerRole.M3, LayerRole.VIA2): 0.065,
        (LayerRole.M3, LayerRole.VIA3): 0.06,  # via3.4
        (LayerRole.M4, LayerRole.VIA3): 0.065,
    },
    cut_size={
        LayerRole.LICON: 0.17,
        LayerRole.MCON: 0.17,
        LayerRole.VIA1: 0.15,
        LayerRole.VIA2: 0.20,
        LayerRole.VIA3: 0.20,  # via3.1
    },
    cut_space={
        LayerRole.LICON: 0.17,
        LayerRole.MCON: 0.19,
        LayerRole.VIA1: 0.17,
        LayerRole.VIA2: 0.20,
        LayerRole.VIA3: 0.20,  # via3.2
    },
    poly_endcap=0.13,
    gate_to_cut_space=0.06,
    gate_to_gate_space=0.21,
    diff_extension=0.25,
    device_min_width=0.42,
    device_min_length=0.15,
    nwell_to_ndiff_space=0.34,
    isolation_ring_inner=1.03,  # nwell.6: dnwell encloses the ring's hole by 1.03 um
    isolation_ring_outer=0.40,  # the ring reaches 0.4 um past the dnwell edge (magic drc)
    isolation_nwell_space=4.5,  # nwell.7: unrelated n-wells stay 4.5 um from a deep n-well
    poly_contact_to_diff=0.19,  # licon.14 (KLayout's mr deck does not check it; magic does)
    poly_contact_to_pdiff=0.235,  # licon.9 + psdm.5a (likewise)
    poly_to_diff=0.075,  # poly.4
    nwell_to_ptap_space=0.13,  # diff/tap.11
    isolation_to_pdiff_space=0.10,  # dnwell.5 margin: magic flags p+ diffusion right at the edge
    latchup_distance=15.0,  # LU.2/LU.3: diffusion within 15 um of a tap
    min_area={
        LayerRole.LI: 0.0561,
        LayerRole.M1: 0.083,
        LayerRole.M2: 0.0676,
        LayerRole.M3: 0.240,
        LayerRole.M4: 0.240,  # m4.4a
    },
)

MODELS: dict[tuple[DeviceKind, str | None], str] = {
    (DeviceKind.NMOS, None): "sky130_fd_pr__nfet_01v8",
    (DeviceKind.NMOS, "lvt"): "sky130_fd_pr__nfet_01v8_lvt",
    (DeviceKind.PMOS, None): "sky130_fd_pr__pfet_01v8",
    (DeviceKind.PMOS, "lvt"): "sky130_fd_pr__pfet_01v8_lvt",
    (DeviceKind.PMOS, "hvt"): "sky130_fd_pr__pfet_01v8_hvt",
}


STACK = MetalStack(
    routing=(LayerRole.LI, LayerRole.M1, LayerRole.M2, LayerRole.M3, LayerRole.M4),
    cuts={
        (LayerRole.LI, LayerRole.M1): LayerRole.MCON,
        (LayerRole.M1, LayerRole.M2): LayerRole.VIA1,
        (LayerRole.M2, LayerRole.M3): LayerRole.VIA2,
        (LayerRole.M3, LayerRole.M4): LayerRole.VIA3,
    },
    local_interconnect=LayerRole.LI,
    contact=LayerRole.LICON,
    lef_names={
        LayerRole.LI: "li1", LayerRole.M1: "met1", LayerRole.M2: "met2", LayerRole.M3: "met3", LayerRole.M4: "met4",
    },
    power_pin_layer=LayerRole.M4,
)

ELECTRICAL = Electrical(
    vdd_nominal=1.8,
    device_voltage_rating=1.95,  # the 1.8 V devices' maximum
    # typical; slow (-10%, hot); fast (+10%, cold); the two skewed ones at nominal
    corners=(
        Corner("tt", 1.8, 25.0), Corner("ss", 1.62, 125.0), Corner("ff", 1.98, -40.0),
        Corner("sf", 1.8, 25.0), Corner("fs", 1.8, 25.0),
    ),
    # poly (silicided) and met3, ohm/square
    sheet_resistance={LayerRole.POLY: 48.2, LayerRole.M3: 0.047},
    strap_contact_resistance=169.2,  # licon on poly (152), mcon (9.3), via (4.5), via2 (3.4)
    gate_cap_per_um2=8.5e-15,  # tox ~ 4 nm
)

TOOLS = ToolSetup(
    variant="sky130A",
    klayout_deck="libs.tech/klayout/drc/sky130A_mr.drc",
    magicrc="libs.tech/magic/sky130A.magicrc",
    netgen_setup="libs.tech/netgen/sky130A_setup.tcl",
    klayout_deck_vars={"feol": "true", "beol": "true", "offgrid": "true", "floating_met": "false"},
    magic_drc_style="drc(full)",
    context_waivers=("nwell.4", "LU.2", "LU.3"),
    # wells ending at a leaf's edges (diff/tap.10, .8 -- closed on the assembled layout); a deep
    # n-well ring's rules that hold once its slices abut ("nwell.5a, 7", nwell.7)
    # (nwell.1: a lone cell's n-well pokes past its edge by the write device's enclosure where the
    # device sits at the poly endcap from the edge -- a sliver the neighbour's well covers in an array)
    leaf_waivers=("diff/tap.10", "diff/tap.8", "nwell.5a, 7", "nwell.7", "nwell.1"),
)

POLICY = LayoutPolicy(
    tap_row_interval=4,  # 4 row pairs of 2.93 um (the default cell): every diffusion well within 15 um of a tap row
    strap_column_interval=16,
    power_stripe=PowerStripe(width=1.2, pitch=25.0, clearance=1.5, site_pitch=3.0, scan_step=0.1),
    join=0.05,
    margin=0.05,
    poly_over_well_is_device=True,  # sky130_fd_pr__cap_var: poly over n-well is a varactor
)

SIZING = Sizing(
    # a standard-Vt write device with a 1.0 um mr. The LVT write device (0.35 um, the default before)
    # writes a lower '0' but leaks 2.6 nA (tt) to 134 nA (fs, 25 C) into a stored '0' whenever its
    # write bit line is high -- every read of a '1' in its column, every write of a '1' -- against
    # ~2 pA standard-Vt: on the extracted macro, 300 reads of another row lose every '0' of the
    # column at tt, ss, ff and fs. The standard device's written '0' is higher (about 1.0 V at sf,
    # the slow PMOS), which a 0.5 um mr misreads there and a 1.0 um one does not
    # (measurements/data/write_device.json)
    bitcell={
        "write_width": 0.42, "write_length": 0.18, "read_width": 0.42, "read_length": 0.15,
        "mr_length": 1.0, "write_flavor": None,
    },
    plain_bitcell={"mr_length": None},
    logic={"pmos_width": 0.64, "nmos_width": 0.42, "length": 0.15},
    # four times the logic cells': the banked predecode drives up to four faces (README)
    predecode={"pmos_width": 2.56, "nmos_width": 1.68},
    # the smallest ratioed pair that switches in every corner (ss/125 C the worst)
    level_shifter={"pmos_width": 1.5, "latch_width": 0.42, "latch_length": 0.8, "input_length": 0.15},
    size_grid=0.01,
    wide_array_cols=64,  # at 64 columns straps and 4x drivers cut a read from 1.78 to 1.09 ns
    wide_driver_strength=4.0,
    write_wl_low=-0.3,  # a written '0' reads back at sf with 0.23 V of margin (measurements/data/vneg.json)
)

#: Measured on sky130 (``measurements/data``, the README): the worst of 20 mismatch dies at ss, 1.62 V,
#: 125 C on the extracted 16x20 macro, and the controller's defaults.
#: The refresh controller's flow (measurements/data/controller_area_block.json, routed blocks): a
#: 4x clock-tree root, not the 16x -- a controller of some 50 flops needs no more; and repeaters on
#: any net longer than 300 um -- a wide banked macro's din and dout reach both banks, and at 128
#: columns those wires broke the antenna rule into the macro's gates, where no diode fits.
FLOW_CONFIG = {"CTS_ROOT_BUFFER": "sky130_fd_sc_hd__clkbuf_4", "DESIGN_REPAIR_MAX_WIRE_LENGTH": 300}

#: Drive-1 cells the PDK's LibreLane setup keeps from synthesis (``no_synth.cells``) that the
#: controller uses: its loads are a few gates, timing and slew are met at every corner with them,
#: and they take 6-8% off its routed area.
DRIVE1_CELLS = frozenset(
    f"sky130_fd_sc_hd__{c}"
    for c in ("dfxtp_1", "dfrtp_1", "dlxtp_1", "inv_1", "nand2_1", "nor2_1", "and2_1", "a21oi_1", "xor2_1", "xnor2_1")
)


def _synth_excluded(pdk) -> tuple[str, ...] | None:
    """The PDK's ``no_synth.cells`` less :data:`DRIVE1_CELLS` (``None`` without the PDK)."""
    path = pdk / "libs.tech" / "openlane" / "sky130_fd_sc_hd" / "no_synth.cells"
    if not path.exists():
        return None
    cells = (c.strip() for c in path.read_text().splitlines())
    return tuple(c for c in cells if c and not c.startswith("#") and c not in DRIVE1_CELLS)


CHARACTERISED = Characterised(
    refresh_intervals=(
        RefreshInterval({}, 154.0),  # the default cell (measurements/data/retention_mc_ss.json, std_mr1)
        # (the LVT write cells': measured with the write bit lines low, as the controller now holds
        # them; a read or write of a '1' in the column still disturbs their '0's -- see SIZING)
        RefreshInterval({"mr_length": 0.5, "write_flavor": "lvt", "write_length": 0.35}, 614.0),
        RefreshInterval({"mr_length": None, "write_flavor": "lvt", "write_length": 0.35, "read_length": 0.5}, 571.0),
        RefreshInterval({"write_length": 0.18, "mr_length": None, "write_flavor": None}, 88.0, negative_rail=-0.3),
    ),
    unmeasured_refresh_us=45.0,  # the plain cell's worst die: the shortest measured
    strobe_factor=1.25,
    controller=ControllerDefaults(
        read_ns=4.0, write_ns=2.0, precharge_ns=2.0, sample_ns=1.5, macro_min_ns=0.5, rdy_lag_ns=1.0,
        timeout_ns=50.0,
    ),
    idle_storage_node_v=0.4,
    decayed_one_fraction=0.7,
    macro_input_pf=0.005,
    macro_output_transition_ns=0.2,
    # the default cell's '0' at sf: 0.92 V written, a loop-back column reads up to 0.86 V (18 of 20
    # mismatch dies fail; 1 of 20 with in-macro refresh, which reads up to 0.98 V)
    loopback_marginal=({},),
    # a '1' decayed to 0.7 vdd, refreshed (the level the access leaves, the worse column): 2 x 32x32 at
    # ss (125 C) brings it back to 1.59 V (2 ns write; restored is 0.95 vdd, 1.54 V), 2 x 64x64 only to
    # 1.19 V (1.27 V however long the write; 1.54 V at tt), 2 x 128x128 to 1.06 V already at tt -- its
    # read bit line has 84 fF, 43 fF of it to its own write bit line
    refresh_max_rows=32,
)

#: What magic extracts a poly line over an n-well as.
VARACTORS = {"sky130_fd_pr__cap_var_lvt", "sky130_fd_pr__cap_var_hvt", "sky130_fd_pr__cap_var"}


class Sky130(Technology):
    """sky130A (1.8 V logic devices, li/met1..met3 stack)."""

    name = "sky130A"
    rules = SKY130_RULES
    styles = STYLES
    stack = STACK
    electrical = ELECTRICAL
    tools = TOOLS
    policy = POLICY
    sizing = SIZING
    characterised = CHARACTERISED

    def standard_cells(self, root: str | None = None) -> StdCellLibrary:
        """``sky130_fd_sc_hd``. Its timing corners closest to the SPICE benches': tt at 25 C, ss at
        1.60 V and 100 C (the library has no 125 C corner), ff at 1.95 V and -40 C."""
        pdk_root = self.tools.pdk_root(root)
        ref = pdk_root / self.tools.variant / "libs.ref" / "sky130_fd_sc_hd"
        corners = {"tt": "tt_025C_1v80", "ss": "ss_100C_1v60", "ff": "ff_n40C_1v95"}
        return StdCellLibrary(
            name="sky130_fd_sc_hd",
            pdk=self.tools.variant,
            pdk_root=str(pdk_root),
            corners=corners,
            liberty={p: str(ref / "lib" / f"sky130_fd_sc_hd__{c}.lib") for p, c in corners.items()},
            verilog=(str(ref / "verilog" / "primitives.v"), str(ref / "verilog" / "sky130_fd_sc_hd.v")),
            verilog_defines=("FUNCTIONAL", "UNIT_DELAY=#1"),
            conditions={"tt": (1.8, 25.0), "ss": (1.6, 100.0), "ff": (1.95, -40.0)},
            site_width=0.46,
            row_height=2.72,
            power_pin="VPWR",
            ground_pin="VGND",
            pdn=PowerGrid(
                vertical_layer="met4", horizontal_layer="met5",
                horizontal_pitch=20.0, horizontal_offset=5.0, vertical_pitch=30.0,
            ),
            rc_corner="nom",
            crossing_nets_per_um=2.5,  # 8x64 banked routed with 128 such nets in 46 um (2.8 a um)
            area_per_flop_um2=50.0,  # 45-50 um2 a flop or latch on routed controllers (8x8 to 8x64)
            flow_config=FLOW_CONFIG,
            synth_excluded_cells=_synth_excluded(pdk_root / self.tools.variant),
            sdc=SdcConstants(
                setup_uncertainty=0.25, hold_uncertainty=0.1, clock_transition=0.15,
                derate_early=0.95, derate_late=1.05, input_transition=0.1, output_load=0.02,
            ),
        )

    def model_library(self, root: str | None = None) -> str:
        return str(self.tools.path("libs.tech/ngspice/sky130.lib.spice", root))

    def model_header(
        self, process: str, root: str | None = None, mismatch: bool = False
    ) -> list[str]:
        """The MOSFET/varactor corner file and the typical resistor parameters it refers to.

        ``sky130.lib.spice`` also pulls in the linearised resistor/capacitor models and the
        several-thousand-bin special-cell library; ngspice needs ~40 s to parse those and a
        timing bench uses none of it (interconnect is extracted, not modelled).

        Mismatch is the library's own ``mc_mm_switch`` (what its ``<corner>_mm`` sections set):
        each transistor's vth0, voff and toxe get an independent Gaussian offset with a
        Pelgrom ``1/sqrt(W*L)`` sigma. The library's ``mc_pr_switch`` (process) is not used by
        any model in this PDK release, so global variation is what the corners are for.
        """
        from pathlib import Path

        library = Path(self.model_library(root))
        corner = library.parent / "corners" / f"{process}.spice"
        params = library.parent / "r+c" / "res_typical__cap_typical.spice"
        return [
            f".param mc_mm_switch={int(mismatch)}",
            ".param mc_pr_switch=0",
            f'.include "{corner}"',
            f'.include "{params}"',
        ]

    def layer(self, role: LayerRole) -> Layer:
        return LAYERS[role]

    def label_layer(self, role: LayerRole) -> Layer | None:
        return LABEL_LAYERS.get(role)

    def pin_layer(self, role: LayerRole) -> Layer | None:
        return PIN_LAYERS.get(role)

    def implant_role(self, kind: DeviceKind) -> LayerRole:
        return LayerRole.NIMP if kind is DeviceKind.NMOS else LayerRole.PIMP

    def well_role(self, kind: DeviceKind) -> LayerRole | None:
        return LayerRole.NWELL if kind is DeviceKind.PMOS else None

    def tap_implant_role(self, kind: DeviceKind) -> LayerRole:
        # A tap biasing an NMOS body is a p+ tap in the substrate, and vice versa.
        return LayerRole.PIMP if kind is DeviceKind.NMOS else LayerRole.NIMP

    def spice_model(self, kind: DeviceKind, flavor: str | None = None) -> str:
        return MODELS[(kind, flavor)]

    def classify_model(self, model: str) -> DeviceKind | str | None:
        if model in VARACTORS:
            return "varactor"
        return super().classify_model(model)

    def decorate_gate_contact(self, component, bbox) -> None:
        """sky130 requires an npc (nitride poly cut) marker over every licon on poly."""
        x0, y0, x1, y1 = bbox
        enc = self.rules.enc(LayerRole.POLY_CUT_MARKER, LayerRole.LICON)
        min_w = self.rules.w(LayerRole.POLY_CUT_MARKER)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hx = max((x1 - x0) / 2 + enc, min_w / 2)
        hy = max((y1 - y0) / 2 + enc, min_w / 2)
        component.add_polygon(
            [
                (cx - hx, cy - hy),
                (cx + hx, cy - hy),
                (cx + hx, cy + hy),
                (cx - hx, cy + hy),
            ],
            layer=self.layer(LayerRole.POLY_CUT_MARKER),
        )
