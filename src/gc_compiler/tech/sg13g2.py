"""IHP SG13G2 adapter (130 nm BiCMOS, 1.2 V LV devices; IHP-Open-PDK).

Verified against IHP-Open-PDK commit ``5e6d592e4002946a4616f798c357f0f3c06cf3b6``
(``ciel enable --pdk-family ihp-sg13g2 5e6d592...``): rule values from its
``libs.tech/klayout/tech/drc/rule_decks/sg13g2_tech_default.json`` (rule ids in the comments),
layers from its ``layers_def.drc`` and magic's ``ihp-sg13g2-cifin.tech``.

**The stack is shifted one metal up.** SG13G2 has no local interconnect: a contact (Cont) lands
on Metal1. The compiler's builders use a local-interconnect role between the contacts and their
first metal, so here that role *is* Metal1 and every routing role is one metal higher::

    role:  LICON  LI      MCON  M1      VIA1  M2      VIA2  M3      VIA3  M4      VIA4     M5
    layer: Cont   Metal1  Via1  Metal2  Via2  Metal3  Via3  Metal4  Via4  Metal5  TopVia1  TopMetal1

It costs a metal (Metal1 does sky130's li1's job), and nothing else: the builders only see roles.
The hard macro's power pins are on TopMetal1 (``power_pin_layer`` M5), the vertical layer of IHP's
place-and-route grid.

Other differences a reader of the sky130 adapter should know:

* diffusion and taps are one layer (Activ); n+ is Activ not under pSD. nSD (7/0) is drawn over n+
  anyway -- the decks ignore it (DRC reads it only for grid and angle checks, LVS and magic derive
  n+ without it), so the builders' n-implant role has a layer;
* no poly-contact marker, no threshold flavours (``flavors`` is just the nominal device);
* the isolated p-well of the negative-rail driver is nBuLay under an NWell ring (the deep n-well role);
* a GatPoly line over an n-tap is an NMOS gate (``policy.poly_over_well_is_device``);
* IHP ships its MOSFET models (PSP 103) as Verilog-A: :func:`osdi_modules` compiles them with
  OpenVAF on first use, and :func:`mismatch_library` turns IHP's per-instance mismatch switch on;
* the sign-off DRC is IHP's KLayout decks, magic is used for extraction only (``TOOLS``).

Measured on it (:data:`CHARACTERISED`): the refresh intervals of the default and plain cells, and
the controller's timing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

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

#: The IHP-Open-PDK commit this adapter is verified with.
PDK_VERSION = "5e6d592e4002946a4616f798c357f0f3c06cf3b6"

#: The Verilog-A models the MOSFET library uses (``psp103va``, and ``pspnqs103va`` for its RF
#: devices), as IHP's ``openvaf-compile-va.sh`` compiles them.
OSDI_SOURCES = ("psp103/psp103.va", "psp103/psp103_nqs.va")


def osdi_cache() -> Path:
    """Where compiled models are kept: ``$GC_CACHE`` or the project's ``build/``, per PDK version."""
    base = os.environ.get("GC_CACHE") or Path(__file__).resolve().parents[3] / "build"
    return Path(base) / "osdi" / f"ihp-sg13g2-{PDK_VERSION[:10]}"


def mismatch_library(corners: str | Path) -> Path:
    """IHP's corner library with local mismatch on for every device: its devices draw mismatch only
    where an instance passes ``mm_ok=1``, so the model file is copied (into :func:`osdi_cache`)
    with that as its default, and the corner file with its includes pointing at the copy and at
    the PDK's other files."""
    corners = Path(corners)
    models = corners.parent
    out = osdi_cache()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "cornerMOSlv_mm.lib"
    if target.exists():
        return target

    def absolute(text: str, own: dict[str, Path]) -> str:
        lines = []
        for line in text.splitlines():
            words = line.split()
            if len(words) == 2 and words[0].lower() == ".include":
                line = line.replace(words[1], str(own.get(words[1], models / words[1])))
            lines.append(line)
        return "\n".join(lines) + "\n"

    mod = out / "sg13g2_moslv_mod_mismatch.lib"
    text = (models / mod.name).read_text()
    if text.count("mm_ok=0") != 2:
        raise RuntimeError(f"{models / mod.name}: expected the NMOS's and PMOS's mm_ok=0 defaults")
    mod.write_text(absolute(text.replace("mm_ok=0", "mm_ok=1"), {}))
    partial = target.with_suffix(".tmp")
    partial.write_text(absolute(corners.read_text(), {mod.name: mod}))
    partial.replace(target)
    return target


def osdi_modules(source_dir: Path) -> list[Path]:
    """The OSDI libraries ngspice loads for the PSP models, compiled with OpenVAF from the PDK's
    Verilog-A sources on first use (IHP ships the sources only)."""
    out = osdi_cache()
    modules = [out / (Path(source).stem + ".osdi") for source in OSDI_SOURCES]
    missing = [(source, module) for source, module in zip(OSDI_SOURCES, modules) if not module.exists()]
    if missing:
        compiler = shutil.which("openvaf-r") or shutil.which("openvaf")
        if compiler is None:
            raise RuntimeError("OpenVAF is not on PATH (enter the nix shell): it compiles IHP's PSP models for ngspice")
        out.mkdir(parents=True, exist_ok=True)
        for source, module in missing:
            partial = module.with_suffix(".osdi.tmp")
            subprocess.run(
                [compiler, "-D__NGSPICE__", "-o", str(partial), str(Path(source_dir) / source)],
                check=True, capture_output=True, text=True,
            )
            partial.replace(module)  # whole or not at all, if two runs compile at once
    return modules

LAYERS: dict[LayerRole, Layer] = {
    LayerRole.NWELL: (31, 0),
    LayerRole.DNWELL: (32, 0),  # nBuLay: under an NWell ring, it isolates a p-well
    LayerRole.DIFF: (1, 0),  # Activ
    LayerRole.TAP: (1, 0),  # Activ: a tap is diffusion under the other implant
    LayerRole.POLY: (5, 0),  # GatPoly
    LayerRole.NIMP: (7, 0),  # nSD (ignored by the decks; n+ is implied)
    LayerRole.PIMP: (14, 0),  # pSD
    LayerRole.LICON: (6, 0),  # Cont
    LayerRole.LI: (8, 0),  # Metal1
    LayerRole.MCON: (19, 0),  # Via1
    LayerRole.M1: (10, 0),  # Metal2
    LayerRole.VIA1: (29, 0),  # Via2
    LayerRole.M2: (30, 0),  # Metal3
    LayerRole.VIA2: (49, 0),  # Via3
    LayerRole.M3: (50, 0),  # Metal4
    LayerRole.VIA3: (66, 0),  # Via4
    LayerRole.M4: (67, 0),  # Metal5
    LayerRole.VIA4: (125, 0),  # TopVia1
    LayerRole.M5: (126, 0),  # TopMetal1: the power pins, on the place-and-route grid's vertical layer
    LayerRole.BOUNDARY: (189, 0),  # prBoundary
}

#: Net labels: the ``.text`` datatype (25), the one LVS and magic read.
LABEL_LAYERS: dict[LayerRole, Layer] = {
    LayerRole.LI: (8, 25),
    LayerRole.M1: (10, 25),
    LayerRole.M2: (30, 25),
    LayerRole.M3: (50, 25),
    LayerRole.M4: (67, 25),
    LayerRole.M5: (126, 25),
    # poly and diffusion have no text datatype magic reads: it takes their labels from the
    # drawing layers (the word lines are poly, and extraction names them from these)
    LayerRole.POLY: (5, 0),
    LayerRole.DIFF: (1, 0),
}

PIN_LAYERS: dict[LayerRole, Layer] = {
    LayerRole.LI: (8, 2),
    LayerRole.M1: (10, 2),
    LayerRole.M2: (30, 2),
    LayerRole.M3: (50, 2),
    LayerRole.M4: (67, 2),
    LayerRole.M5: (126, 2),
}

STYLES: dict[LayerRole, LayerStyle] = {
    LayerRole.M5: LayerStyle("#ffd54f", "lightly hatched"),
    LayerRole.NWELL: LayerStyle("#c8a2c8", "lightly left-hatched"),
    LayerRole.DNWELL: LayerStyle("#8e6bb0", "dotted"),
    LayerRole.DIFF: LayerStyle("#4caf50", "solid"),
    LayerRole.POLY: LayerStyle("#e53935", "solid"),
    LayerRole.LICON: LayerStyle("#ffffff", "solid"),
    LayerRole.PIMP: LayerStyle("#ffb74d", "lightly right-hatched"),
    LayerRole.LI: LayerStyle("#1e88e5", "left-hatched"),
    LayerRole.MCON: LayerStyle("#ffffff", "solid"),
    LayerRole.M1: LayerStyle("#fb8c00", "right-hatched"),
    LayerRole.VIA1: LayerStyle("#ffffff", "solid"),
    LayerRole.M2: LayerStyle("#43a047", "cross-hatched"),
    LayerRole.VIA2: LayerStyle("#ffffff", "solid"),
    LayerRole.M3: LayerStyle("#8d6e63", "left-hatched"),
    LayerRole.VIA3: LayerStyle("#ffffff", "solid"),
    LayerRole.M4: LayerStyle("#7e57c2", "right-hatched"),
    LayerRole.BOUNDARY: LayerStyle("#9e9e9e", "hollow"),
}

_METALS = (LayerRole.M1, LayerRole.M2, LayerRole.M3, LayerRole.M4)  # Metal2..Metal5

SG13G2_RULES = DesignRules(
    grid=0.005,
    min_width={
        LayerRole.NWELL: 0.62,  # NW.a
        LayerRole.DNWELL: 1.0,  # NBL.a
        LayerRole.DIFF: 0.15,  # Act.a
        LayerRole.TAP: 0.15,
        LayerRole.POLY: 0.13,  # Gat.a
        LayerRole.NIMP: 0.31,  # (as pSD.a)
        LayerRole.PIMP: 0.31,  # pSD.a
        LayerRole.LI: 0.16,  # M1.a
        **{m: 0.20 for m in _METALS},  # Mn.a
        LayerRole.M5: 1.64,  # TM1.a
    },
    min_space={
        LayerRole.NWELL: 1.8,  # NW.b1: different nets (0.62 for the same net)
        LayerRole.DNWELL: 3.2,  # NBL.c: different nets (NBL.b 1.5 for the same net)
        LayerRole.DIFF: 0.21,  # Act.b
        LayerRole.TAP: 0.21,
        LayerRole.POLY: 0.18,  # Gat.b
        LayerRole.NIMP: 0.31,
        LayerRole.PIMP: 0.31,  # pSD.b
        LayerRole.LI: 0.18,  # M1.b
        **{m: 0.21 for m in _METALS},  # Mn.b
        LayerRole.M5: 1.64,  # TM1.b
    },
    enclosure={
        (LayerRole.DIFF, LayerRole.LICON): 0.07,  # Cnt.c
        (LayerRole.TAP, LayerRole.LICON): 0.07,
        (LayerRole.POLY, LayerRole.LICON): 0.07,  # Cnt.d
        (LayerRole.LI, LayerRole.LICON): 0.05,  # M1.c1 endcap (M1.c: 0 on the other pair)
        (LayerRole.LI, LayerRole.MCON): 0.05,  # V1.c1 (V1.c 0.01)
        (LayerRole.M1, LayerRole.MCON): 0.05,  # M2.c1 (Mn.c 0.005)
        (LayerRole.M1, LayerRole.VIA1): 0.05,  # Vn.c1 (Vn.c 0.005)
        (LayerRole.M2, LayerRole.VIA1): 0.05,
        (LayerRole.M2, LayerRole.VIA2): 0.05,
        (LayerRole.M3, LayerRole.VIA2): 0.05,
        (LayerRole.M3, LayerRole.VIA3): 0.05,
        (LayerRole.M4, LayerRole.VIA3): 0.05,
        (LayerRole.M4, LayerRole.VIA4): 0.10,  # TV1.c
        (LayerRole.M5, LayerRole.VIA4): 0.42,  # TV1.d
        (LayerRole.NWELL, LayerRole.DIFF): 0.31,  # NW.c
        (LayerRole.NWELL, LayerRole.TAP): 0.24,  # NW.e
        # a device's implant reaches pSD.i's 0.30 past its gate (pSD.c asks 0.18 of the diffusion;
        # the stricter one on every side), and the implants never overlap, so pSD also keeps
        # pSD.j's 0.30 from an NFET gate and pSD.d's 0.18 from n+ Activ
        (LayerRole.NIMP, LayerRole.DIFF): 0.30,
        (LayerRole.PIMP, LayerRole.DIFF): 0.30,  # pSD.i, pSD.j
        (LayerRole.NIMP, LayerRole.TAP): 0.18,
        (LayerRole.PIMP, LayerRole.TAP): 0.18,
    },
    relaxed_enclosure={
        (LayerRole.LI, LayerRole.LICON): 0.0,  # M1.c
        (LayerRole.LI, LayerRole.MCON): 0.01,  # V1.c
        (LayerRole.M1, LayerRole.MCON): 0.005,  # Mn.c
        (LayerRole.M1, LayerRole.VIA1): 0.005,  # Vn.c
        (LayerRole.M2, LayerRole.VIA1): 0.005,
        (LayerRole.M2, LayerRole.VIA2): 0.005,
        (LayerRole.M3, LayerRole.VIA2): 0.005,
        (LayerRole.M3, LayerRole.VIA3): 0.005,
        (LayerRole.M4, LayerRole.VIA3): 0.005,
        # a tap strip's implant may end close to its end: pSD.c1 (p+ Activ in the p-well), and
        # nSD as pSD (the decks derive n+ as Activ outside pSD, but the two are kept alike)
        (LayerRole.NIMP, LayerRole.TAP): 0.03,
        (LayerRole.PIMP, LayerRole.TAP): 0.03,  # pSD.c1
    },
    cut_size={
        LayerRole.LICON: 0.16,  # Cnt.a (exact)
        LayerRole.MCON: 0.19,  # V1.a (exact)
        LayerRole.VIA1: 0.19,  # Vn.a
        LayerRole.VIA2: 0.19,
        LayerRole.VIA3: 0.19,
        LayerRole.VIA4: 0.42,  # TV1.a
    },
    cut_space={
        LayerRole.LICON: 0.18,  # Cnt.b
        LayerRole.MCON: 0.22,  # V1.b
        LayerRole.VIA1: 0.22,  # Vn.b
        LayerRole.VIA2: 0.22,
        LayerRole.VIA3: 0.22,
        LayerRole.VIA4: 0.42,  # TV1.b
    },
    poly_endcap=0.18,  # Gat.c
    gate_to_cut_space=0.11,  # Cnt.f
    gate_to_gate_space=0.18,  # Gat.b
    diff_extension=0.23,  # Act.c
    device_min_width=0.15,  # Act.a
    device_min_length=0.13,  # Gat.a1/a2
    # NW.d is 0.31, but the isolated p-well's n+ Activ keeps 0.39 from its NWell ring (nmosi.c):
    # the stricter one everywhere
    nwell_to_ndiff_space=0.39,
    # the isolated p-well: NWell over nBuLay's edge as a ring, at least 0.62 of it over the
    # nBuLay (nmosi.d), whose edge keeps 1.24 from the p-well's Activ (nmosi.b: 1.0 + 0.39), and
    # reaching 0.7 past it so n+ Activ outside the ring (0.39 from it) keeps NBL.e's 1.0. The
    # decks also generate nBuLay 1.0 inside any NWell wider than 3 um (the level shifter's, which
    # the ring's wall joins): the drawn one starts there too, so the two meet (NBL.b)
    isolation_ring_inner=1.0,
    isolation_ring_outer=0.70,
    isolation_nwell_space=2.2,  # NBL.d
    isolation_to_pdiff_space=0.5,  # NBL.f
    poly_contact_to_diff=0.14,  # Cnt.e
    poly_contact_to_pdiff=0.14,
    poly_to_diff=0.07,  # Gat.d
    nwell_to_ptap_space=0.24,  # NW.f
    latchup_distance=20.0,  # LU.a/LU.b
    min_area={
        LayerRole.DIFF: 0.122,  # Act.d
        LayerRole.TAP: 0.122,
        LayerRole.POLY: 0.09,  # Gat.e
        LayerRole.PIMP: 0.25,  # pSD.k
        LayerRole.NIMP: 0.25,
        LayerRole.LI: 0.09,  # M1.d
        **{m: 0.144 for m in _METALS},  # Mn.d
    },
)

MODELS: dict[tuple[DeviceKind, str | None], str] = {
    (DeviceKind.NMOS, None): "sg13_lv_nmos",
    (DeviceKind.PMOS, None): "sg13_lv_pmos",
}

STACK = MetalStack(
    routing=(LayerRole.LI, LayerRole.M1, LayerRole.M2, LayerRole.M3, LayerRole.M4, LayerRole.M5),
    cuts={
        (LayerRole.LI, LayerRole.M1): LayerRole.MCON,
        (LayerRole.M1, LayerRole.M2): LayerRole.VIA1,
        (LayerRole.M2, LayerRole.M3): LayerRole.VIA2,
        (LayerRole.M3, LayerRole.M4): LayerRole.VIA3,
        (LayerRole.M4, LayerRole.M5): LayerRole.VIA4,
    },
    local_interconnect=LayerRole.LI,  # Metal1, in the shifted stack
    contact=LayerRole.LICON,
    lef_names={
        LayerRole.LI: "Metal1", LayerRole.M1: "Metal2", LayerRole.M2: "Metal3", LayerRole.M3: "Metal4",
        LayerRole.M4: "Metal5", LayerRole.M5: "TopMetal1",
    },
    power_pin_layer=LayerRole.M5,
)

ELECTRICAL = Electrical(
    vdd_nominal=1.2,
    device_voltage_rating=1.32,  # the LV devices' maximum (the fast library's 1.32 V)
    corners=(
        Corner("tt", 1.2, 25.0), Corner("ss", 1.08, 125.0), Corner("ff", 1.32, -40.0),
        Corner("sf", 1.2, 25.0), Corner("fs", 1.2, 25.0),
    ),
    # approximate (the timing models only; nothing is characterised on this technology yet)
    sheet_resistance={LayerRole.POLY: 7.0, LayerRole.M3: 0.088},
    strap_contact_resistance=30.0,
    gate_cap_per_um2=12e-15,
)

TOOLS = ToolSetup(
    variant="ihp-sg13g2",
    klayout_deck="libs.tech/klayout/tech/drc/ihp-sg13g2.drc",
    magicrc="libs.tech/magic/ihp-sg13g2.magicrc",
    netgen_setup="libs.tech/netgen/ihp-sg13g2_setup.tcl",
    # the main table (density is a table of its own, not run: a macro is not filled), then the
    # "extra" rules IHP's runner adds by default (the implant rules pSD.c/d/i/j among them)
    klayout_deck_vars={"run_mode": "flat", "threads": "8"},
    klayout_top_cell_var="topcell",
    klayout_extra_decks=("libs.tech/klayout/tech/drc/rule_decks/sg13g2_maximal.drc",),
    # KLayout is the sign-off; IHP's (beta) magic deck applies the contact-bar rules CntB.b/h1 to
    # square contacts, and its one rule the KLayout decks lacked (pSD.i) is in the extra deck
    magic_drc=False,
    magic_drc_style="drc(full)",
    context_waivers=("LU.a", "LU.b"),
    leaf_waivers=("NW.c", "NW.e", "NW.f"),
)

POLICY = LayoutPolicy(
    tap_row_interval=4,
    strap_column_interval=16,
    # TopMetal1 stripes, as wide as IHP's grid's (2.2): the grid's TopMetal2 straps land on them
    # through TopVia2 (0.9, enclosed 0.5 by TopMetal1: TV2.a/c), and 2.0 apart (TM1.b 1.64)
    power_stripe=PowerStripe(width=2.2, pitch=25.0, clearance=2.0, site_pitch=3.0, scan_step=0.1),
    join=0.05,
    margin=0.05,
    poly_over_well_is_device=True,  # GatPoly over an n-tap is an NMOS gate
)

SIZING = Sizing(
    # LV devices at a little over the minimums (W 0.15, L 0.13). The default cell's mr is 1.0 um
    # long: at ss/125 C on the extracted 16x20 macro a written '0' then lasts 121 us before it
    # misreads, against 35 us with mr at the minimum length (the plain cell), for 28% more area
    # (0.5 um: 52 us, 8%; a longer write device adds nothing)
    bitcell={
        "write_width": 0.3, "write_length": 0.13, "read_width": 0.3, "read_length": 0.13,
        "mr_length": 1.0, "write_flavor": None,
    },
    plain_bitcell={"mr_length": None},
    logic={"pmos_width": 0.6, "nmos_width": 0.3, "length": 0.13},
    predecode={"pmos_width": 2.4, "nmos_width": 1.2},
    level_shifter={"pmos_width": 1.0, "latch_width": 0.3, "latch_length": 0.6, "input_length": 0.13},
    size_grid=0.01,
    wide_array_cols=64,
    wide_driver_strength=4.0,
    # --neg-wl's level: vdd - vneg across a write's gate oxide at the LV devices' 1.32 V rating
    write_wl_low=-0.12,
)


#: Measured on SG13G2 (``measurements/data/retention_mc_ss_sg13g2.json``): the worst of 20 mismatch dies at
#: ss, 1.08 V, 125 C on the extracted 16x20 macro, as on sky130; the controller's timing from the
#: macro's five-corner characterisation (``ControllerTiming.from_characterisation``: read 7.9 ns,
#: write 1.1, precharge 0.64), rounded up as sky130's are, the write to the 10 ns a '0' takes to
#: settle (below). The controller's own share of a read
#: (``sample_ns``) from the routed 8x8 blocks' static timing at ss: 1.53 ns past ``read_ns`` one
#: bank (the macro arc's 5% late derate 0.40, re's flop 0.42, the input mux 0.24, setup 0.19, clock
#: uncertainty 0.25 and skew), 1.79 ns banked (a wider mux); the hold budget (``macro_min_ns``) under the fastest output, a 4x4
#: macro's dout 0.57 ns after re at ff; dout trails rdy by at most 0.2 ns (``rdy_lag_ns`` kept at
#: sky130's 1.0). The timing library's pin figures are sky130's.
CHARACTERISED = Characterised(
    refresh_intervals=(
        RefreshInterval({}, 100.0),  # the default cell (mr 1.0 um); nominal 121 us
        RefreshInterval({"mr_length": None}, 26.9),  # the plain cell; nominal 35 us
        # with --neg-wl's -0.12 V rail (measurements/data/retention_mc_neg_sg13g2.json): the '0' written
        # 0.30 / 0.26 V instead of ~0.4, the leakage at 125 C unchanged
        RefreshInterval({}, 120.9, negative_rail=-0.12),  # nominal 143 us
        RefreshInterval({"mr_length": None}, 37.8, negative_rail=-0.12),  # nominal 42 us
    ),
    unmeasured_refresh_us=26.9,  # the shortest measured
    strobe_factor=1.25,  # dout is valid by 0.87 x rdy at every corner
    controller=ControllerDefaults(
        read_ns=8.0, write_ns=10.0, precharge_ns=2.0, sample_ns=2.0, macro_min_ns=0.5, rdy_lag_ns=1.0,
        timeout_ns=50.0,
        # a '0' through the PMOS write device, its word line only 0.12 V below ground, takes ~10 ns
        # to settle (its crossing is 1.1 ns): a 2 ns write -- or a refresh's write-back -- leaves it
        # far above the level the refresh intervals start from
        write_settle_ns=10.0,
    ),
    idle_storage_node_v=0.4,
    decayed_one_fraction=0.7,
    macro_input_pf=0.005,
    macro_output_transition_ns=0.2,
    # a '1' decayed to 0.7 vdd, refreshed (the level the access leaves, the worse bank; restored is
    # 0.95 vdd): 2 x 16x20 at ss (125 C) brings it back to 1.06 V (1.03 V needed), 2 x 32x32 only
    # to 1.138 V already at tt (1.14 V), 2 x 64x64 1.11 V, 2 x 128x128 0.78 V (at ss it misreads it)
    refresh_max_rows=16,
)


class SG13G2(Technology):
    """IHP SG13G2 (1.2 V LV devices), its metal stack shifted one up (see the module docstring)."""

    name = "ihp-sg13g2"
    cell_prefix = "ihp_"
    rules = SG13G2_RULES
    styles = STYLES
    stack = STACK
    electrical = ELECTRICAL
    tools = TOOLS
    policy = POLICY
    sizing = SIZING
    characterised = CHARACTERISED
    flavors = (None,)

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
        return LayerRole.PIMP if kind is DeviceKind.NMOS else LayerRole.NIMP

    def spice_model(self, kind: DeviceKind, flavor: str | None = None) -> str:
        return MODELS[(kind, flavor)]

    def spice_device(self, ref, drain, gate, source, bulk, kind, width, length, flavor=None) -> str:
        """IHP's models take metres: sizes carry the ``u`` suffix."""
        model = self.spice_model(kind, flavor)
        return f"X{ref} {drain} {gate} {source} {bulk} {model} w={width:g}u l={length:g}u"

    def model_library(self, root: str | None = None) -> str:
        return str(self.tools.path("libs.tech/ngspice/models/cornerMOSlv.lib", root))

    def model_header(self, process: str, root: str | None = None, mismatch: bool = False) -> list[str]:
        """Load the PSP models' OSDI modules (compiled on first use, :func:`osdi_modules`), then
        ``cornerMOSlv.lib``'s ``mos_<corner>`` section."""
        return [
            ".control",
            *(f"pre_osdi {module}" for module in osdi_modules(self.tools.path("libs.tech/verilog-a", root))),
            ".endc",
            f'.lib "{mismatch_library(self.model_library(root)) if mismatch else self.model_library(root)}" '
            f'mos_{process}{"_mismatch" if mismatch else ""}',
        ]

    def standard_cells(self, root: str | None = None) -> StdCellLibrary:
        """``sg13g2_stdcell`` (not used by milestone 1: the hard-block flow needs power pins on
        IHP's TopMetal grid)."""
        pdk_root = self.tools.pdk_root(root)
        ref = pdk_root / self.tools.variant / "libs.ref" / "sg13g2_stdcell"
        corners = {"tt": "typ_1p20V_25C", "ss": "slow_1p08V_125C", "ff": "fast_1p32V_m40C"}
        return StdCellLibrary(
            name="sg13g2_stdcell",
            pdk=self.tools.variant,
            pdk_root=str(pdk_root),
            corners=corners,
            liberty={p: str(ref / "lib" / f"sg13g2_stdcell_{c}.lib") for p, c in corners.items()},
            verilog=(str(ref / "verilog" / "sg13g2_udp.v"), str(ref / "verilog" / "sg13g2_stdcell.v")),
            verilog_defines=("FUNCTIONAL",),
            conditions={"tt": (1.2, 25.0), "ss": (1.08, 125.0), "ff": (1.32, -40.0)},
            site_width=0.48,
            row_height=3.78,
            power_pin="VDD",
            ground_pin="VSS",
            pdn=PowerGrid(
                vertical_layer="TopMetal1", horizontal_layer="TopMetal2",
                # denser than IHP's default 75.6 um, so straps cross a macro a few tens of um tall
                # and land on its TopMetal1 stripes, and several vertical pairs reach a small
                # block's standard-cell rows (a vdd/vss pair needs 8.4 um)
                horizontal_pitch=20.0, horizontal_offset=5.0, vertical_pitch=30.0,
                # pdngen stops the TopMetal1 straps at a macro: one of each net is put on the
                # macro's own stripes, which carry it across (IHP's 2.2 um straps, TM1.b 1.64)
                align_to_macro=True, vertical_width=2.2, vertical_min_spacing=1.64, grid=0.005,
            ),
            rc_corner="nom",
            # 77 um2 a flop synthesized (43 flops, 3308 um2 on an 8x8 controller), before the
            # clock tree and repair buffers: at 60 (sky130's) the band overfilled in CTS
            crossing_nets_per_um=2.2,  # 8x64 banked routed with 128 such nets in 57 um (2.3 a um)
            area_per_flop_um2=95.0,  # 83-89 um2 a flop or latch on routed controllers (8x8 to 8x64)
            # (measurements/data/controller_area_block.json, routed blocks) a 4x clock-tree root, not
            # the 16x: a controller of some 50 flops needs no more; repeaters on any net longer than
            # 300 um (a wide banked macro's din and dout reach both banks: antenna and slew)
            flow_config={"CTS_ROOT_BUFFER": "sg13g2_buf_4", "DESIGN_REPAIR_MAX_WIRE_LENGTH": 300},
            sdc=SdcConstants(
                setup_uncertainty=0.25, hold_uncertainty=0.1, clock_transition=0.15,
                derate_early=0.95, derate_late=1.05, input_transition=0.1, output_load=0.02,
            ),
        )
