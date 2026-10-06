from __future__ import annotations

import shutil

import klayout.db as kdb
import pytest

from gc_compiler.config import ArrayConfig
from gc_compiler.netlist.decoder import (
    address_bits,
    all_row_outputs,
    banked_face_outputs,
    banked_outputs,
    gate_subckts,
    logical_row,
    nand2_subckt,
    nand3_subckt,
    nor2_subckt,
    nor3_subckt,
    decoder_gates,
    split_address,
    row_decoder_pins,
    row_decoder_subckt,
    side_outputs,
)
from gc_compiler.netlist.decoder import _CELL_OF
from gc_compiler.netlist.spice import inverter_subckt, macro_subckt
from gc_compiler.periphery.decoder import _plan, row_decoder
from gc_compiler.periphery.logic import LogicConfig
from gc_compiler.periphery.standard_cells import standard_cell, standard_cell_row
from gc_compiler.tech.rules import LayerRole
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.verify.drc import LEAF_RULES, default_deck, run_drc
from gc_compiler.verify.lvs import default_magicrc, default_netgen_setup, lvs_layout_vs_netlist
from gc_compiler.verify.sim import default_model_lib, run_decoder_op


@pytest.fixture(scope="session")
def tech() -> Sky130:
    return Sky130()


def test_address_bits():
    assert address_bits(2) == 1
    assert address_bits(4) == 2
    assert address_bits(5) == 3
    assert address_bits(8) == 3
    assert address_bits(9) == 4


def test_split_address_uses_the_fewest_gates():
    for bits in range(1, 8):
        outputs = all_row_outputs(1 << bits)
        low = split_address(bits, outputs)
        assert 1 <= low <= bits
        cfg = ArrayConfig.basic(rows=1 << bits, cols=2) if bits > 1 else ArrayConfig.basic(rows=2, cols=2)
        gates = len(decoder_gates(cfg, outputs))
        # every other split is at least as big
        from gc_compiler.netlist.decoder import _gates

        assert all(len(_gates(bits, other, outputs)) >= gates for other in range(1, bits + 1))


def test_every_row_gate_is_a_single_two_input_nand():
    """The property that lets the row cells sit at the word-line pitch."""
    cfg = ArrayConfig.basic(rows=32, cols=4)
    for gate in decoder_gates(cfg):
        if gate.role in ("row_nand_r", "row_nand_w"):
            assert gate.kind == "nand"
    outs = {g.output for g in decoder_gates(cfg) if g.role.startswith("row")}
    assert len(outs) == 2 * cfg.rows


def _simulate(subckt_text: str, known: dict[str, int]) -> dict[str, int]:
    """Evaluate a flat subckt of gc_inv/gc_nand2 instances to a fixed point.

    This is a purely combinational digital simulation (no analog/timing behaviour), used
    to check the decoder's logic independently of the generator code that produced it.
    """
    lines = [
        line.strip()
        for line in subckt_text.splitlines()
        if line.strip() and not line.strip().startswith(("*", ".subckt", ".ends"))
    ]
    instances = []
    for line in lines:
        tokens = line.split()
        instances.append((tokens[-1], tokens[1:-1]))

    values = dict(known)
    values["vdd"] = 1
    values["vss"] = 0
    changed = True
    while changed:
        changed = False
        for kind, nets in instances:
            if kind == "gc_inv":
                a, y, _vdd, _vss = nets
                if a in values and y not in values:
                    values[y] = 1 - values[a]
                    changed = True
            elif kind in ("gc_nand2", "gc_nand3", "gc_nor2", "gc_nor3"):
                *ins, y, _vdd, _vss = nets
                if all(net in values for net in ins) and y not in values:
                    if kind.startswith("gc_nand"):
                        values[y] = 0 if all(values[net] for net in ins) else 1
                    else:
                        values[y] = 0 if any(values[net] for net in ins) else 1
                    changed = True
    return values


@pytest.mark.parametrize("rows", [2, 4, 8, 16, 32, 64])
def test_row_decoder_selects_exactly_one_row(tech, rows):
    """Exercises every predecode shape: a single low block, a low and a high block, and
    multi-level trees for three or more bits per block.
    """
    cfg = ArrayConfig.basic(rows=rows, cols=4)
    subckt = row_decoder_subckt(tech, cfg, LogicConfig())
    bits = address_bits(rows)

    for row in range(rows):
        addr = {f"addr_{i}": (row >> i) & 1 for i in range(bits)}
        write = _simulate(subckt, {**addr, "we": 1, "re": 0})
        assert [r for r in range(rows) if write[f"wwl_in_{r}"] == 1] == [row]
        assert all(write[f"rwl_in_{r}"] == 1 for r in range(rows))

        read = _simulate(subckt, {**addr, "we": 0, "re": 1})
        assert [r for r in range(rows) if read[f"rwl_in_{r}"] == 0] == [row]
        assert all(read[f"wwl_in_{r}"] == 0 for r in range(rows))

        idle = _simulate(subckt, {**addr, "we": 0, "re": 0})
        assert all(idle[f"wwl_in_{r}"] == 0 for r in range(rows))
        assert all(idle[f"rwl_in_{r}"] == 1 for r in range(rows))


@pytest.mark.parametrize("rows", [2, 6, 8, 12, 32])
def test_banked_decoder_selects_exactly_one_row_of_one_bank(tech, rows):
    """One predecode for two banks: the bank is the address bit above the row bits, so for a
    non-power-of-two bank (6, 12 rows) the logical addresses between the banks decode to
    nothing -- and every address that does exist selects one row of one bank."""
    cfg = ArrayConfig.basic(rows=rows, cols=4)
    subckt = row_decoder_subckt(tech, cfg, LogicConfig(), banked=True)
    bits = address_bits(rows)
    outs = [(bank, r) for bank in (0, 1) for r in range(rows)]

    for bank, row in outs:
        addr = {f"addr_{i}": (row >> i) & 1 for i in range(bits)} | {f"addr_{bits}": bank}
        write = _simulate(subckt, {**addr, "we": 1, "re": 0})
        assert [(b, r) for b, r in outs if write[f"wwl_in_{b}_{r}"] == 1] == [(bank, row)]
        assert all(write[f"rwl_in_{b}_{r}"] == 1 for b, r in outs)

        read = _simulate(subckt, {**addr, "we": 0, "re": 1})
        assert [(b, r) for b, r in outs if read[f"rwl_in_{b}_{r}"] == 0] == [(bank, row)]
        assert all(read[f"wwl_in_{b}_{r}"] == 0 for b, r in outs)

        idle = _simulate(subckt, {**addr, "we": 0, "re": 0})
        assert all(idle[f"wwl_in_{b}_{r}"] == 0 and idle[f"rwl_in_{b}_{r}"] == 1 for b, r in outs)


def test_banked_faces_cover_every_output_once():
    """Each bank's near and far word lines together are every word line of that bank, once."""
    rows = 12
    everything = []
    for bank in (0, 1):
        near, far = banked_face_outputs(rows, bank)
        assert len(near) == len(far) == rows
        everything += near + far
    assert sorted(everything) == sorted(banked_outputs(rows))
    # bank 0's near side is the right driver column, bank 1's is the left one
    assert banked_face_outputs(rows, 0)[0][0] == (logical_row(rows, 0, 0), side_outputs(rows)[1][0][1])
    assert banked_face_outputs(rows, 1)[0][0] == (logical_row(rows, 1, 0), side_outputs(rows)[0][0][1])
    assert row_decoder_pins(ArrayConfig.basic(rows=rows, cols=2), banked=True)[: address_bits(rows) + 3] == [
        *(f"addr_{i}" for i in range(address_bits(rows) + 1)), "we", "re",
    ]


def test_row_decoder_pins_match_macro_expectations():
    cfg = ArrayConfig.basic(rows=8, cols=4)
    pins = row_decoder_pins(cfg)
    assert pins[:3] == ["addr_0", "addr_1", "addr_2"]
    assert "we" in pins and "re" in pins
    assert "wwl_in_5" in pins and "rwl_in_5" in pins


@pytest.mark.parametrize("rows", [2, 4, 6, 8, 16, 32, 64])
@pytest.mark.parametrize("side", ["full", "left", "right"])
def test_physical_plan_matches_schematic(tech, rows, side):
    cfg = ArrayConfig.basic(rows=rows, cols=4)
    left, right = side_outputs(rows)
    outputs = {"full": all_row_outputs(rows), "left": left, "right": right}[side]
    plan = _plan(cfg, outputs)
    layout_netlist = "\n".join(
        f"Xgate{index} {' '.join(gate.pins.values())} vdd vss "
        f"{ _CELL_OF[gate.kind] }"
        for index, gate in enumerate(plan.gates)
    )
    schematic = row_decoder_subckt(tech, cfg, outputs=outputs)
    bits = address_bits(rows)
    for address in range(1 << bits):
        for write_enable, read_enable in ((0, 0), (0, 1), (1, 0), (1, 1)):
            inputs = {f"addr_{bit}": (address >> bit) & 1 for bit in range(bits)}
            inputs.update(we=write_enable, re=read_enable)
            actual = _simulate(layout_netlist, inputs)
            expected = _simulate(schematic, inputs)
            for row, signal in outputs:
                net = f"{signal}_{row}"
                enable = write_enable if signal == "wwl_in" else read_enable
                selected = int(address == row and enable)
                level = selected if signal == "wwl_in" else 1 - selected
                assert actual[net] == expected[net] == level


@pytest.mark.parametrize("rows", [2, 6, 32, 64])
def test_decoder_and_drivers_select_correct_device_polarity(tech, rows):
    cfg = ArrayConfig.basic(rows=rows, cols=4)
    logic = LogicConfig()
    decoders = [row_decoder_subckt(tech, cfg, outputs=side) for side in side_outputs(rows)]
    drivers = macro_subckt(tech, cfg, logic, "macro", "array")
    netlist = "\n".join([*decoders, drivers])
    bits = address_bits(rows)
    for address in range(1 << bits):
        for write_enable, read_enable in ((0, 0), (0, 1), (1, 0), (1, 1)):
            inputs = {f"addr_{bit}": (address >> bit) & 1 for bit in range(bits)}
            inputs.update(we=write_enable, re=read_enable)
            values = _simulate(netlist, inputs)
            assert [row for row in range(rows) if values[f"wwl_{row}"] == 0] == (
                [address] if write_enable and address < rows else []
            )
            assert [row for row in range(rows) if values[f"rwl_{row}"] == 1] == (
                [address] if read_enable and address < rows else []
            )


def test_decoder_cache_distinguishes_output_subsets(tech):
    cfg = ArrayConfig.basic(rows=4, cols=4)
    left_outputs, right_outputs = side_outputs(cfg.rows)
    left = row_decoder(tech, cfg, outputs=left_outputs)
    right = row_decoder(tech, cfg, outputs=right_outputs)
    assert left is not right
    assert left.name != right.name
    assert {port.name for port in left.ports} == set(row_decoder_pins(cfg, left_outputs))
    assert {port.name for port in right.ports} == set(row_decoder_pins(cfg, right_outputs))
    assert row_decoder(tech, cfg, outputs=list(left_outputs)) is left


def test_decoder_dimensions_include_routing(tech):
    decoder = row_decoder(tech, ArrayConfig.basic(rows=4, cols=4))
    assert decoder.info["width"] == pytest.approx(decoder.dbbox().width())
    assert decoder.info["height"] == pytest.approx(decoder.dbbox().height())
    # Every net owns at least one track; a net with several (spanning channels, or split
    # into one per side) is tied together by exactly one trunk.
    assert decoder.info["track_count"] >= decoder.info["net_count"]
    assert decoder.info["track_count"] - decoder.info["net_count"] >= decoder.info["tie_count"]


def test_outputs_land_on_the_word_line_pitch(tech):
    """Built alone, output ``k`` sits exactly ``k`` array row pitches from the first: the row
    cells are placed at the driver inputs, not packed."""
    from gc_compiler.cells.bitcell_3t import bitcell_geometry

    cfg = ArrayConfig.basic(rows=32, cols=4)
    pitch = bitcell_geometry(tech, cfg.bitcell).pitch_y
    outputs = side_outputs(32)[0]
    decoder = row_decoder(tech, cfg, outputs=outputs)
    for k, (row, signal) in enumerate(outputs):
        assert decoder.ports[f"{signal}_{row}"].dcenter[0] == pytest.approx(0.5 + k * pitch)
    # about as wide as the array is tall, and shallow -- the level-strip design was ~230 um deep
    assert decoder.info["width"] < 32 * pitch * 1.3
    assert decoder.info["height"] < 45


@pytest.mark.parametrize("rows", [4, 16, 64])
@pytest.mark.parametrize("side", [0, 1])
def test_row_neighbours_abut_or_keep_diffusion_clear(tech, rows, side):
    """The inverters share a rail with the NAND strip, so a gap between two NANDs (or between
    a NAND and a neighbour's inverter) must be zero or at least the diffusion spacing."""
    from gc_compiler.periphery.decoder import _layout

    outputs = side_outputs(rows)[side]
    layout = _layout(tech, ArrayConfig.basic(rows=rows, cols=4), LogicConfig(), outputs, ())
    space = tech.rules.s(LayerRole.DIFF)
    nands = sorted((x, x + layout.cells[g.kind].info["width"]) for g, x in zip(layout.strips[1], layout.strip_x[1]))
    for (_, end), (start, _) in zip(nands, nands[1:]):
        assert abs(start - end) < 1e-6 or start - end >= space
    invs = sorted((x, x + layout.cells[g.kind].info["width"]) for g, x in zip(layout.strips[0], layout.strip_x[0]))
    for left, right in zip(invs, invs[1:]):
        assert right[0] - left[1] >= space
    for inv_start, inv_end in invs:
        for nand_start, nand_end in nands:
            gap = max(inv_start - nand_end, nand_start - inv_end)  # negative when they overlap in x
            assert gap <= 1e-6 or gap >= space


def test_predecode_levels_share_strips(tech):
    """Logic levels are packed into as few strips as fit along the rows, instead of one strip
    per level: a 64-row half-decoder's three levels fit one strip beside its two row strips."""
    from gc_compiler.periphery.decoder import _layout, _predecode_levels

    cfg = ArrayConfig.basic(rows=64, cols=4)
    layout = _layout(tech, cfg, LogicConfig(), side_outputs(64)[0], ())
    assert len(_predecode_levels(layout.plan)) == 3
    assert len(layout.strips) == 3  # write inverters, row NANDs, and one predecode strip
    unpacked = _layout(tech, cfg, LogicConfig(), side_outputs(64)[0], (), pack=False)
    assert len(unpacked.strips) == 5  # one strip per level


def test_decoder_uses_no_metal3(tech, tmp_path):
    """Routing channels have no metal1 in them, so tracks sit on it and metal3 stays free."""
    import klayout.db as kdb

    cfg = ArrayConfig.basic(rows=16, cols=4)
    decoder = row_decoder(tech, cfg, outputs=side_outputs(16)[1])
    layout = kdb.Layout()
    layout.read(str(decoder.write_gds(tmp_path / "decoder.gds")))
    cell = layout.top_cell()
    for role in (LayerRole.M3, LayerRole.VIA2):
        index = layout.layer(*tech.layer(role))
        assert kdb.Region(cell.begin_shapes_rec(index)).is_empty(), role
    assert not kdb.Region(cell.begin_shapes_rec(layout.layer(*tech.layer(LayerRole.M1)))).is_empty()


physical_tools_available = (
    all(shutil.which(tool) for tool in ("klayout", "magic", "netgen"))
    and default_deck().exists()
    and default_magicrc().exists()
    and default_netgen_setup().exists()
)


@pytest.mark.skipif(not physical_tools_available, reason="Sky130 DRC/LVS tools are missing")
@pytest.mark.parametrize("tight", [False, True])
@pytest.mark.parametrize("kind", ["inv", "nand2", "nand3", "nor2", "nor3"])
@pytest.mark.parametrize("logic", [LogicConfig(), LogicConfig(pmos_width=0.84, nmos_width=0.56)])
def test_standard_cell_drc_lvs(tech, tmp_path, kind, logic, tight):
    cell = standard_cell(tech, kind, logic, tight=tight)
    gds = cell.write_gds(tmp_path / f"{kind}.gds")
    drc = run_drc(gds, waive=LEAF_RULES)
    assert drc.clean, str(drc)
    reference = tmp_path / "reference.spice"
    reference.write_text(
        {
            "inv": inverter_subckt, "nand2": nand2_subckt, "nand3": nand3_subckt,
            "nor2": nor2_subckt, "nor3": nor3_subckt,
        }[kind](tech, logic)
    )
    result = lvs_layout_vs_netlist(gds, cell.name, reference, f"gc_{kind}", workdir=tmp_path / "lvs")
    assert result.matched, str(result)


@pytest.mark.parametrize("logic", [LogicConfig(), LogicConfig(pmos_width=0.84, nmos_width=0.56)])
def test_standard_cell_row_contract(tech, logic):
    cells = [standard_cell(tech, kind, logic) for kind in ("inv", "nand2")]
    assert cells[0].info["height"] == cells[1].info["height"]
    assert cells[0].info["width"] < cells[1].info["width"]
    for cell in cells:
        assert cell.dbbox().left == pytest.approx(0)
        assert cell.dbbox().bottom == pytest.approx(0)
        assert cell.dbbox().width() == pytest.approx(cell.info["width"])
        assert cell.dbbox().height() == pytest.approx(cell.info["height"])
        assert cell.info["width"] / cell.info["site_width"] == pytest.approx(round(cell.info["width"] / cell.info["site_width"]))
    for net in ("vdd", "vss"):
        assert cells[0].ports[net].dcenter[1] == cells[1].ports[net].dcenter[1]


@pytest.mark.skipif(not physical_tools_available, reason="Sky130 DRC/LVS tools are missing")
@pytest.mark.parametrize("kinds", [("inv", "inv"), ("nand2", "nand2"), ("inv", "nand2", "inv")])
def test_standard_cell_row_abutment(tech, tmp_path, kinds):
    row = standard_cell_row(tech, kinds)
    gds = row.write_gds(tmp_path / "row.gds")
    drc = run_drc(gds, waive=LEAF_RULES)
    assert drc.clean, str(drc)
    layout = kdb.Layout()
    layout.read(str(gds))
    region = kdb.Region(layout.top_cell().begin_shapes_rec(layout.layer(*tech.layer(LayerRole.M1)))).merged()
    for net in ("vdd", "vss"):
        rail_y = round(row.ports[net].dcenter[1] / layout.dbu)
        start = kdb.Point(1, rail_y)
        end = kdb.Point(round(row.info["width"] / layout.dbu) - 1, rail_y)
        assert any(polygon.inside(start) and polygon.inside(end) for polygon in region.each())
    pins = []
    instances = []
    for index, kind in enumerate(kinds):
        signals = ("a", "y") if kind == "inv" else ("a", "b", "y")
        nets = [f"{signal}_{index}" for signal in signals]
        pins.extend(nets)
        instances.append(f"Xgate{index} {' '.join(nets)} vdd vss gc_{kind}")
    reference = tmp_path / "row.spice"
    reference.write_text("\n".join([
        inverter_subckt(tech, LogicConfig()), nand2_subckt(tech, LogicConfig()),
        f".subckt row {' '.join(pins)} vdd vss", *instances, ".ends row",
    ]))
    result = lvs_layout_vs_netlist(gds, row.name, reference, "row", workdir=tmp_path / "lvs")
    assert result.matched, str(result)
    assert result.mismatched_nets == []


@pytest.mark.skipif(not physical_tools_available, reason="Sky130 DRC/LVS tools are missing")
@pytest.mark.parametrize("rows,side", [(2, "full"), (4, "full"), (8, "left"), (16, "right"), (32, "full")])
def test_physical_decoder_drc_lvs(tech, tmp_path, rows, side):
    cfg = ArrayConfig.basic(rows=rows, cols=4)
    logic = LogicConfig()
    left, right = side_outputs(rows)
    outputs = {"full": all_row_outputs(rows), "left": left, "right": right}[side]
    decoder = row_decoder(tech, cfg, logic, outputs=outputs)
    gds = decoder.write_gds(tmp_path / "decoder.gds")
    drc = run_drc(gds)
    assert drc.clean, str(drc)
    reference = tmp_path / "decoder.spice"
    reference.write_text("\n\n".join([
        inverter_subckt(tech, logic),
        *gate_subckts(tech, logic),
        row_decoder_subckt(tech, cfg, logic, outputs=outputs),
    ]))
    lvs = lvs_layout_vs_netlist(
        gds, decoder.name, reference, "gc_row_decoder", workdir=tmp_path / "lvs"
    )
    assert lvs.matched, str(lvs)
    assert lvs.mismatched_nets == []
    assert lvs.net_count[0] == lvs.net_count[1]


@pytest.mark.skipif(
    shutil.which("ngspice") is None or not default_model_lib().exists(),
    reason="ngspice or Sky130 models are missing",
)
@pytest.mark.parametrize("write_enable,read_enable", [(False, False), (True, False), (False, True), (True, True)])
def test_decoder_operating_point(tech, tmp_path, write_enable, read_enable):
    cfg = ArrayConfig.basic(rows=4, cols=4)
    levels = run_decoder_op(
        tech, cfg, row=2, we=write_enable, re=read_enable, workdir=tmp_path
    )
    for row in range(cfg.rows):
        assert levels[f"wwl_in_{row}"] == pytest.approx(
            1.8 if write_enable and row == 2 else 0.0, abs=0.01
        )
        assert levels[f"rwl_in_{row}"] == pytest.approx(
            0.0 if read_enable and row == 2 else 1.8, abs=0.01
        )
