"""What a place-and-route flow needs to use the gain-cell macro as a hard macro: the macro moved
to the origin with metal4 power pins (:func:`hard_macro`), a LEF abstract of it
(:func:`write_lef`), a timing library (:func:`write_liberty`) and a Verilog blackbox
(:func:`write_blackbox`). All of them describe the same pins as ``macro_netlist``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gdsfactory as gf
import klayout.db as kdb

from gc_compiler.array.power import PowerStripe, add_power_pins
from gc_compiler.config import ArrayConfig
from gc_compiler.netlist.spice import macro_pins
from gc_compiler.periphery.logic import LogicConfig
from gc_compiler.primitives.geometry import Box, draw
from gc_compiler.tech.rules import LayerRole, Technology

def pin_direction(pin: str) -> str:
    if pin.startswith(("dout", "rdy")):
        return "OUTPUT"
    if pin in ("vdd", "vss", "vneg") or pin.startswith("vnw"):
        return "INOUT"
    return "INPUT"


def pin_use(pin: str) -> str:
    if pin == "vss" or pin == "vneg":
        return "GROUND"
    if pin == "vdd" or pin.startswith("vnw"):
        return "POWER"
    return "SIGNAL"


@dataclass(frozen=True)
class HardMacro:
    component: gf.Component
    """The macro, lower-left corner at the origin, with metal4 power stripes."""
    name: str
    pins: list[str]
    stripes: list[PowerStripe]
    width: float
    height: float


def hard_macro(tech: Technology, macro: gf.Component, cfg: ArrayConfig, name: str, banked: bool) -> HardMacro:
    """The decoded ``macro`` as a hard macro called ``name``: moved so its bounding box starts at
    the origin (LEF geometry is relative to the lower-left corner), with metal4 power pins."""
    bank_x = macro.info.get("bank_x") if banked else None
    stripes = add_power_pins(macro, tech, bank_x)
    bounds = macro.dbbox()
    wrapper = gf.Component(name)
    ref = wrapper.add_ref(macro)
    ref.dmove((-bounds.left, -bounds.bottom))
    wrapper.add_ports(ref.ports)
    # the placement boundary: what magic reads the macro's size from when it streams the design out
    draw(wrapper, tech.layer(LayerRole.BOUNDARY), Box(0.0, 0.0, bounds.width(), bounds.height()))
    pins = macro_pins(cfg, decoded=True, banked=banked)
    shifted = [PowerStripe(s.net, s.x - bounds.left, tuple(y - bounds.bottom for y in s.sites)) for s in stripes]
    _declare_pins(tech, wrapper, pins, shifted, bounds.height())
    for key in ("bits", "decoded", "banked", "bank_x"):
        if macro.info.get(key) is not None:
            wrapper.info[key] = macro.info[key]
    wrapper.info["width"], wrapper.info["height"] = bounds.width(), bounds.height()
    return HardMacro(wrapper, name, pins, shifted, bounds.width(), bounds.height())


def _declare_pins(tech: Technology, cell: gf.Component, pins: list[str], stripes: list[PowerStripe], height: float) -> None:
    """Name the macro's pins in its own top cell -- a shape on the metal's pin layer and a label --
    so an extraction of it (magic's, inside the flow's LVS) gives the cell those ports; they are
    otherwise named only inside the cell it wraps."""
    routing = tech.stack.routing
    role_of = {gf.get_layer(tech.layer(role)): role for role in routing}
    ports = {p.name: p for p in cell.ports}
    dbu = cell.kcl.layout.dbu
    half = round(tech.rules.w(tech.stack.above(routing[0])) / 2 / dbu)
    metal = {role: _layer_region(cell, tech.layer(role)) for role in routing}
    for pin in pins:
        if pin_use(pin) != "SIGNAL":
            continue
        port = ports[pin]
        role = role_of[gf.get_layer(port.layer)]
        x, y = port.dcenter
        cx, cy = round(x / dbu), round(y / dbu)
        # the marker only where there is metal: a port sits at the end of its track, and a marker
        # centred on it would reach past the metal the LEF describes -- magic reads a pin shape
        # as metal, and the router, going by the LEF, may route that close
        marker = metal[role] & kdb.Region(kdb.Box(cx - half, cy - half, cx + half, cy + half))
        if marker.is_empty():
            raise ValueError(f"no {role.value} under pin {pin} at ({x:.3f}, {y:.3f})")
        layer = cell.kcl.layout.layer(*tech.pin_layer(role))
        for polygon in marker.each():
            cell.shapes(layer).insert(polygon)
        cell.add_label(text=pin, position=(x, y), layer=tech.label_layer(role))
    pin_role = tech.stack.power_pin_layer
    # a marker inside the stripe, clear of its edges
    marker = tech.policy.power_stripe.width / 2 - 2 * tech.policy.join
    for stripe in stripes:
        x, y = stripe.x, height / 2
        draw(cell, tech.pin_layer(pin_role), Box(x - marker, y - marker, x + marker, y + marker))
        cell.add_label(text=stripe.net, position=(x, y), layer=tech.label_layer(pin_role))


def write_gds(component: gf.Component, path: Path) -> Path:
    """The GDS without KLayout's context-information cell (``$$$CONTEXT_INFO$$$``), which
    gdsfactory writes and magic -- the flow's stream-out and extraction -- cannot read."""
    component.write_gds(path)
    layout = kdb.Layout()
    layout.read(str(path))
    options = kdb.SaveLayoutOptions()
    options.format = "GDS2"
    options.write_context_info = False
    layout.write(str(path), options)
    return path


def _layer_region(component: gf.Component, layer: tuple[int, int]) -> kdb.Region:
    index = component.kcl.layout.find_layer(*layer)
    if index is None:
        return kdb.Region()
    return kdb.Region(component.begin_shapes_rec(index)).merged()


def _rects(region: kdb.Region) -> list[kdb.Box]:
    boxes = []
    for polygon in region.each():
        for piece in polygon.decompose_trapezoids(kdb.Polygon.TD_htrapezoids):
            boxes.append(piece.bbox())
    return boxes


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def antenna_areas(tech: Technology, logic: LogicConfig | None = None) -> tuple[float, float]:
    """``(gate, diffusion)``: lower bounds of what a macro input pin drives (the gates of one
    logic-cell inverter -- the smallest thing any input reaches) and of what drives an output pin
    (the drain of one logic-cell NMOS). Lower bounds make the flow's antenna check stricter, never
    laxer: a smaller gate area raises the ratio, a smaller diffusion protects less."""
    rules = tech.rules
    logic = logic or LogicConfig.for_tech(tech)
    gate = (logic.pmos_width + logic.nmos_width) * logic.length
    drain = logic.nmos_width * (rules.cut(LayerRole.LICON) + 2 * rules.gate_to_cut_space)
    return round(gate, 4), round(drain, 4)


def write_lef(
    tech: Technology, hard: HardMacro, path: Path, logic: LogicConfig | None = None, unused: tuple[str, ...] = ()
) -> Path:
    """A LEF abstract: each signal pin is the metal it is labelled on (the whole polygon, so the
    router can reach it anywhere), the supplies are the metal4 stripes, and every other shape
    on metal1 to metal3 is an obstruction; li1 is blocked outright. The ``unused`` pins (what the
    block leaves unconnected: ``rdy`` without the handshake) are obstructions too, a spacing wider:
    the router may run a wire between the macro's pins, and passed right over an unconnected one.
    Signal pins carry their antenna areas (:func:`antenna_areas`), so the flow's antenna check knows a net into the macro
    ends on a gate and a net out of it is driven by a diffusion."""
    component = hard.component
    dbu = component.kcl.layout.dbu
    gate_area, diff_area = antenna_areas(tech, logic)
    ports = {p.name: p for p in component.ports}
    stack = tech.stack
    regions = {role: _layer_region(component, tech.layer(role)) for role in stack.routing}
    role_of = {gf.get_layer(tech.layer(role)): role for role in stack.routing}
    pin_shapes: dict[str, list[tuple[LayerRole, kdb.Box]]] = {}
    for pin in hard.pins:
        if pin_use(pin) != "SIGNAL":
            continue
        port = ports[pin]
        role = role_of[gf.get_layer(port.layer)]
        x, y = port.dcenter
        probe = kdb.Region(kdb.Box(round(x / dbu) - 1, round(y / dbu) - 1, round(x / dbu) + 1, round(y / dbu) + 1))
        polygon = regions[role].interacting(probe)
        if polygon.is_empty():
            raise ValueError(f"no {role.value} under pin {pin} at ({x:.3f}, {y:.3f})")
        pin_shapes[pin] = [(role, b) for b in _rects(polygon)]
    half = tech.policy.power_stripe.width / 2
    for stripe in hard.stripes:
        box = kdb.Box(round((stripe.x - half) / dbu), 0, round((stripe.x + half) / dbu), round(hard.height / dbu))
        pin_shapes.setdefault(stripe.net, []).append((stack.power_pin_layer, box))
    lines = [
        "VERSION 5.7 ;", 'BUSBITCHARS "[]" ;', 'DIVIDERCHAR "/" ;',
        f"MACRO {hard.name}",
        "  CLASS BLOCK ;",
        f"  FOREIGN {hard.name} 0 0 ;",
        "  ORIGIN 0 0 ;",
        f"  SIZE {_fmt(hard.width)} BY {_fmt(hard.height)} ;",
        "  SYMMETRY X Y ;",
    ]
    for pin in hard.pins:
        if pin not in pin_shapes:
            raise ValueError(f"pin {pin} has no shape (no stripe for a supply?)")
        lines += [f"  PIN {pin}", f"    DIRECTION {pin_direction(pin)} ;", f"    USE {pin_use(pin)} ;"]
        if pin_use(pin) == "SIGNAL":
            if pin_direction(pin) == "OUTPUT":
                lines.append(f"    ANTENNADIFFAREA {diff_area} ;")
            else:
                lines.append(f"    ANTENNAGATEAREA {gate_area} ;")
        lines.append("    PORT")
        layer = None
        for role, box in pin_shapes[pin]:
            if role is not layer:
                lines.append(f"      LAYER {stack.lef_name(role)} ;")
                layer = role
            lines.append(
                f"        RECT {_fmt(box.left * dbu)} {_fmt(box.bottom * dbu)} {_fmt(box.right * dbu)} {_fmt(box.top * dbu)} ;"
            )
        lines += ["    END", f"  END {pin}"]
    lines.append("  OBS")
    # the local interconnect is blocked outright; the metals between it and the pin layer are
    # obstructed shape by shape, so the router can reach the pins between them -- but for the
    # unused pins, which are blocked a spacing wide
    lowest = stack.local_interconnect or stack.routing[0]
    lines += [f"    LAYER {stack.lef_name(lowest)} ;", f"      RECT 0.000 0.000 {_fmt(hard.width)} {_fmt(hard.height)} ;"]
    blocked = {role: kdb.Region() for role in stack.routing}
    for pin in unused:
        for role, box in pin_shapes.get(pin, []):
            blocked[role] += kdb.Region(box).sized(round(tech.rules.s(role) / dbu))
    pins = {role: kdb.Region() for role in stack.routing}
    for pin, shapes in pin_shapes.items():
        if pin_use(pin) == "SIGNAL" and pin not in unused:
            for role, box in shapes:
                pins[role] += kdb.Region(box)
    for role in [r for r in stack.routing if r not in (lowest, stack.power_pin_layer)]:
        rest = (regions[role] + blocked[role]) - pins[role]
        if rest.is_empty():
            continue
        lines.append(f"    LAYER {stack.lef_name(role)} ;")
        for box in _rects(rest):
            lines.append(
                f"      RECT {_fmt(box.left * dbu)} {_fmt(box.bottom * dbu)} {_fmt(box.right * dbu)} {_fmt(box.top * dbu)} ;"
            )
    lines += ["  END", f"END {hard.name}", "END LIBRARY", ""]
    path.write_text("\n".join(lines))
    return path


def write_blackbox(hard: HardMacro, path: Path, powered: bool = False) -> Path:
    """The macro as a Verilog blackbox: signal pins only for synthesis, or with its supply pins
    too (``powered``) for the powered netlist LVS compares against."""
    pins = [p for p in hard.pins if powered or pin_use(p) == "SIGNAL"]
    kind = {"OUTPUT": "output", "INPUT": "input", "INOUT": "inout"}
    decl = [f"    {kind[pin_direction(p)]} {p}" for p in pins]
    path.write_text(f"(* blackbox *)\nmodule {hard.name} (\n" + ",\n".join(decl) + "\n);\nendmodule\n")
    return path


def write_liberty(
    hard: HardMacro, path: Path, corner: str, read_ns: float, input_pf: float | None = None,
    vdd: float | None = None, temperature: float | None = None, tech: Technology | None = None,
) -> Path:
    """A timing library of the macro at one corner: its pins' directions and an input load, and a
    read as a combinational arc from ``re`` to every ``dout``/``rdy`` taking ``read_ns`` (the
    characterised access at that corner, or the controller's conservative default). Its supply
    and temperature are the standard-cell library's at that corner (``tech``'s, by default the
    default technology's), unless given."""
    if tech is None:
        from gc_compiler.tech import default_technology

        tech = default_technology()
    from gc_compiler.rtl import _measured

    measured = _measured(tech)
    input_pf = measured.macro_input_pf if input_pf is None else input_pf
    transition = measured.macro_output_transition_ns
    if vdd is None or temperature is None:
        lib_vdd, lib_temp = tech.standard_cells().conditions[corner]
        vdd = lib_vdd if vdd is None else vdd
        temperature = lib_temp if temperature is None else temperature
    signals = [p for p in hard.pins if pin_use(p) == "SIGNAL"]
    body = []
    for pin in hard.pins:
        use = pin_use(pin)
        if use != "SIGNAL":
            kind = "primary_ground" if use == "GROUND" else "primary_power"
            body.append(f'    pg_pin ({pin}) {{ pg_type : {kind}; voltage_name : "{pin.upper()}"; }}')
    for pin in signals:
        if pin_direction(pin) == "INPUT":
            body.append(f"    pin ({pin}) {{ direction : input; capacitance : {input_pf}; }}")
        else:
            body.append(
                f"    pin ({pin}) {{\n      direction : output;\n      timing () {{\n"
                f'        related_pin : "re";\n        timing_type : combinational;\n'
                f'        cell_rise (scalar) {{ values ("{read_ns:.4f}"); }}\n'
                f'        cell_fall (scalar) {{ values ("{read_ns:.4f}"); }}\n'
                f'        rise_transition (scalar) {{ values ("{transition:g}"); }}\n'
                f'        fall_transition (scalar) {{ values ("{transition:g}"); }}\n      }}\n    }}'
            )
    voltages = "\n".join(f'  voltage_map ({p.upper()}, {0.0 if pin_use(p) == "GROUND" else vdd});' for p in hard.pins if pin_use(p) != "SIGNAL")
    path.write_text(
        f"""library ({hard.name}_{corner}) {{
  delay_model : table_lookup;
  time_unit : "1ns";
  voltage_unit : "1V";
  current_unit : "1mA";
  capacitive_load_unit (1, pf);
  pulling_resistance_unit : "1kohm";
  leakage_power_unit : "1nW";
  nom_voltage : {vdd};
  nom_temperature : {temperature:g};
  nom_process : 1;
  input_threshold_pct_rise : 50;
  input_threshold_pct_fall : 50;
  output_threshold_pct_rise : 50;
  output_threshold_pct_fall : 50;
  slew_lower_threshold_pct_rise : 20;
  slew_lower_threshold_pct_fall : 20;
  slew_upper_threshold_pct_rise : 80;
  slew_upper_threshold_pct_fall : 80;
{voltages}
  cell ({hard.name}) {{
    area : {hard.width * hard.height:.3f};
    interface_timing : true;
{chr(10).join(body)}
  }}
}}
"""
    )
    return path
