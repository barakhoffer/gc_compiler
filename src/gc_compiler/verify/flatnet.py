"""The flat, parasitic-annotated netlist magic extracts from a macro layout, and edits to it.

``ext2spice`` writes every device and every extracted capacitor of the macro at one level,
with nets named after the labels the layout carries (``addr_0``, ``we``, ``dout_3``, ``rdy``
at the top; ``<array>/rwl_5``, ``<array>/rbl_3`` and the like inside). That is directly
simulable, but for a timing characterisation it is both too big (every bitcell of the array)
and missing something (extraction gives a word line's capacitance, not its resistance).

:class:`FlatNetlist` parses it and offers the two edits :mod:`gc_compiler.verify.timing`
needs:

* :meth:`FlatNetlist.reduce` keeps a few columns as real cells and stands in for the rest with
  gate loads on the word lines (which is all the rest of the array does to a read or a write in
  another column), dropping the unused columns' bit lines, cells and column circuits;
* :meth:`FlatNetlist.add_wordline_rc` turns each word line into a distributed RC ladder along
  the columns, driven from whichever end its driver is at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gc_compiler.tech.rules import DeviceKind

if TYPE_CHECKING:
    from gc_compiler.tech.rules import Technology

_RWL = re.compile(r"(?:^|/)rwl_(\d+)$")
_WWL = re.compile(r"(?:^|/)wwl_(\d+)$")
_RBL = re.compile(r"(?:^|/)rbl_(\d+|rep)$")
_WBL = re.compile(r"(?:^|/)wbl_(\d+|rep)$")

_SI = {"a": 1e-18, "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3}


def parse_si(text: str) -> float:
    """``0.02546f`` -> 2.546e-17."""
    if text and text[-1] in _SI:
        return float(text[:-1]) * _SI[text[-1]]
    return float(text)


@dataclass
class Device:
    """``X<name> <terminals...> <model> <params...>`` as magic writes a device.

    Transistors have four terminals (drain, gate, source, bulk); the tap rows' varactors three.
    """

    name: str
    nodes: list[str]
    model: str
    params: str = ""
    kind: DeviceKind | None = None

    @property
    def d(self) -> str:
        return self.nodes[0]

    @property
    def g(self) -> str:
        return self.nodes[1]

    @g.setter
    def g(self, net: str) -> None:
        self.nodes[1] = net

    @property
    def s(self) -> str:
        return self.nodes[2]

    @property
    def b(self) -> str:
        return self.nodes[3]

    @property
    def is_nmos(self) -> bool:
        return self.kind is DeviceKind.NMOS

    @property
    def is_pmos(self) -> bool:
        return self.kind is DeviceKind.PMOS

    def terminals(self) -> list[str]:
        return self.nodes

    def line(self) -> str:
        return f"{self.name} {' '.join(self.nodes)} {self.model} {self.params}".rstrip()


@dataclass
class Cap:
    name: str
    a: str
    b: str
    value: float

    def line(self) -> str:
        return f"{self.name} {self.a} {self.b} {self.value:.6g}"


@dataclass
class Cell:
    """One bitcell found in the netlist: its three transistors and its two internal nodes."""

    row: int
    col: str  # "0", "1", ... or "rep"
    ms: Device
    mr: Device
    mw: Device
    n_int: str
    sn: str

    def devices(self) -> tuple[Device, Device, Device]:
        return (self.ms, self.mr, self.mw)


@dataclass
class FlatNetlist:
    devices: list[Device] = field(default_factory=list)
    caps: list[Cap] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    """Other lines to emit unchanged (resistors, comments)."""
    _counter: int = 0
    array_prefix: str | None = None
    """With several arrays (a banked macro), the hierarchy prefix of the one to work on: cells
    and word lines are found under it only (see :meth:`array_prefixes`)."""

    @classmethod
    def parse(cls, text: str, tech: Technology | None = None) -> FlatNetlist:
        """``tech`` (default: the default technology) tells transistors and varactors by model."""
        if tech is None:
            from gc_compiler.tech import default_technology

            tech = default_technology()
        joined: list[str] = []
        for line in text.splitlines():
            if line.startswith("+") and joined:
                joined[-1] += " " + line[1:].strip()
            else:
                joined.append(line)
        net = cls()
        for line in joined:
            tokens = line.split()
            if not tokens or line.startswith("*") or line.startswith("."):
                continue
            if line.startswith("X"):
                kinds = [(i, tech.classify_model(t)) for i, t in enumerate(tokens)]
                model, kind = next((i, k) for i, k in kinds if k is not None)
                if kind == "varactor":
                    # A word line's poly over a tap row's n-well: magic calls it a varactor. What it is
                    # is ~0.2 fF of gate-to-well capacitance, and the varactor model is out of range
                    # at the sizes such a sliver has (sqrt of a negative in the ss/sf corner files),
                    # so it becomes a plain capacitor of Cox * W * L.
                    params = dict(t.split("=", 1) for t in tokens[model + 1 :] if "=" in t)
                    area = parse_si(params["w"]) * parse_si(params["l"])
                    cox = tech.electrical.gate_cap_per_um2
                    net.caps.append(Cap(f"C{tokens[0]}", tokens[1], tokens[2], area * cox))
                    continue
                net.devices.append(
                    Device(tokens[0], tokens[1:model], tokens[model], " ".join(tokens[model + 1 :]), kind)
                )
            elif line.startswith("C") and len(tokens) >= 4:
                net.caps.append(Cap(tokens[0], tokens[1], tokens[2], parse_si(tokens[3].split("$")[0])))
            else:
                net.extra.append(line)
        return net

    # --- structure -------------------------------------------------------------------

    def find_cells(self) -> dict[tuple[int, str], Cell]:
        """Every bitcell, keyed by ``(row, column)``, from the transistors' connections.

        The read-select device is the nfet whose gate is a read word line and one of whose
        terminals is a read bit line; the read device is the nfet from its other terminal to
        ``vss`` (its gate is the storage node); the write device is the pfet whose gate is the
        row's write word line and one of whose terminals is the column's write bit line.
        """
        nfets = [d for d in self.devices if d.is_nmos]
        pfets = [d for d in self.devices if d.is_pmos]
        by_terminal: dict[str, list[Device]] = {}
        for device in nfets:
            for net in (device.d, device.s):
                by_terminal.setdefault(net, []).append(device)
        mine_only = self.array_prefix

        writes: dict[tuple[int, str], Device] = {}
        for device in pfets:
            gate = _WWL.search(device.g)
            if not gate or (mine_only is not None and not device.g.startswith(mine_only)):
                continue
            for net in (device.d, device.s):
                bit = _WBL.search(net)
                if bit:
                    writes[(int(gate.group(1)), bit.group(1))] = device

        cells: dict[tuple[int, str], Cell] = {}
        for device in nfets:
            gate = _RWL.search(device.g)
            if not gate or (mine_only is not None and not device.g.startswith(mine_only)):
                continue
            for mine, other in ((device.d, device.s), (device.s, device.d)):
                bit = _RBL.search(mine)
                if not bit:
                    continue
                key = (int(gate.group(1)), bit.group(1))
                read = next(
                    (m for m in by_terminal.get(other, []) if m is not device and "vss" in (m.d, m.s)),
                    None,
                )
                if read is None or key not in writes:
                    continue
                cells[key] = Cell(key[0], key[1], device, read, writes[key], other, read.g)
        return cells

    def wordline_prefix(self) -> str:
        """The hierarchy prefix of the array's nets, e.g. ``gc_array_16x20_rep_0/``."""
        if self.array_prefix is not None:
            return self.array_prefix
        for device in self.devices:
            match = _RWL.search(device.g)
            if match:
                return device.g[: device.g.rindex("rwl_")]
        raise ValueError("no read word line in the netlist")

    def array_prefixes(self) -> list[str]:
        """The hierarchy prefix of every array whose read word lines gate a device."""
        return sorted({d.g[: d.g.rindex("rwl_")] for d in self.devices if _RWL.search(d.g)})

    def drop_array(self, prefix: str) -> None:
        """Remove the array under ``prefix`` entirely: its cells, and everything on its bit lines
        (its column circuits). What is left of it -- its word lines and their drivers -- no
        longer touches anything that is kept."""
        selected, self.array_prefix = self.array_prefix, prefix
        try:
            cells = self.find_cells()
        finally:
            self.array_prefix = selected
        dead_devices = {id(d) for cell in cells.values() for d in cell.devices()}
        dead_nets = {net for cell in cells.values() for net in (cell.n_int, cell.sn)}
        dead_nets |= {f"{prefix}{kind}_{col}" for _row, col in cells for kind in ("rbl", "wbl")}
        self._kill(dead_nets, dead_devices)

    def net_matching(self, pattern: str) -> str:
        """The one net whose name matches ``pattern`` (a regex)."""
        found = sorted({n for d in self.devices for n in d.nodes if re.search(pattern, n)})
        if len(found) != 1:
            raise ValueError(f"expected one net matching {pattern!r}, found {found[:4]}")
        return found[0]

    # --- edits -----------------------------------------------------------------------

    def _fresh(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def reduce(self, keep: set[str], group: int = 4) -> dict[str, list[str]]:
        """Keep only the columns in ``keep`` (``"0"``, ``"7"``, ``"rep"``) as real cells.

        Every other column's cells, bit lines and column circuits are dropped. What those cells
        do to the columns that remain is load the word lines: each one's read-select and write
        gates. Each row gets, per group of ``group`` dropped columns, one pair of gate-load
        devices with a multiplier (their drains and sources on the rails), on the word line
        itself for now. Returns the dropped columns of each group, keyed ``"<first>"``; the
        caller (:meth:`add_wordline_rc`) puts the group loads on their ladder nodes.
        """
        cells = self.find_cells()
        columns = sorted({c for _r, c in cells if c != "rep"}, key=int)
        dropped = [c for c in columns if c not in keep]
        dropped_set = set(dropped)
        rows = sorted({r for r, _c in cells})
        prefix = self.wordline_prefix()

        dead_nets: set[str] = set()
        dead_devices: set[int] = set()
        for (_row, col), cell in cells.items():
            if col in dropped_set:
                dead_devices.update(id(d) for d in cell.devices())
                dead_nets.update((cell.n_int, cell.sn))
        for col in dropped:
            dead_nets.update((f"{prefix}rbl_{col}", f"{prefix}wbl_{col}", f"dout_{col}", f"din_{col}"))
        self._kill(dead_nets, dead_devices)

        loads: dict[str, list[str]] = {}
        for i in range(0, len(dropped), group):
            chunk = dropped[i : i + group]
            loads[chunk[0]] = chunk
        return loads

    def _kill(self, dead_nets: set[str], dead_devices: set[int]) -> None:
        """Drop ``dead_devices`` and every device with a drain, gate or source on a dead net
        (the column circuits of dropped bit lines); a capacitor to a dead net becomes one to
        ground (coupling to a quiet net), one between two dead nets goes."""
        for device in self.devices:
            if id(device) not in dead_devices and any(t in dead_nets for t in device.nodes[:3]):
                dead_devices.add(id(device))
        self.devices = [d for d in self.devices if id(d) not in dead_devices]
        # a net no device touches any more (a dropped column's pins, say) is dead too: left with
        # only capacitors it floats, and the simulator's operating point cannot solve it
        touched = {net for device in self.devices for net in device.nodes}
        dead_nets = dead_nets | {net for cap in self.caps for net in (cap.a, cap.b) if net not in touched}
        dead_nets -= {"vss", "vdd", "0"}
        kept_caps: list[Cap] = []
        for cap in self.caps:
            a_dead, b_dead = cap.a in dead_nets, cap.b in dead_nets
            if a_dead and b_dead:
                continue
            if a_dead or b_dead:
                other = cap.b if a_dead else cap.a
                kept_caps.append(Cap(cap.name, other, "vss", cap.value))
            else:
                kept_caps.append(cap)
        self.caps = kept_caps

    def add_wordline_rc(
        self,
        columns: list[str],
        x_of: dict[str, float],
        left_driven: dict[tuple[str, int], bool],
        resistance_per_um: dict[str, float],
        loads: dict[str, list[str]],
        gate_load: dict[str, tuple[str, str]],
        edges: tuple[float, float],
        taps: tuple[float, ...] = (),
        strap_per_um: float = 0.0,
        tap_resistance: float = 0.0,
    ) -> dict[tuple[str, int, str], str]:
        """Turn every word line into an RC ladder along ``columns`` (kept columns and dropped
        groups, left to right, at x positions ``x_of``).

        ``left_driven[(kind, row)]`` says which end the row's driver is at (the net's own name
        stays the driver end). ``resistance_per_um`` is the poly resistance per um for
        ``"rwl"`` and ``"wwl"``. Kept cells' gates move to their column's ladder node; each
        dropped group becomes a load at its node (``gate_load`` gives the ``(model, size parameters)`` of the read-select
        and write devices, whose gates a word line drives); the wire capacitance on the net is split evenly over the ladder nodes.
        ``taps`` are the x positions where a strapped word line meets its metal strap: the strap
        is a second ladder from the driver end (``strap_per_um``) joined to the poly there
        through ``tap_resistance``.
        Returns ``{(kind, row, column): node}`` for probing.
        """
        cells = self.find_cells()
        prefix = self.wordline_prefix()
        rows = sorted({r for r, _c in cells})
        left, right = edges
        nodes: dict[tuple[str, int, str], str] = {}
        weight = {col: len(loads.get(col, [col])) for col in columns}
        total = sum(weight.values())
        ladder_of: dict[str, dict[str, str]] = {}

        for kind in ("rwl", "wwl"):
            for row in rows:
                net = f"{prefix}{kind}_{row}"
                driven_left = left_driven[(kind, row)]
                points = [(x_of[col], col, None) for col in columns] + [(x, None, i) for i, x in enumerate(taps)]
                points.sort(key=lambda p: p[0], reverse=not driven_left)
                previous_x = left if driven_left else right
                previous_node = net
                strap_node, strap_x = net, previous_x
                for x, col, tap in points:
                    node = f"{net}__c{col}" if col is not None else f"{net}__t{tap}"
                    length = abs(x - previous_x)
                    self.extra.append(
                        f"R{self._fresh('wl')} {previous_node} {node} {resistance_per_um[kind] * length:.6g}"
                    )
                    if col is not None:
                        nodes[(kind, row, col)] = node
                    else:
                        metal = f"{net}__s{tap}"
                        self.extra.append(
                            f"R{self._fresh('ws')} {strap_node} {metal} {max(strap_per_um * abs(x - strap_x), 1e-3):.6g}"
                        )
                        self.extra.append(f"R{self._fresh('wt')} {metal} {node} {tap_resistance:.6g}")
                        strap_node, strap_x = metal, x
                    previous_node, previous_x = node, x
                ladder_of[net] = {col: nodes[(kind, row, col)] for col in columns}

                # gates of kept cells move to their column's node
                for (r, col), cell in cells.items():
                    if r != row or col not in x_of or col in loads:
                        continue
                    device = cell.ms if kind == "rwl" else cell.mw
                    if device.g == net:
                        device.g = nodes[(kind, row, col)]
                # dropped groups: one multiplied gate load per group
                for first, members in loads.items():
                    node = nodes[(kind, row, first)]
                    if kind == "rwl":
                        model, sizes = gate_load["n"]
                        terminals, device_kind = ["vss", node, "vss", "vss"], DeviceKind.NMOS
                    else:
                        model, sizes = gate_load["p"]
                        terminals, device_kind = ["vdd", node, "vdd", "vdd"], DeviceKind.PMOS
                    params = f"{sizes} m={len(members)}"
                    self.devices.append(Device(f"X{self._fresh('ld')}", terminals, model, params, device_kind))

        # Every wire capacitance on a word line moves onto its ladder. One to a kept cell's own
        # nodes (its storage node, its middle node) stays with that cell's column; the rest --
        # to the substrate, to bit lines, to other word lines -- is summed into one capacitance
        # to ground (coupling to a switching neighbour is treated as ground capacitance) and
        # spread along the ladder in proportion to the columns each node stands for.
        cell_column: dict[str, str] = {}
        for cell in cells.values():
            if cell.col in x_of and cell.col not in loads:
                cell_column[cell.sn] = cell.col
                cell_column[cell.n_int] = cell.col
        ground: dict[str, float] = {net: 0.0 for net in ladder_of}
        kept: list[Cap] = []
        for cap in self.caps:
            a_wl, b_wl = cap.a in ladder_of, cap.b in ladder_of
            if not (a_wl or b_wl):
                kept.append(cap)
                continue
            wl, other = (cap.a, cap.b) if a_wl else (cap.b, cap.a)
            column = cell_column.get(other)
            if column is not None:
                kept.append(Cap(f"C{self._fresh('wc')}", ladder_of[wl][column], other, cap.value))
                continue
            ground[wl] += cap.value
            if a_wl and b_wl:
                ground[cap.b] += cap.value
        for net, total_cap in ground.items():
            for col in columns:
                kept.append(Cap(f"C{self._fresh('wc')}", ladder_of[net][col], "vss", total_cap * weight[col] / total))
        self.caps = kept
        return nodes

    def drop_static_caps(self, dc_nets: set[str]) -> int:
        """Drop capacitors between two DC nets (rails, the n-well tie) or from a net to itself;
        they load only the supplies. Returns how many went."""
        before = len(self.caps)
        self.caps = [
            c for c in self.caps if c.a != c.b and not (c.a in dc_nets and c.b in dc_nets)
        ]
        return before - len(self.caps)

    def emit(self) -> str:
        lines = [d.line() for d in self.devices]
        lines += [c.line() for c in self.caps]
        lines += self.extra
        return "\n".join(lines) + "\n"
