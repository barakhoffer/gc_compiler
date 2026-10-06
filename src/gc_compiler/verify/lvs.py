"""LVS: layout-vs-schematic checking via magic extraction + netgen comparison.

``run_lvs`` compares a GDS against a reference SPICE netlist by extracting a netlist from
the layout with magic and comparing the two circuits with netgen's graph-isomorphism
engine. Unlike DRC, this catches *connectivity* bugs that pure geometry checks cannot see
(a floating well tap, a bit line landing on the wrong node, ...).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


def _tools(tech):
    if tech is None:
        from gc_compiler.tech import default_technology

        tech = default_technology()
    return tech.tools


def default_magicrc(pdk_root: str | os.PathLike[str] | None = None, tech=None) -> Path:
    """The technology's magic setup (see :class:`~gc_compiler.tech.tools.ToolSetup`)."""
    return _tools(tech).magicrc_path(pdk_root)


def default_netgen_setup(pdk_root: str | os.PathLike[str] | None = None, tech=None) -> Path:
    """The technology's netgen LVS setup."""
    return _tools(tech).netgen_setup_path(pdk_root)


# Magic's Tcl binding leaves its own cell-loading command in the ``::magic`` namespace
# because plain ``load`` is shadowed by Tcl's builtin (which loads shared libraries).
#
# The port-declaration block (built by ``_port_script``, when ``ports`` is given) runs
# *before* ``select top cell``: selecting the top cell first makes every label in the whole
# hierarchy visible to ``select area``, so a query for one port's label ambiguously matches
# a same-position label belonging to a child cell too ("Exactly one label may be present
# under the cursor box"). Run before that select, ``box``/``select area`` only see the top
# cell's own directly-drawn labels (the ones ``floorplan.py`` adds at each port), so each
# query is unambiguous.
_EXTRACT_SCRIPT = """\
gds read {gds}
::magic::load {cell}
{ports}select top cell
extract path {extdir}
extract all
ext2spice lvs
ext2spice hierarchy on
ext2spice -p {extdir} -o {out}
quit -noprompt
"""

# A tiny margin around each port's exact position, not a zero-size point: a label position
# computed in Python (e.g. a bus trunk's midpoint, which need not land on-grid) can be a
# few nanometers off from wherever magic's own internal grid snapped the same label when it
# read the GDS, and a zero-size box misses it. 0.02 um is far below any real feature pitch in
# this layout (the tightest, decoder bus tracks, is still >0.4 um), so it cannot pick up an
# unrelated neighbouring label.
_PORT_MARGIN = 0.02
_PORT_LINE = (
    "box values {x0}um {y0}um {x1}um {y1}um\nselect area labels\ncatch {{port make}}\n"
)


def _port_script(ports: Mapping[str, tuple[float, float]]) -> str:
    lines = [
        _PORT_LINE.format(x0=x - _PORT_MARGIN, y0=y - _PORT_MARGIN, x1=x + _PORT_MARGIN, y1=y + _PORT_MARGIN)
        for x, y in ports.values()
    ]
    lines.append("port renumber\n")
    return "".join(lines)


_EXT_PORT_LINE = re.compile(r'^port\s+"([^"]+)"\s+(-?\d+)', re.MULTILINE)


def _fix_truncated_subckt_pins(spice: Path, cell: str, ext_file: Path) -> None:
    """Rebuild ``cell``'s ``.subckt`` pin list from the ``.ext`` file's own ``port`` records.

    ``ext2spice`` (magic 8.3.629) silently truncates a top-level ``.subckt`` pin list after
    the first several ports on a macro with two dozen or so pins -- the ``.ext`` file itself
    (the source ``ext2spice`` reads from) has every port correctly, so this rebuilds the
    line from there instead of trusting ``ext2spice``'s own output.
    """
    if not ext_file.exists():
        return
    by_index = sorted(
        ((int(idx), name) for name, idx in _EXT_PORT_LINE.findall(ext_file.read_text())),
        key=lambda pair: pair[0],
    )
    if not by_index:
        return
    names = [name for _, name in by_index]
    text = spice.read_text()
    fixed = re.sub(
        rf"^\.subckt {re.escape(cell)}(?:\s.*)?$",
        ".subckt " + cell + " " + " ".join(names),
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if fixed != text:
        spice.write_text(fixed)


def magic_extract(
    gds: str | os.PathLike[str],
    cell: str,
    workdir: str | os.PathLike[str],
    magicrc: str | os.PathLike[str] | None = None,
    timeout: int = 1800,
    ignore_devices: tuple[str, ...] = (),
    ports: Mapping[str, tuple[float, float]] | None = None,
    tech=None,
) -> Path:
    """Extract a SPICE netlist from ``cell`` in ``gds`` using magic.

    Hierarchy is preserved: every gdsfactory sub-cell becomes its own ``.subckt``, matching
    the structure of the compiler's own generated netlists.

    ``ignore_devices`` drops instances of the named device models from the extracted
    netlist (none by default). It used to drop ``sky130_fd_pr__cap_var_lvt``: the strap
    columns' n-well taps sat under the write word lines' poly, which the extractor rightly
    took for MOS varactors; the taps now keep clear of the poly (see
    :mod:`gc_compiler.array.strap_column`), no varactor is extracted, and LVS would catch one
    that came back.

    ``ports``, when given, is ``{name: (x, y)}`` in microns for every pin ``cell`` should
    expose at the top of the extracted netlist -- typically a gdsfactory component's own
    ``{p.name: tuple(p.dcenter) for p in component.ports}``. Magic only emits a top-level
    ``.subckt`` pin list for labels explicitly turned into ports (see :func:`_port_script`);
    without this, the *reference* netlist's pins have nothing to compare against on the
    layout side, and netgen reports a topology match with unverified ports (see
    :attr:`LvsResult.port_errors`). Each named position needs a label already drawn there --
    which every port ``gc_macro``/``gc_array`` add is paired with (see
    ``array/floorplan.py::_add_ports``) -- except ``vnw``, which has no macro-level pin at
    all (the array's own n-well bias net terminates at its internal tap, never reaching the
    macro boundary); asking for a name with no matching label is silently a no-op.
    """
    magic = shutil.which("magic")
    if magic is None:
        raise RuntimeError("magic is not on PATH (enter the nix shell first)")

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    extdir = workdir / "ext"
    extdir.mkdir(exist_ok=True)
    out = workdir / f"{cell}.extracted.spice"

    script = workdir / "extract.tcl"
    script.write_text(
        _EXTRACT_SCRIPT.format(
            gds=Path(gds).resolve(),
            cell=cell,
            extdir=extdir,
            out=out,
            ports=_port_script(ports) if ports else "",
        )
    )

    magicrc = Path(magicrc) if magicrc else default_magicrc(tech=tech)
    command = [magic, "-dnull", "-noconsole", "-rcfile", str(magicrc), str(script)]
    process = subprocess.run(
        command, cwd=workdir, capture_output=True, text=True, timeout=timeout
    )
    if not out.exists():
        raise RuntimeError(f"magic extraction failed:\n{process.stdout[-4000:]}\n{process.stderr[-2000:]}")
    if ports:
        _fix_truncated_subckt_pins(out, cell, extdir / f"{cell}.ext")
    if ignore_devices:
        _strip_devices(out, ignore_devices)
    return out


def _strip_devices(spice: Path, device_models: tuple[str, ...]) -> None:
    """Remove ``X...`` instance lines that reference an ignored device model."""
    lines = spice.read_text().splitlines()
    kept = [
        line
        for line in lines
        if not (line.startswith("X") and set(line.split()) & set(device_models))
    ]
    if len(kept) != len(lines):
        spice.write_text("\n".join(kept) + "\n")


@dataclass
class LvsResult:
    matched: bool
    port_errors: bool = False
    """Topology matches uniquely, but netgen could not verify the top-level port list.

    Without ``ports`` passed to :func:`magic_extract`/:func:`lvs_layout_vs_netlist`, this is
    always set: the extracted layout's top cell has no declared SPICE ports at all (magic
    only emits those from an explicit ``port`` assignment, which a GDS imported from another
    tool never has), so nothing about the pin list is checked. With ``ports`` given, this is
    set only for the pins actually listed in :attr:`pin_mismatches` -- on ``gc_macro``, that
    is normally just ``vnw`` (see its docstring in :func:`magic_extract`), not the whole list.
    """
    device_counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    net_count: tuple[int | None, int | None] = (None, None)
    #: (layout net, reference net) for every net netgen could not pair up; either side is
    #: ``None`` when that circuit has no corresponding net at all.
    mismatched_nets: list[tuple[str | None, str | None]] = field(default_factory=list)
    #: (layout pin, reference pin) for every top-level pin netgen could not verify by name;
    #: either side is ``None`` when that circuit has no port there at all. Only meaningful
    #: when ``port_errors`` is set.
    pin_mismatches: list[tuple[str | None, str | None]] = field(default_factory=list)
    report: str = ""

    def __bool__(self) -> bool:
        return self.matched

    def __str__(self) -> str:
        if self.matched:
            if not self.port_errors:
                return "LVS match"
            if self.pin_mismatches:
                names = ", ".join(l or r or "?" for l, r in self.pin_mismatches)
                return f"LVS match (unverified pins: {names})"
            return "LVS match (port list not verified)"
        lines = ["LVS mismatch"]
        if self.net_count != (None, None):
            lines.append(f"  nets: layout={self.net_count[0]} reference={self.net_count[1]}")
        for layout_net, ref_net in self.mismatched_nets:
            lines.append(f"  unmatched net: layout={layout_net!r} reference={ref_net!r}")
        for layout_pin, ref_pin in self.pin_mismatches:
            lines.append(f"  unmatched pin: layout={layout_pin!r} reference={ref_pin!r}")
        return "\n".join(lines)


_NET_LINE = re.compile(r"Number of nets:\s+(\d+)")
_NET_CELL = re.compile(r"^Net:\s*(\S+)")
_PIN_COLUMN = 43  # index of the "|" column separator in netgen's "Subcircuit pins:" table


def _parse_pin_mismatches(text: str) -> list[tuple[str | None, str | None]]:
    # netgen prints one "Subcircuit pins:" block per hierarchical cell it compares (device
    # placeholders first, then each subcell, then the top cell last, right before "Final
    # result:") -- only the *last* one is the top-level pin list this function reports on.
    marker = "Subcircuit pins:"
    if marker not in text:
        return []
    block = text.rsplit(marker, 1)[1].split("Cell pin lists", 1)[0]
    mismatches: list[tuple[str | None, str | None]] = []
    for line in block.splitlines():
        if "|" not in line:
            continue
        left, right = line[:_PIN_COLUMN].strip(), line[_PIN_COLUMN + 1 :].strip()
        if not left or not right or left.startswith("Circuit") or left.startswith("---"):
            continue
        left_name = None if left.startswith("(no pin") else left.replace("**Mismatch**", "").strip()
        right_name = None if right.startswith("(no pin") else right.replace("**Mismatch**", "").strip()
        if left_name != right_name:
            mismatches.append((left_name, right_name))
    return mismatches


def _parse_netgen_report(text: str) -> LvsResult:
    # Netgen still declares a full topology match even when it cannot verify the
    # top-level port list (see LvsResult.port_errors); both count as a real pass here.
    matched = "Circuits match uniquely" in text or "match uniquely with port errors" in text
    port_errors = "Top level cell failed pin matching" in text or "with port errors" in text
    nets = [int(m) for m in _NET_LINE.findall(text)]
    net_count = (nets[0], nets[1]) if len(nets) >= 2 else (None, None)
    pin_mismatches = _parse_pin_mismatches(text) if port_errors else []

    mismatched: list[tuple[str | None, str | None]] = []
    marker = "NET mismatches"
    if marker in text:
        block = text.split(marker, 1)[1].split("Netlists do not match", 1)[0]
        for line in block.splitlines():
            if "|" not in line or "Net:" not in line and "no matching net" not in line:
                continue
            left, right = line[:44].strip(), line[44:].strip()
            left_net = _NET_CELL.match(left)
            right_net = _NET_CELL.match(right)
            if left_net or right_net:
                mismatched.append(
                    (left_net.group(1) if left_net else None, right_net.group(1) if right_net else None)
                )
    matched = matched and not mismatched
    return LvsResult(
        matched=matched,
        port_errors=port_errors and matched,
        net_count=net_count,
        mismatched_nets=mismatched,
        pin_mismatches=pin_mismatches if matched else [],
        report=text,
    )


def run_lvs(
    layout_spice: str | os.PathLike[str],
    layout_cell: str,
    reference_spice: str | os.PathLike[str],
    reference_cell: str,
    setup: str | os.PathLike[str] | None = None,
    report: str | os.PathLike[str] | None = None,
    timeout: int = 1800,
    tech=None,
) -> LvsResult:
    """Compare two SPICE netlists with netgen and parse the result."""
    netgen = shutil.which("netgen")
    if netgen is None:
        raise RuntimeError("netgen is not on PATH (enter the nix shell first)")

    setup = Path(setup) if setup else default_netgen_setup(tech=tech)
    report = Path(report) if report else Path(layout_spice).with_suffix(".lvs.txt")

    command = [
        netgen,
        "-batch",
        "lvs",
        f"{layout_spice} {layout_cell}",
        f"{reference_spice} {reference_cell}",
        str(setup),
        str(report),
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    text = report.read_text() if report.exists() else process.stdout
    return _parse_netgen_report(text)


def lvs_layout_vs_netlist(
    gds: str | os.PathLike[str],
    cell: str,
    reference_spice: str | os.PathLike[str],
    reference_cell: str,
    workdir: str | os.PathLike[str] = "build/lvs",
    magicrc: str | os.PathLike[str] | None = None,
    setup: str | os.PathLike[str] | None = None,
    ports: Mapping[str, tuple[float, float]] | None = None,
    tech=None,
) -> LvsResult:
    """Extract ``cell`` from ``gds`` with magic and compare it to a reference netlist (with
    ``tech``'s magic and netgen setups; the default technology's without).

    ``ports``, forwarded to :func:`magic_extract`, declares ``cell``'s top-level pins so
    netgen can verify the macro's pin list instead of just its net topology (see
    :attr:`LvsResult.port_errors`).
    """
    workdir = Path(workdir)
    extracted = magic_extract(gds, cell, workdir, magicrc, ports=ports, tech=tech)
    return run_lvs(extracted, cell, reference_spice, reference_cell, setup, tech=tech)
