"""DRC: the PDK's KLayout deck and magic's full deck, both required.

The KLayout ``mr`` deck is a subset of the rules: it misses, among others, poly-contact and poly
spacing to diffusion (licon.14, poly.4, licon.9), well enclosure of taps (diff/tap.10/.8), the
deep n-well rules (dnwell.5, nwell.7) and every varactor rule. magic's ``drc(full)`` style has
them, so :func:`run_drc` runs both and reports magic's rules as ``magic:<rule>``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


def _tech(tech):
    if tech is None:
        from gc_compiler.tech import default_technology

        return default_technology()
    return tech


def default_deck(pdk_root: str | os.PathLike[str] | None = None, tech=None) -> Path:
    """The technology's KLayout DRC deck (see :class:`~gc_compiler.tech.tools.ToolSetup`)."""
    return _tech(tech).tools.klayout_deck_path(pdk_root)


@dataclass
class DrcResult:
    gds: Path
    report: Path
    by_rule: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.by_rule.values())

    @property
    def clean(self) -> bool:
        return self.total == 0

    def __bool__(self) -> bool:
        return self.clean

    def __str__(self) -> str:
        if self.clean:
            return f"DRC clean: {self.gds.name}"
        lines = [f"DRC violations in {self.gds.name}: {self.total}"]
        lines += [f"  {rule}: {count}" for rule, count in sorted(self.by_rule.items())]
        return "\n".join(lines)


def parse_report(report: Path) -> dict[str, int]:
    """Count violations per rule in a KLayout ``.lyrdb`` report."""
    if not report.exists():
        raise FileNotFoundError(report)
    counts: Counter[str] = Counter()
    for _, element in ET.iterparse(report, events=("end",)):
        if element.tag != "item":
            continue
        category = element.findtext("category", default="").strip("'")
        counts[category] += 1
        element.clear()
    return dict(counts)


def context_rules(tech=None) -> tuple[str, ...]:
    """``tech``'s rules that only hold for a complete block (every n-well tapped, every diffusion
    within latch-up distance of a tap), which a leaf cell checked alone cannot meet: what a check
    of a leaf cell waives (``tech.tools.context_waivers``)."""
    return _tech(tech).tools.context_waivers


def leaf_rules(tech=None) -> tuple[str, ...]:
    """What a leaf cell checked on its own cannot meet: :func:`context_rules`, and the rules its
    wells ending at its edges for its neighbours' to continue break (``tech.tools.leaf_waivers``).
    The assembled macros are checked without waivers."""
    tools = _tech(tech).tools
    return tools.context_waivers + tools.leaf_waivers


#: :func:`context_rules` and :func:`leaf_rules` of the default technology.
CONTEXT_RULES = context_rules()
LEAF_RULES = leaf_rules()


def run_magic_drc(
    gds: str | os.PathLike[str], top_cell: str | None = None, workdir: str | os.PathLike[str] | None = None,
    timeout: int = 3600, tech=None,
) -> dict[str, int]:
    """magic's full DRC deck on ``gds`` (flattened first, so each shape is checked in the context
    it is placed in), the way LibreLane runs it: violations per rule, keyed by the rule's name
    (``licon.14``). Coordinates are in ``<cell>.magic.rpt`` next to the GDS."""
    import klayout.db as kdb

    tools = _tech(tech).tools
    magic = shutil.which("magic")
    if magic is None:
        raise RuntimeError("magic is not on PATH (enter the nix shell first)")
    gds = Path(gds).resolve()
    layout = kdb.Layout()
    layout.read(str(gds))
    top = layout.cell(top_cell) if top_cell else layout.top_cell()
    top.flatten(-1, True)
    top.name = re.sub(r"\W", "_", top.name)  # magic's load does not take every GDS name
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="magic_drc_"))
    workdir.mkdir(parents=True, exist_ok=True)
    flat = workdir / f"{top.name}.flat.gds"
    options = kdb.SaveLayoutOptions()
    options.write_context_info = False
    layout.write(str(flat), options)
    report = gds.with_name(f"{gds.stem}.magic.rpt")
    report.unlink(missing_ok=True)
    script = workdir / "drc.tcl"
    script.write_text(
        f"""crashbackups disable
units microns
gds noduplicates true
gds readonly true
gds read {flat}
load {top.name}
select top cell
drc euclidean on
drc style {tools.magic_drc_style}
drc check
set out [open {report} w]
foreach {{why boxes}} [drc listall why] {{
    puts $out "RULE\t[llength $boxes]\t$why"
    foreach b $boxes {{ puts $out "\t$b" }}
}}
close $out
quit -noprompt
"""
    )
    # no stdin: a Tcl error would otherwise leave magic waiting at its prompt
    process = subprocess.run(
        [magic, "-dnull", "-noconsole", "-rcfile", str(tools.magicrc_path()), str(script)],
        cwd=workdir, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
    )
    if not report.exists():
        raise RuntimeError(f"magic DRC failed:\n{process.stdout[-3000:]}\n{process.stderr[-2000:]}")
    counts: Counter[str] = Counter()
    for line in report.read_text().splitlines():
        if line.startswith("RULE\t"):
            _tag, count, why = line.split("\t", 2)
            match = re.search(r"\(([^()]*)\)\s*$", why)
            counts[match.group(1) if match else why] += int(count)
    return dict(counts)


def run_drc(
    gds: str | os.PathLike[str],
    deck: str | os.PathLike[str] | None = None,
    report: str | os.PathLike[str] | None = None,
    deck_vars: dict[str, str] | None = None,
    top_cell: str | None = None,
    timeout: int = 3600,
    magic: bool | None = None,
    waive: tuple[str, ...] = (),
    tech=None,
) -> DrcResult:
    """Run the technology's KLayout DRC deck (and its extra decks) and (``magic``, default
    ``tech.tools.magic_drc``) magic's full deck on ``gds``.
    ``deck_vars`` overrides the deck's ``-rd`` variables (``tech.tools.klayout_deck_vars``).
    ``waive`` lists rules not to count, in either deck (both name them by the design manual) --
    ``leaf_rules(tech)`` for a leaf cell checked on its own."""
    tech = _tech(tech)
    klayout = shutil.which("klayout")
    if klayout is None:
        raise RuntimeError("klayout is not on PATH (enter the nix shell first)")

    gds = Path(gds).resolve()
    deck = Path(deck) if deck else default_deck(tech=tech)
    if not deck.exists():
        raise FileNotFoundError(f"DRC deck not found: {deck}; is PDK_ROOT set?")
    report = Path(report) if report else gds.with_suffix(".lyrdb")

    variables = {**tech.tools.klayout_deck_vars, **(deck_vars or {})}
    by_rule: dict[str, int] = {}
    decks = [(deck, report)] + [
        (extra, report.with_name(f"{report.stem}.{extra.stem}{report.suffix}"))
        for extra in tech.tools.klayout_extra_deck_paths()
    ]
    for path, out in decks:
        command = [klayout, "-b", "-r", str(path), "-rd", f"input={gds}", "-rd", f"report={out}"]
        for name, value in variables.items():
            command += ["-rd", f"{name}={value}"]
        if top_cell:
            command += ["-rd", f"{tech.tools.klayout_top_cell_var}={top_cell}"]
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if process.returncode != 0:
            raise RuntimeError(f"klayout DRC failed:\n{process.stdout[-4000:]}\n{process.stderr[-2000:]}")
        for rule, count in parse_report(out).items():
            if rule not in waive:
                by_rule[rule] = by_rule.get(rule, 0) + count
    if magic if magic is not None else tech.tools.magic_drc:
        for rule, count in run_magic_drc(gds, top_cell, timeout=timeout, tech=tech).items():
            if rule not in waive:
                by_rule[f"magic:{rule}"] = count
    return DrcResult(gds=gds, report=report, by_rule=by_rule)
