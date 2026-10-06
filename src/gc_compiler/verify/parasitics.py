"""Parasitic extraction with magic.

``ext2spice`` with a capacitance threshold writes, next to every device, the wiring
capacitances between nets and to the substrate, so the result can be simulated as it is. The
flow is the LVS one (:mod:`gc_compiler.verify.lvs`) without its ``lvs`` netlist mode and with
hierarchy flattened, so nets keep the names the layout's labels gave them.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from gc_compiler.verify.lvs import default_magicrc

_SCRIPT = """\
gds read {gds}
::magic::load {cell}
select top cell
extract path {extdir}
extract all
ext2spice cthresh {cthresh}
ext2spice format ngspice
ext2spice hierarchy off
ext2spice -p {extdir} -o {out}
quit -noprompt
"""


def extract_flat(
    gds: str | Path,
    cell: str,
    workdir: str | Path,
    cthresh: float = 0.02,
    magicrc: str | Path | None = None,
    timeout: int = 1800,
    tech=None,
) -> Path:
    """Extract ``cell`` with parasitic capacitances into one flat SPICE file.

    ``cthresh`` (fF) drops capacitors smaller than it, which keeps the netlist a few times
    smaller at a negligible cost in accuracy. Returns the path of the netlist.
    """
    magic = shutil.which("magic")
    if magic is None:
        raise RuntimeError("magic is not on PATH (enter the nix shell first)")
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    extdir = workdir / "ext"
    extdir.mkdir(exist_ok=True)
    out = workdir / f"{cell}.flat.spice"
    script = workdir / "extract_parasitics.tcl"
    script.write_text(
        _SCRIPT.format(gds=Path(gds).resolve(), cell=cell, extdir=extdir, out=out, cthresh=cthresh)
    )
    rc = Path(magicrc) if magicrc else default_magicrc(tech=tech)
    process = subprocess.run(
        [magic, "-dnull", "-noconsole", "-rcfile", str(rc), str(script)],
        cwd=workdir, capture_output=True, text=True, timeout=timeout,
    )
    if not out.exists():
        raise RuntimeError(f"magic extraction failed:\n{process.stdout[-3000:]}\n{process.stderr[-2000:]}")
    return out
