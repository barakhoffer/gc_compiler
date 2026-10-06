"""Gate-level simulation: a generated testbench run with a synthesized netlist in place of its
RTL (iverilog, with the standard cells' functional models)."""

from __future__ import annotations

from pathlib import Path

from gc_compiler.flow import FlowError, require, run
from gc_compiler.tech.rules import Technology


def simulate_gates(
    tech: Technology, netlist: Path, sources: list[Path], workdir: str | Path, check: bool = True
) -> str:
    """Run a testbench with ``netlist`` in place of its RTL: ``sources`` are the other files
    (behavioural model, top level, testbench). Returns the simulator's output; ``check`` raises
    unless the testbench printed PASS."""
    iverilog, vvp = require("iverilog"), require("vvp")
    lib = tech.standard_cells()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    files = [*lib.verilog, str(Path(netlist).resolve()), *(str(Path(s).resolve()) for s in sources)]
    defines = [f"-D{d}" for d in lib.verilog_defines]
    run([iverilog, "-g2012", *defines, "-o", "gates.vvp", *files], workdir, "iverilog.log")
    out = run([vvp, "gates.vvp"], workdir, "vvp.log")
    if check and "PASS" not in out:
        raise FlowError(f"the gate-level testbench did not pass ({workdir / 'vvp.log'}):\n{out[-2000:]}")
    return out
