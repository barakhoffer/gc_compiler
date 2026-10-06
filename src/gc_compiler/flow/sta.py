"""Static timing of a routed block with OpenSTA, for what its constraints cannot express."""

from __future__ import annotations

import re
from pathlib import Path

from gc_compiler.flow import FlowError, require, run
from gc_compiler.tech.rules import Technology


def loopback_arrival(
    tech: Technology,
    corner: str,
    netlist: Path,
    top: str,
    sdc: Path,
    spef: Path,
    macro_liberty: Path,
    workdir: str | Path,
    instance: str = "macro",
) -> float:
    """The latest arrival (ns after the launching clock edge) at the macro's ``din`` pins of any
    path through its ``dout`` pins: the refresh loop-back, from the flop that raised ``re``,
    through the macro's read and the controller's multiplexer, with the routed parasitics.

    The macro's ``din`` pins have no timing check (a user write changes them as ``we`` rises,
    which is harmless: the cell captures as ``we`` falls), so no constraint can end on them; this
    reports the path's arrival for the caller to hold against its own budget."""
    lib = tech.standard_cells()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    script = "\n".join([
        f"read_liberty {lib.liberty[corner]}",
        f"read_liberty {Path(macro_liberty).resolve()}",
        f"read_verilog {Path(netlist).resolve()}",
        f"link_design {top}",
        f"read_spef {Path(spef).resolve()}",
        f"read_sdc {Path(sdc).resolve()}",
        "set_propagated_clock [all_clocks]",
        f"report_checks -path_delay max -through [get_pins {instance}/dout*] -to [get_pins {instance}/din*] "
        "-unconstrained -digits 4",
        "",
    ])
    (workdir / f"loopback_{corner}.tcl").write_text(script)
    out = run([require("sta"), "-no_splash", "-exit", f"loopback_{corner}.tcl"], workdir, f"loopback_{corner}.log")
    arrivals = [float(v) for v in re.findall(r"^\s*(-?[\d.]+)\s+data arrival time", out, re.M)]
    if not arrivals:
        raise FlowError(f"no loop-back path found ({workdir / f'loopback_{corner}.log'}):\n{out[-2000:]}")
    return max(arrivals)
