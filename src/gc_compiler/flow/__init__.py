"""The digital flow for the refresh controller: synthesis to the process's standard cells
(:mod:`.synth`), static timing (:mod:`.sta`), and place and route (:mod:`.pnr`).

The tools are yosys, OpenSTA (``sta``), OpenROAD and iverilog; every function here writes its
scripts and logs into a work directory and raises :class:`FlowError` with the tool's log when a
run fails.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FlowError(RuntimeError):
    """A flow tool failed; the message ends with the tail of its log."""


def require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise FlowError(f"{tool} is not on PATH (enter the nix shell first)")
    return path


def run(argv: list[str], workdir: Path, log: str, timeout: float = 3600) -> str:
    """Run a tool in ``workdir``, keep its output in ``log``, and return it."""
    workdir.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(argv, cwd=workdir, capture_output=True, text=True, timeout=timeout)
    text = process.stdout + process.stderr
    (workdir / log).write_text(text)
    if process.returncode != 0:
        raise FlowError(f"{Path(argv[0]).name} failed ({workdir / log}):\n{text[-3000:]}")
    return text
