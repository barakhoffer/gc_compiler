"""Process facts stay behind the technology: no sky130 names, and no bare dimension numbers, in
the compiler outside ``gc_compiler/tech``.

Counted per file, in code (docstrings and comments are free to explain):

* strings naming sky130 or SG13G2 things (``sg13``, ``Metal1``.., ``TopMetal``, ``Activ``, ``GatPoly``,
  ``pSD``/``nSD``), and for sky130: ``sky130``, LEF layer names (``li1``, ``met1``..), standard-cell
  pins (``VPWR``/``VGND``), device model fragments (``nfet``, ``pfet``, ``cap_var``), deck and
  setup file names;
* float literals other than a few neutral ones (0.5, 1, 2, unit factors and tolerances): a bare
  0.375 in a layout builder is almost always a sky130 dimension.

The counts are a ratchet: they may only go down. ``GC_UPDATE_GOLDEN=1`` records lower counts;
raising one needs ``GC_ALLOW_INCREASE=1`` as well, and a reason. A line can opt out with a
``# tech-neutral: <reason>`` comment.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "gc_compiler"
BASELINE = Path(__file__).parent / "golden" / "tech_neutral_baseline.json"

NAMES = re.compile(
    r"sky130|\bli1\b|\bmet[1-5]\b|VPWR|VGND|nfet|pfet|cap_var|\bnom_(?!voltage|temperature|process)|magicrc|_mr\.drc"
    r"|_setup\.tcl|sg13|\bMetal[1-5]\b|TopMetal|\bActiv\b|GatPoly|\bpSD\b|\bnSD\b"
)
NEUTRAL_FLOATS = {0.0, 0.5, 1.0, 2.0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e3, 1e6, 1e9, 1e12, 1e15, 100.0}


def _docstrings(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
        # attribute docstrings: a string statement right after an assignment
        if hasattr(node, "body") and isinstance(node.body, list):
            for stmt in node.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    ids.add(id(stmt.value))
    return ids


def scan(path: Path) -> dict[str, int]:
    text = path.read_text()
    lines = text.splitlines()
    tree = ast.parse(text)
    docs = _docstrings(tree)
    names = floats = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in docs:
            continue
        line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        if "# tech-neutral:" in line:
            continue
        if isinstance(node.value, str) and NAMES.search(node.value):
            names += 1
        elif isinstance(node.value, float) and abs(node.value) not in NEUTRAL_FLOATS:
            floats += 1
    return {"names": names, "floats": floats}


def current() -> dict[str, dict[str, int]]:
    out = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("tech/"):
            continue
        counts = scan(path)
        if any(counts.values()):
            out[rel] = counts
    return out


def test_no_new_process_facts_outside_the_technology():
    now = current()
    if os.environ.get("GC_UPDATE_GOLDEN") == "1":
        old = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        grown = [
            f"{f}: {k} {old.get(f, {}).get(k, 0)} -> {v}"
            for f, c in now.items() for k, v in c.items() if old and v > old.get(f, {}).get(k, 0)
        ]
        if grown and os.environ.get("GC_ALLOW_INCREASE") != "1":
            raise AssertionError("counts went up (set GC_ALLOW_INCREASE=1 with a reason):\n" + "\n".join(grown))
        BASELINE.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n")
        return
    base = json.loads(BASELINE.read_text())
    problems = []
    for f in sorted(set(now) | set(base)):
        for k in ("names", "floats"):
            v, b = now.get(f, {}).get(k, 0), base.get(f, {}).get(k, 0)
            if v > b:
                problems.append(f"{f}: {v} {k}, baseline {b} -- a process fact outside tech/?")
            elif v < b:
                problems.append(f"{f}: {v} {k}, baseline {b} -- fewer now: record it (GC_UPDATE_GOLDEN=1)")
    assert not problems, "\n".join(problems)
