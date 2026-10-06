"""Layout regression: the geometry of a set of reference structures, pinned.

Each structure is flattened and every layer's merged polygons and every label are hashed, so a
refactor that must not move anything (turning a literal into the rule it stood for) is checked
exactly, while one that only reorganises shapes (a different overlap inside a shape it joins) is
not flagged. ``GC_UPDATE_GOLDEN=1`` rewrites the golden file after a deliberate change.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import klayout.db as kdb
import pytest

from gc_compiler.tech.sky130 import Sky130
from structures import reference_structures

GOLDEN = Path(__file__).parent / "golden" / "layout_digests.json"
UPDATE = os.environ.get("GC_UPDATE_GOLDEN") == "1"


def layout_digest(component) -> dict[str, str]:
    """``{"layer/datatype": hash}`` of the merged, flattened geometry, and one entry for the labels."""
    layout = kdb.Layout()
    layout.dbu = component.kcl.layout.dbu
    top = layout.create_cell("top")
    top.copy_tree(component.kcl.layout.cell(component.name))
    top.flatten(-1)
    digest = {}
    labels = []
    for index in layout.layer_indexes():
        info = layout.get_info(index)
        shapes = top.shapes(index)
        for shape in shapes.each(kdb.Shapes.STexts):
            t = shape.text
            labels.append(f"{info.layer}/{info.datatype} {t.string} {t.x} {t.y}")
        region = kdb.Region(shapes).merged()
        if region.is_empty():
            continue
        polygons = sorted(str(p) for p in region.each())
        digest[f"{info.layer}/{info.datatype}"] = hashlib.sha256("\n".join(polygons).encode()).hexdigest()[:16]
    digest["labels"] = hashlib.sha256("\n".join(sorted(labels)).encode()).hexdigest()[:16]
    return digest


STRUCTURES = reference_structures(Sky130())


@pytest.fixture(scope="module")
def golden() -> dict:
    data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    yield data
    if UPDATE:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


@pytest.mark.parametrize("name", list(STRUCTURES))
def test_layout_is_unchanged(golden, name):
    digest = layout_digest(STRUCTURES[name]())
    if UPDATE:
        golden[name] = digest
        return
    assert name in golden, f"no golden digest for {name}: run with GC_UPDATE_GOLDEN=1"
    changed = sorted(k for k in set(digest) | set(golden[name]) if digest.get(k) != golden[name].get(k))
    assert not changed, f"{name}: layers changed: {changed}"
