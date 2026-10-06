"""Cell cache.

gdsfactory registers every component by name in a global layout, so building the same
cell twice in one session raises. Cells here are pure functions of their name (the name
encodes the configuration) and of the technology, so the cache is keyed on both: each
technology's :attr:`~gc_compiler.tech.rules.Technology.cell_prefix` goes in front of the name,
so two technologies' cells never meet under one name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import gdsfactory as gf

if TYPE_CHECKING:
    from gc_compiler.tech.rules import Technology

_CELLS: dict[str, gf.Component] = {}


def cached_cell(name: str, builder: Callable[[str], gf.Component], tech: Technology) -> gf.Component:
    """Return ``tech``'s cell called ``name``, building it on first use."""
    name = f"{tech.cell_prefix}{name}"
    cell = _CELLS.get(name)
    if cell is None or cell.kcl is not gf.kcl:
        cell = builder(name)
        _CELLS[name] = cell
    return cell


def clear_cache() -> None:
    _CELLS.clear()
