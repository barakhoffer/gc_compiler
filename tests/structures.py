"""Reference structures built with a given technology: what the layout regression pins (sky130)
and what the portability tests check against each technology's own rules."""

from __future__ import annotations

from gc_compiler.array.floorplan import gc_banked_macro, gc_macro
from gc_compiler.array.tiler import gc_array
from gc_compiler.cells.bitcell_3t import gain_cell_3t
from gc_compiler.config import ArrayConfig, BitcellConfig
from gc_compiler.flow.abstract import hard_macro
from gc_compiler.periphery.logic import LogicConfig

#: Structures a leaf cell's wells end at the edges of (checked with ``leaf=True``).
LEAF = {"bitcell_default", "bitcell_plain", "array_straps"}


def reference_structures(tech, prefix: str = "rg") -> dict:
    """``{name: builder}``: each builder returns a fresh component (cell names start ``prefix``)."""
    logic = LogicConfig.for_tech(tech)
    # (the reference layouts keep the loop-back refresh they were recorded with; one asks for the in-macro one)
    A = lambda **kw: ArrayConfig.for_tech(tech, **{"refresh_in_macro": False, **kw})  # noqa: E731
    basic = lambda **kw: ArrayConfig.basic(tech, **kw)  # noqa: E731
    return {
        "bitcell_default": lambda: gain_cell_3t(tech, BitcellConfig.for_tech(tech), name=f"{prefix}_gc3t"),
        "bitcell_plain": lambda: gain_cell_3t(tech, BitcellConfig.plain(tech), name=f"{prefix}_gc3t_plain"),
        "array_straps": lambda: gc_array(
            tech, basic(rows=8, cols=20, wordline_straps=True), name=f"{prefix}_array_straps"
        ),
        "macro_basic_undecoded": lambda: gc_macro(tech, basic(rows=6, cols=5), logic, name=f"{prefix}_undecoded"),
        "macro_basic_decoded": lambda: gc_macro(
            tech, basic(rows=8, cols=6), logic, name=f"{prefix}_basic_dec", decoded=True
        ),
        "macro_default_decoded": lambda: gc_macro(
            tech, A(rows=8, cols=8), logic, name=f"{prefix}_default_dec", decoded=True
        ),
        "macro_banked": lambda: gc_banked_macro(tech, A(rows=8, cols=8), logic, name=f"{prefix}_banked"),
        "macro_neg_wl": lambda: gc_macro(
            tech, basic(rows=4, cols=3, negative_write_wl=True), logic, name=f"{prefix}_neg", decoded=True
        ),
        "macro_wide_banked": lambda: gc_banked_macro(tech, A(rows=4, cols=64), logic, name=f"{prefix}_wide"),
        "macro_neg_strong_straps": lambda: gc_macro(
            tech, basic(rows=4, cols=14, wordline_driver_strength=4, negative_write_wl=True, replica_column=True,
                        wordline_straps=True),
            logic, name=f"{prefix}_neg_strong", decoded=True,
        ),
        "macro_banked_strong_straps": lambda: gc_banked_macro(
            tech, basic(rows=4, cols=10, wordline_driver_strength=4, wordline_straps=True), logic,
            name=f"{prefix}_banked_strong",
        ),
        "macro_refresh_in_macro": lambda: gc_macro(
            tech, A(rows=4, cols=6, refresh_in_macro=True), logic, name=f"{prefix}_rim", decoded=True
        ),
        "hard_macro_banked": lambda: hard_macro(
            tech, gc_banked_macro(tech, A(rows=6, cols=5), logic, name=f"{prefix}_hard_core"),
            A(rows=6, cols=5), f"{prefix}_hard", True,
        ).component,
    }
