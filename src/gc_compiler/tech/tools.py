"""Where the process's sign-off tool setups are, and what their rules are called."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ToolSetup:
    """The PDK's files for KLayout DRC, magic (DRC, extraction) and netgen (LVS), relative to its
    install directory (``<PDK_ROOT>/<variant>``), and the names of rules a check waives."""

    variant: str
    """The PDK's install directory under the PDK root (open_pdks: ``sky130A``)."""
    klayout_deck: str
    """The KLayout DRC deck."""
    magicrc: str
    netgen_setup: str
    klayout_deck_vars: dict[str, str] = field(default_factory=dict)
    """``-rd`` variables the deck takes."""
    klayout_top_cell_var: str = "top_cell"
    """The deck's ``-rd`` variable naming the cell to check."""
    klayout_extra_decks: tuple[str, ...] = ()
    """Further KLayout decks the sign-off runs after ``klayout_deck``, with the same variables
    (rules the main deck leaves out)."""
    magic_drc: bool = True
    """Whether magic's DRC deck is part of the sign-off (it is always used for extraction)."""
    magic_drc_style: str = "drc(full)"
    context_waivers: tuple[str, ...] = ()
    """magic rules only a complete block can meet (every well tapped, every diffusion within
    latch-up distance of a tap): what a check of a leaf cell waives."""
    leaf_waivers: tuple[str, ...] = ()
    """What a leaf cell checked on its own cannot meet either (wells that end at its edges for the
    neighbours' to continue), on top of ``context_waivers``."""

    def pdk_root(self, root: str | os.PathLike[str] | None = None) -> Path:
        return Path(root or os.environ.get("PDK_ROOT", Path.home() / ".ciel"))

    def path(self, relative: str, root: str | os.PathLike[str] | None = None) -> Path:
        return self.pdk_root(root) / self.variant / relative

    def klayout_deck_path(self, root=None) -> Path:
        return self.path(self.klayout_deck, root)

    def klayout_extra_deck_paths(self, root=None) -> list[Path]:
        return [self.path(deck, root) for deck in self.klayout_extra_decks]

    def magicrc_path(self, root=None) -> Path:
        return self.path(self.magicrc, root)

    def netgen_setup_path(self, root=None) -> Path:
        return self.path(self.netgen_setup, root)
