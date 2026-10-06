"""The CLI's argument resolution: the default cell and configuration, ``--plain-cell``, and the
opt-outs of the automatic choices."""

from __future__ import annotations

import pytest

from gc_compiler.cli import build_parser, resolve_bitcell
from gc_compiler.config import WIDE_ARRAY, ArrayConfig, BitcellConfig


def _bitcell(*argv: str) -> BitcellConfig:
    return resolve_bitcell(build_parser().parse_args(list(argv)))


def test_the_default_cell_is_the_lvt_write_with_a_long_mr():
    cell = _bitcell()
    assert cell == BitcellConfig()
    assert (cell.write_flavor, cell.write_length, cell.read_length, cell.mr_length) == ("lvt", 0.35, 0.15, 0.5)


def test_plain_cell_is_the_old_default():
    cell = _bitcell("--plain-cell")
    assert cell == BitcellConfig.plain()
    assert (cell.write_flavor, cell.write_length, cell.mr_length) == (None, 0.18, None)
    assert cell.effective_mr_length == cell.read_length


def test_explicit_geometry_overrides_either_cell():
    cell = _bitcell("--write-length", "0.4", "--mr-length", "0.6")
    assert (cell.write_length, cell.mr_length, cell.write_flavor) == (0.4, 0.6, "lvt")
    plain = _bitcell("--plain-cell", "--read-length", "0.5")
    assert plain.effective_mr_length == 0.5 and plain.write_length == 0.18
    assert _bitcell("--write-flavor", "svt").write_flavor is None
    assert _bitcell("--plain-cell", "--write-flavor", "hvt").write_flavor == "hvt"


def test_array_defaults_are_automatic():
    small, wide = ArrayConfig(cols=WIDE_ARRAY - 1), ArrayConfig(cols=WIDE_ARRAY)
    assert small.replica_column and small.self_timed_read
    assert not small.wordline_straps and small.wordline_driver_strength == 1
    assert wide.wordline_straps and wide.wordline_driver_strength == 4
    bare = ArrayConfig(replica_column=False)
    assert not bare.self_timed_read
    with pytest.raises(ValueError, match="replica"):
        ArrayConfig(replica_column=False, self_timed_read=True)
    assert not ArrayConfig(cols=WIDE_ARRAY, strap_column_interval=0).wordline_straps


def test_opt_out_flags_parse():
    args = build_parser().parse_args(["--no-replica", "--no-self-timed", "--no-wl-straps", "--one-bank"])
    assert (args.replica, args.self_timed, args.wl_straps, args.one_bank) == (False, False, False, True)
    args = build_parser().parse_args([])
    assert (args.replica, args.self_timed, args.wl_straps, args.wl_driver_strength) == (True, None, None, None)
