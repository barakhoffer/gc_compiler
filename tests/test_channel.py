"""Track assignment for a routing channel (tool-free: no DRC/LVS needed)."""

from __future__ import annotations

import math

from gc_compiler.periphery.channel import Pin, assign_tracks

REACH = 0.4


def _pin(net, x, hangs):
    return Pin(net, x, 0, 0.0, hangs)


def _route(pins, pad_pitch=0.5, tied=lambda net: False):
    return assign_tracks(pins, REACH, pad_pitch, pad=0.37, spacing=0.3, min_len=0.8, tied=tied)


def _rows(channel):
    return {(node.net, tuple(sorted(p.x for p in node.pins))): row for node, row in zip(channel.nodes, channel.rows)}


def _check(pins, channel):
    """Every close pair of stubs from opposite sides needs the top one's row strictly higher."""
    index = {id(pin): i for i, node in enumerate(channel.nodes) for pin in node.pins}
    assert sorted(index) == sorted(id(pin) for pin in pins)
    for above in (p for p in pins if p.hangs):
        for below in (p for p in pins if not p.hangs):
            i, j = index[id(above)], index[id(below)]
            if abs(above.x - below.x) < REACH and i != j:
                assert channel.rows[i] < channel.rows[j], (above, below)
    for i, node in enumerate(channel.nodes):
        lo, hi = min(p.x for p in node.pins), max(p.x for p in node.pins)
        for j in range(i):
            if channel.rows[i] == channel.rows[j]:  # sharing a row: extents must not touch
                other = channel.nodes[j]
                olo, ohi = min(p.x for p in other.pins), max(p.x for p in other.pins)
                assert hi + 0.37 + 0.3 <= olo or ohi + 0.37 + 0.3 <= lo


def test_close_stubs_force_the_order():
    pins = [_pin("a", 0.0, True), _pin("b", 0.1, False), _pin("c", 4.0, True), _pin("d", 4.1, False)]
    channel = _route(pins)
    assert channel.splits == 0
    _check(pins, channel)
    rows = {node.net: row for node, row in zip(channel.nodes, channel.rows)}
    assert rows["a"] < rows["b"] and rows["c"] < rows["d"]


def test_a_cycle_is_broken_by_splitting_a_net():
    # a must sit above b (close pair at x~0) and b above a (close pair at x~5): impossible
    # with one track each, so one of them has to be split into a track per side.
    pins = [
        _pin("a", 0.0, True), _pin("b", 0.1, False),
        _pin("b", 5.0, True), _pin("a", 5.1, False),
    ]
    channel = _route(pins)
    assert channel.splits == 1
    _check(pins, channel)
    assert len(channel.nodes) == 3


def test_same_net_pads_that_are_too_close_are_split():
    pins = [_pin("a", 1.0, True), _pin("a", 1.45, False)]
    assert len(_route(pins, pad_pitch=0.5).nodes) == 2
    assert len(_route(pins, pad_pitch=0.4).nodes) == 1


def test_nets_with_disjoint_extents_share_a_row():
    pins = [_pin(f"n{i}", 10.0 * i, True) for i in range(5)]
    channel = _route(pins)
    assert channel.row_count == 1
    _check(pins, channel)


def test_overlapping_extents_need_separate_rows():
    pins = [_pin("a", 0.0, True), _pin("a", 6.0, True), _pin("b", 3.0, True), _pin("b", 9.0, True)]
    channel = _route(pins)
    assert channel.row_count == 2
    _check(pins, channel)


def test_a_tied_net_reserves_its_row_out_to_the_trunk():
    pins = [_pin("wide", 0.0, True), _pin("late", 20.0, True)]
    assert _route(pins).row_count == 1
    # "wide" now runs to a trunk beyond everything, so "late" cannot share its row.
    assert _route(pins, tied=lambda net: net == "wide").row_count == 2


def test_sharing_never_breaks_the_constraints_on_a_dense_channel():
    # a diagonal of tops against bottoms: each top pin close to the next bottom pin
    pins = []
    for i in range(30):
        pins.append(_pin(f"t{i}", 2.1 * i, True))
        pins.append(_pin(f"b{i}", 2.1 * i + 0.2, False))
    channel = _route(pins)
    _check(pins, channel)
    assert channel.row_count < 2 * 30
    assert all(math.isfinite(row) for row in channel.rows)
