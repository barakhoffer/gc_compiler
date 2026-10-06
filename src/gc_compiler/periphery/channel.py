"""Track assignment for one routing channel between two rows of pins.

A channel is a horizontal strip with pins hanging into it from above and pins standing up
into it from below. Every net gets metal2 stubs (vertical) from its pins to a metal3 track
(horizontal); nets whose x-extents do not overlap may share a track row.

Two stubs from opposite sides can only touch if the track of the net hanging from above
sits *below* the track of the net standing up from below, and then only if their pins are
closer than ``reach`` in x (see :func:`assign_tracks`). So each such close pair is a
constraint "this net's row is above that one's". Everything else is free, which is what
makes the result provably short-free without a maze router.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Pin:
    """One pin as seen from the channel it opens into."""

    net: str
    x: float
    strip: int
    """Index of the strip the pin belongs to, or -1 for a pin at an absolute ``local_y``."""
    local_y: float
    hangs: bool
    """True for a pin above the channel, whose stub therefore goes down."""


@dataclass
class Node:
    """One horizontal track: the pins of one net on one or both sides of one channel."""

    net: str
    pins: list[Pin]

    @property
    def rank(self) -> int:
        """Placement priority: tracks fed only from above first, only from below last."""
        sides = {pin.hangs for pin in self.pins}
        return 0 if sides == {True} else 2 if sides == {False} else 1


@dataclass
class Channel:
    """The result of :func:`assign_tracks`: nodes and the row (0 = top) each sits on."""

    nodes: list[Node] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)
    row_count: int = 0
    splits: int = 0


def check_same_side_pitch(pins: list[Pin], stub_pitch: float, pad_pitch: float) -> None:
    """Pins on one side of a channel are spread by the cells (or driver pitch) themselves;
    refuse to route if that assumption -- which the constraint graph relies on -- breaks.
    """
    for hangs in (True, False):
        side = sorted((pin for pin in pins if pin.hangs is hangs), key=lambda pin: pin.x)
        for left, right in zip(side, side[1:]):
            need = pad_pitch if left.net == right.net else stub_pitch
            if right.x - left.x < need:
                raise ValueError(
                    f"pins {left.net!r} and {right.net!r} are {right.x - left.x:.3f} um apart "
                    f"on one side of a channel; at least {need:.3f} um is required"
                )


def _topological(edges: list[set[int]], nodes: list[Node]) -> list[int] | None:
    """Order nodes top to bottom (``edges[i]`` must sit below ``i``); ``None`` on a cycle."""
    indegree = [0] * len(nodes)
    for below in edges:
        for j in below:
            indegree[j] += 1
    ready = [i for i, count in enumerate(indegree) if count == 0]
    order: list[int] = []
    while ready:
        ready.sort(key=lambda i: (nodes[i].rank, nodes[i].net), reverse=True)
        i = ready.pop()
        order.append(i)
        for j in edges[i]:
            indegree[j] -= 1
            if indegree[j] == 0:
                ready.append(j)
    return order if len(order) == len(nodes) else None


def _cycle_candidate(edges: list[set[int]], nodes: list[Node]) -> int:
    """A node on (or between) the cycles left once every acyclic node is peeled away."""
    alive = set(range(len(nodes)))
    changed = True
    while changed:
        changed = False
        for i in sorted(alive):
            has_out = bool(edges[i] & alive)
            has_in = any(i in edges[j] for j in alive)
            if not (has_out and has_in):
                alive.discard(i)
                changed = True
    splittable = [i for i in alive if len({pin.hangs for pin in nodes[i].pins}) == 2]
    if not splittable:
        raise ValueError("routing constraint cycle with nothing left to split")
    return max(splittable, key=lambda i: (len(edges[i]), nodes[i].net))


def assign_tracks(
    pins: list[Pin],
    reach: float,
    pad_pitch: float,
    pad: float,
    spacing: float,
    min_len: float,
    tied: Callable[[str], bool] = lambda net: False,
    tied_left: Callable[[str], bool] = lambda net: False,
) -> Channel:
    """Assign every net in one channel to a track row so no two stubs can touch.

    A stub hanging from above (net A at ``xa``) spans from the channel top down to A's
    row; one standing up from below (net B at ``xb``) spans from the bottom up to B's. They
    can only meet if B's row is not below A's, and then only if ``|xa - xb| < reach``. So a
    close pair forces "A's row strictly above B's". Two pins of the *same* net on opposite
    sides share a track only if their via pads clear each other (``pad_pitch``). A cyclic
    set of constraints is broken by splitting one net into one track per side, which the
    caller must tie back together with a trunk.

    Nets with disjoint x-extents share a row (extents include ``pad`` and need ``spacing``
    between them; a metal3 wire is at least ``min_len`` long). A net that ``tied`` says
    reaches a trunk to the right (or that gets split) is treated as running to infinity, and
    one that ``tied_left`` says reaches a trunk on the left as running to minus infinity.
    """
    by_net: dict[str, list[Pin]] = {}
    for pin in pins:
        by_net.setdefault(pin.net, []).append(pin)
    split: set[str] = set()
    while True:
        nodes: list[Node] = []
        node_of: dict[int, int] = {}
        for net in sorted(by_net):
            groups = [by_net[net]]
            if net in split:
                groups = [[pin for pin in by_net[net] if pin.hangs], [pin for pin in by_net[net] if not pin.hangs]]
            for group in groups:
                for pin in group:
                    node_of[id(pin)] = len(nodes)
                nodes.append(Node(net, group))
        edges: list[set[int]] = [set() for _ in nodes]
        for above in (pin for pin in pins if pin.hangs):
            for below in (pin for pin in pins if not pin.hangs):
                if abs(above.x - below.x) < (pad_pitch if above.net == below.net else reach):
                    edges[node_of[id(above)]].add(node_of[id(below)])
        loops = [i for i, below in enumerate(edges) if i in below]
        if loops:
            split.add(nodes[loops[0]].net)
            continue
        order = _topological(edges, nodes)
        if order is not None:
            break
        split.add(nodes[_cycle_candidate(edges, nodes)].net)

    preds: list[set[int]] = [set() for _ in nodes]
    for i, below in enumerate(edges):
        for j in below:
            preds[j].add(i)

    def extent(node: Node) -> tuple[float, float]:
        xs = [pin.x for pin in node.pins]
        lo, hi = min(xs) - pad / 2, max(xs) + pad / 2
        if tied_left(node.net):  # all its tracks, split or not, reach a trunk on the left
            return float("-inf"), hi
        if node.net in split or tied(node.net):
            return lo, float("inf")
        if hi - lo < min_len:
            mid = (lo + hi) / 2
            lo, hi = mid - min_len / 2, mid + min_len / 2
        return lo, hi

    rows: list[list[tuple[float, float]]] = []
    row_of = [0] * len(nodes)
    for i in order:
        lo, hi = extent(nodes[i])
        first = 1 + max((row_of[p] for p in preds[i]), default=-1)

        def fits(r: int) -> bool:
            return all(hi + spacing <= a or b + spacing <= lo for a, b in rows[r])

        if nodes[i].rank == 2:  # only pins from below: sit as low as the constraints allow
            candidates: range = range(len(rows) - 1, first - 1, -1)
        else:
            candidates = range(first, len(rows))
        row = next((r for r in candidates if fits(r)), None)
        if row is None:
            rows.append([])
            row = len(rows) - 1
        rows[row].append((lo, hi))
        row_of[i] = row

    return Channel(nodes=nodes, rows=row_of, row_count=len(rows), splits=len(split))
