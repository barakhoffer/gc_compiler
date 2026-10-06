"""Small geometry helpers shared by the layout builders."""

from __future__ import annotations

from dataclasses import dataclass

import gdsfactory as gf

from gc_compiler.tech.rules import Layer


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def grow(self, dx: float, dy: float | None = None) -> Box:
        dy = dx if dy is None else dy
        return Box(self.x0 - dx, self.y0 - dy, self.x1 + dx, self.y1 + dy)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @staticmethod
    def centered(cx: float, cy: float, width: float, height: float) -> Box:
        return Box(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)

    @staticmethod
    def hull(boxes: list[Box]) -> Box:
        return Box(
            min(b.x0 for b in boxes),
            min(b.y0 for b in boxes),
            max(b.x1 for b in boxes),
            max(b.y1 for b in boxes),
        )


def draw(component: gf.Component, layer: Layer, box: Box) -> Box:
    """Add an axis-aligned rectangle and return it, for chaining."""
    component.add_polygon(
        [(box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1)],
        layer=layer,
    )
    return box


def cut_positions(available: float, size: float, pitch: float) -> list[float]:
    """Centers (relative to the middle of ``available``) of a maximal cut array."""
    if available < size - 1e-9:  # (a region exactly one cut wide, less rounding)
        return []
    n = int((available - size) / pitch + 1e-9) + 1
    span = (n - 1) * pitch
    return [-span / 2 + i * pitch for i in range(n)]


def add_port(component: gf.Component, *, width: float | None = None, **kwargs) -> None:
    """``component.add_port`` with the width rounded up to an even number of database units, which
    gdsfactory requires (a process's rule widths need not be: 0.175 um is 175 units). Without a
    width (a copied ``port=``) it is passed on as it is."""
    if width is None:
        component.add_port(**kwargs)
        return
    dbu = component.kcl.layout.dbu
    units = round(width / dbu - 1e-6)
    units += units % 2
    component.add_port(width=units * dbu, **kwargs)
