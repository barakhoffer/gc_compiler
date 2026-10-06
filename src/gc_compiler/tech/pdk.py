"""Bridge between a :class:`~gc_compiler.tech.rules.Technology` and gdsfactory's PDK.

gdsfactory needs an active PDK to resolve layers and to colour plots; this module builds
one on the fly from whatever technology adapter is in use.
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.technology import LayerMap, LayerView, LayerViews

from gc_compiler.tech.rules import LayerRole, Technology

_ACTIVE: dict[str, gf.Pdk] = {}


def layer_map(tech: Technology) -> LayerMap:
    """Build a gdsfactory LayerMap enum from the technology's layer roles."""
    entries = {role.name: tech.layer(role) for role in LayerRole if tech.has(role)}
    for role in (LayerRole.LI, LayerRole.M1, LayerRole.M2, LayerRole.M3, LayerRole.M4, LayerRole.M5, LayerRole.POLY):
        label = tech.label_layer(role)
        if label is not None:
            entries[f"{role.name}_LABEL"] = label
        pin = tech.pin_layer(role)
        if pin is not None:
            entries[f"{role.name}_PIN"] = pin
    return LayerMap(f"{tech.name.upper()}_LAYERS", entries)


def layer_views(tech: Technology) -> LayerViews:
    views = {}
    for role, style in tech.styles.items():
        if not tech.has(role):
            continue
        layer, datatype = tech.layer(role)
        views[role.name] = LayerView(
            layer=(layer, datatype),
            name=role.name,
            fill_color=style.color,
            frame_color=style.color,
            hatch_pattern=style.hatch,
            transparent=style.transparent,
        )
    return LayerViews(layer_views=views)


def activate(tech: Technology) -> gf.Pdk:
    """Create (once) and activate a gdsfactory PDK mirroring ``tech``."""
    if tech.name not in _ACTIVE:
        _ACTIVE[tech.name] = gf.Pdk(
            name=f"gc_{tech.name}",
            layers=layer_map(tech),
            layer_views=layer_views(tech),
        )
    pdk = _ACTIVE[tech.name]
    pdk.activate()
    return pdk
