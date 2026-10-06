from gc_compiler.tech.electrical import Corner, Electrical
from gc_compiler.tech.pdk import activate, layer_views
from gc_compiler.tech.policy import LayoutPolicy, PowerStripe
from gc_compiler.tech.rules import (
    CUT_BETWEEN,
    ROUTING_ROLES,
    DesignRules,
    DeviceKind,
    LayerRole,
    LayerStyle,
    StdCellLibrary,
    Technology,
)
from gc_compiler.tech.sg13g2 import SG13G2
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.tech.stack import MetalStack
from gc_compiler.tech.tools import ToolSetup

#: Technologies by name, for ``--tech``; a module path (``package.module:Class``) also works.
TECHNOLOGIES: dict[str, type[Technology]] = {"sky130A": Sky130, "ihp-sg13g2": SG13G2}


def technology(name: str | None = None) -> Technology:
    """The technology called ``name`` (a :data:`TECHNOLOGIES` key or ``module:Class``); the
    default one without a name."""
    if name is None:
        return Sky130()
    if name in TECHNOLOGIES:
        return TECHNOLOGIES[name]()
    if ":" in name:
        import importlib

        module, cls = name.split(":", 1)
        return getattr(importlib.import_module(module), cls)()
    raise KeyError(f"unknown technology {name!r} (have {', '.join(TECHNOLOGIES)}, or give module:Class)")


def default_technology() -> Technology:
    return technology()


__all__ = [
    "CUT_BETWEEN",
    "ROUTING_ROLES",
    "TECHNOLOGIES",
    "Corner",
    "DesignRules",
    "DeviceKind",
    "Electrical",
    "LayerRole",
    "LayerStyle",
    "LayoutPolicy",
    "MetalStack",
    "PowerStripe",
    "SG13G2",
    "Sky130",
    "StdCellLibrary",
    "Technology",
    "ToolSetup",
    "activate",
    "default_technology",
    "layer_views",
    "technology",
]
