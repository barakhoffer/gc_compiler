"""Layout choices a process's rules do not dictate but its adapter settles: how often to tap, how
to lay out the supply pins, how much two joined shapes overlap."""

from __future__ import annotations

from dataclasses import dataclass

from gc_compiler.tech.rules import LayerRole


@dataclass(frozen=True)
class PowerStripe:
    """A hard macro's supply pins: vertical stripes on the stack's power-pin layer."""

    width: float
    pitch: float
    """Between two stripes of the same net."""
    clearance: float
    """Between stripes of different nets (at least the layer's spacing, wide-metal rules included)."""
    site_pitch: float
    """Between the via stacks that tie a stripe down, along it."""
    scan_step: float
    """How finely the free spots for those stacks are searched (a multiple of the grid)."""


@dataclass(frozen=True)
class LayoutPolicy:
    tap_row_interval: int
    """Row pairs between two tap rows (within the latch-up distance at the cell's pitch)."""
    strap_column_interval: int
    """Columns between two strap columns."""
    power_stripe: PowerStripe
    join: float
    """How far one shape runs into another it merges with (any positive amount inside it)."""
    margin: float
    """Slack added where a computed extent must not land exactly on a rule."""
    poly_over_well_is_device: bool = False
    """A bare poly line crossing an n-well is a device (sky130: a varactor), so a word line
    crossing a strap column's n-tap must be cut and bridged, not run over it."""


def check_policy(policy: LayoutPolicy, rules, stripe_role: LayerRole) -> None:
    """Raise if a policy value is below the rule it stands in for."""
    stripe = policy.power_stripe
    if stripe.width < rules.w(stripe_role) or stripe.clearance < rules.s(stripe_role):
        raise ValueError(f"power stripes ({stripe}) under the {stripe_role.value} width/space rules")
    if abs(stripe.scan_step / rules.grid - round(stripe.scan_step / rules.grid)) > 1e-6:
        raise ValueError("the power stripe scan step is not a multiple of the grid")


@dataclass(frozen=True)
class Sizing:
    """The devices the compiler draws by default, as field values of its configuration objects
    (:class:`~gc_compiler.config.BitcellConfig`, :class:`~gc_compiler.periphery.logic.LogicConfig`,
    :class:`~gc_compiler.periphery.logic.LevelShifterConfig`): what ``X.for_tech(tech)`` builds.
    Every width and length must be at least the rules' device minimums (checked where devices
    are drawn)."""

    bitcell: dict
    """The default cell."""
    plain_bitcell: dict
    """The plain cell (:meth:`~gc_compiler.config.BitcellConfig.plain`), on top of ``bitcell``."""
    logic: dict
    """The periphery's logic cells (inverters, gates, drivers at strength 1)."""
    predecode: dict
    """The predecode gates, which drive a whole bus of rows."""
    level_shifter: dict
    size_grid: float
    """What a scaled device size snaps to."""
    wide_array_cols: int
    """From this many columns word lines are strapped and driven harder by default."""
    wide_driver_strength: float
    write_wl_low: float
    """The negative write rail's level, volts (``ArrayConfig.write_wl_low``)."""
