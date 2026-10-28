"""Is the layout driven by its technology's rules, or by sky130 numbers?

Each reference structure is built with sky130 and with IHP SG13G2 -- a second real process, with a
different stack (no local interconnect: every role one metal up), contact and via sizes, implant
scheme (n+ implied) and spacings -- and checked against *that technology's own* rules
(:func:`gc_compiler.verify.rules_drc.check_rules`). A number that stands for a sky130 rule instead of
reading it passes with sky130 and fails with SG13G2's rules. The check needs only the adapters, not
the PDKs.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from gc_compiler.tech.rules import DesignRules
from gc_compiler.tech.sg13g2 import SG13G2
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.verify.rules_drc import CHECKED, NOT_CHECKED, check_rules
from structures import LEAF, reference_structures

SKY130 = Sky130()
IHP = SG13G2()



def test_every_rule_is_checked_or_waived_with_a_reason():
    names = {f.name for f in fields(DesignRules)}
    assert names == set(CHECKED) | set(NOT_CHECKED), names ^ (set(CHECKED) | set(NOT_CHECKED))
    assert not set(CHECKED) & set(NOT_CHECKED)


@pytest.mark.parametrize("name", list(reference_structures(SKY130, "pt")))
def test_sky130_layout_follows_its_rules(name):
    component = reference_structures(SKY130, "pt_sky")[name]()
    result = check_rules(component, SKY130, leaf=name in LEAF)
    assert result.clean, str(result)


@pytest.mark.parametrize("name", list(reference_structures(IHP, "pt")))
def test_sg13g2_layout_follows_its_rules(name):
    component = reference_structures(IHP, "pt_ihp")[name]()
    result = check_rules(component, IHP, leaf=name in LEAF)
    assert result.clean, str(result)


def test_a_negative_rail_without_an_isolated_well_is_refused():
    from gc_compiler.tech.rules import LayerRole

    class NoIsolation(SG13G2):
        name = "ihp-sg13g2-noiso"
        cell_prefix = "ihpnoiso_"

        def has(self, role):
            return role is not LayerRole.DNWELL and super().has(role)

    with pytest.raises(ValueError, match="deep n-well"):
        reference_structures(NoIsolation(), "pt_noiso")["macro_neg_wl"]()


def test_a_threshold_flavor_the_technology_lacks_is_refused():
    from gc_compiler.cells.bitcell_3t import gain_cell_3t
    from gc_compiler.config import BitcellConfig

    with pytest.raises(ValueError, match="no 'lvt' threshold flavor"):
        gain_cell_3t(IHP, BitcellConfig.for_tech(IHP, write_flavor="lvt"), name="pt_ihp_lvt")


def test_a_technology_without_measurements_refuses_to_guess():
    """A refresh interval measured on sky130 could lose data elsewhere: without measurements the
    compiler raises instead of borrowing them."""
    from gc_compiler.config import ArrayConfig
    from gc_compiler.rtl import ControllerTiming, default_refresh_interval_us
    from gc_compiler.tech.characterised import NotCharacterised

    class Unmeasured(SG13G2):
        name = "ihp-sg13g2-unmeasured"
        characterised = None

    unmeasured = Unmeasured()
    cfg = ArrayConfig.for_tech(unmeasured, rows=8, cols=8)
    with pytest.raises(NotCharacterised):
        default_refresh_interval_us(cfg, unmeasured)
    with pytest.raises(NotCharacterised):
        ControllerTiming.default(unmeasured)
    assert default_refresh_interval_us(ArrayConfig.for_tech(SKY130, rows=8, cols=8), SKY130) == 154.0
    assert default_refresh_interval_us(ArrayConfig.for_tech(IHP, rows=8, cols=8), IHP) == 100.0
    assert default_refresh_interval_us(ArrayConfig.basic(IHP, rows=8, cols=8), IHP) == 26.9
    negative = ArrayConfig.for_tech(IHP, rows=8, cols=8, negative_write_wl=True)
    assert negative.write_wl_low == -0.12  # vdd - vneg at the LV devices' 1.32 V rating
    assert default_refresh_interval_us(negative, IHP) == 120.9
    assert default_refresh_interval_us(ArrayConfig.basic(IHP, rows=8, cols=8, negative_write_wl=True), IHP) == 37.8


def test_devices_below_the_minimum_are_refused():
    """A configuration below the technology's minimum gate length is refused, not drawn illegal."""
    from gc_compiler.cells.bitcell_3t import gain_cell_3t
    from gc_compiler.config import BitcellConfig

    short = IHP.rules.device_min_length - 0.01
    with pytest.raises(ValueError, match="below"):
        gain_cell_3t(IHP, BitcellConfig.for_tech(IHP, read_length=short), name="pt_too_small")
    assert BitcellConfig.for_tech(IHP).read_length >= IHP.rules.device_min_length


def test_the_loop_back_warns_only_where_it_misreads():
    """sky130's default cell needs in-macro refresh at sf; its LVT cell and IHP's default do not."""
    from gc_compiler.config import ArrayConfig, BitcellConfig
    from gc_compiler.rtl import loopback_warning

    def cfg(tech, **fields):
        return ArrayConfig.for_tech(tech, rows=8, cols=8, **fields)

    assert loopback_warning(cfg(SKY130), SKY130) is None
    assert loopback_warning(cfg(SKY130, refresh_in_macro=False), SKY130)
    assert loopback_warning(cfg(SKY130, self_timed_read=False), SKY130)
    lvt = BitcellConfig.for_tech(SKY130, write_flavor="lvt", write_length=0.35, mr_length=0.5)
    assert loopback_warning(cfg(SKY130, bitcell=lvt, refresh_in_macro=False), SKY130) is None
    assert loopback_warning(cfg(IHP, refresh_in_macro=False), IHP) is None


def test_banks_taller_than_a_measured_refresh_are_refused():
    """sky130's refresh was measured to restore a '1' up to 32 rows a bank, SG13G2's up to 16:
    taller warns, and the hard block, which refreshes it, refuses."""
    from gc_compiler.compiler import GainCellCompiler
    from gc_compiler.config import ArrayConfig
    from gc_compiler.rtl import rows_warning

    assert rows_warning(ArrayConfig.for_tech(SKY130, rows=32, cols=64), SKY130) is None
    tall = ArrayConfig.for_tech(SKY130, rows=64, cols=64)
    assert "64 rows" in rows_warning(tall, SKY130)
    assert rows_warning(ArrayConfig.for_tech(IHP, rows=16, cols=64), IHP) is None
    assert "32 rows" in rows_warning(ArrayConfig.for_tech(IHP, rows=32, cols=32), IHP)
    with pytest.raises(ValueError, match="64 rows"):
        GainCellCompiler(tall, SKY130).memory()
