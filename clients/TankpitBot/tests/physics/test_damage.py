"""Tests for :mod:`tankpit_bot.physics.damage`.

Reference values from ``wiki/pages/game-economy.md`` (Damage taken),
wire-verified 2026-06-20 in the multi-tank PvP capture.
"""

from __future__ import annotations

from tankpit_bot.physics.damage import (
    ARMOR_ABSORB_PER_SHIELD,
    DUAL_HIT_VICTIM_COST,
    HOMING_HIT_VICTIM_COST,
    MINE_DETONATION_COST,
    MISSILE_HIT_VICTIM_COST,
    SINGLE_HIT_VICTIM_COST,
)


class TestVictimCosts:
    """Damage is a fuel decrement; these are the per-source amounts."""

    def test_single_hit_costs_victim_45(self) -> None:
        """An enemy single hit drains 45 (3 Yuppler hits, each -45)."""
        assert SINGLE_HIT_VICTIM_COST == 45

    def test_dual_hit_costs_victim_90(self) -> None:
        """An enemy dual hit drains 90 (3 Yuppler dual hits, each -90)."""
        assert DUAL_HIT_VICTIM_COST == 90

    def test_mine_detonation_costs_45(self) -> None:
        """Walking onto a hostile mine drains 45 (t+373.35s of the
        multi-pickup capture) — the fact behind mine-tile impassability
        in the composed decision terrain."""
        assert MINE_DETONATION_COST == 45


class TestVictimCostSession:
    """The 2026-07-21 victim-cost session: missile/homing/armor."""

    def test_missile_hit_costs_victim_45(self) -> None:
        """Five isolated missile hits each drained exactly 45."""
        assert MISSILE_HIT_VICTIM_COST == 45

    def test_homing_hit_costs_victim_45(self) -> None:
        """Five isolated homing hits each drained exactly 45."""
        assert HOMING_HIT_VICTIM_COST == 45

    def test_one_shield_absorbs_45_damage(self) -> None:
        """Shields consume at damage/45: duals eat 2, everything else 1."""
        assert ARMOR_ABSORB_PER_SHIELD == 45
        assert DUAL_HIT_VICTIM_COST // ARMOR_ABSORB_PER_SHIELD == 2
