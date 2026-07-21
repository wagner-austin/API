"""Tests for :mod:`tankpit_bot.physics.damage`.

Reference values from ``wiki/pages/game-economy.md`` (Damage taken),
wire-verified 2026-06-20 in the multi-tank PvP capture.
"""

from __future__ import annotations

from tankpit_bot.physics.damage import (
    DUAL_HIT_VICTIM_COST,
    MINE_DETONATION_COST,
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
