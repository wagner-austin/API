"""Tests for :mod:`tankpit_bot.physics.costs`.

Reference values from ``wiki/pages/game-economy.md`` (action-cost
table, closed 2026-07-20). Every constant is pinned to its wire-
measured value; the teleport formula is probed at exact hand-computed
points including both isqrt rounding directions.
"""

from __future__ import annotations

from tankpit_bot.physics.costs import (
    BLOCK_OP_COST,
    DUAL_SHOT_COST,
    HOMING_SHOT_COST,
    MINE_PRESS_COST,
    MISSILE_SHOT_COST,
    RADAR_COST,
    SINGLE_SHOT_COST,
    WALK_COST_PER_TILE,
    teleport_cost,
)


class TestActionCosts:
    """The closed action economy: everything is 10 except walk and single."""

    def test_walk_costs_1_per_tile(self) -> None:
        """Walking costs 1 fuel per tile (0x47 Manhattan vs fuel delta)."""
        assert WALK_COST_PER_TILE == 1

    def test_single_shot_costs_6(self) -> None:
        """Single shot costs 6 (62 clean isolation windows, 2026-07-20)."""
        assert SINGLE_SHOT_COST == 6

    def test_dual_shot_costs_10(self) -> None:
        """Dual shot costs 10 (589 clean isolation windows)."""
        assert DUAL_SHOT_COST == 10

    def test_missile_costs_10(self) -> None:
        """Missile costs 10 (sniff-20260720-213208: 6 clean windows)."""
        assert MISSILE_SHOT_COST == 10

    def test_homing_costs_10(self) -> None:
        """Homing costs 10 total (398 windows at -10 plus split -5 pairs)."""
        assert HOMING_SHOT_COST == 10

    def test_radar_costs_10(self) -> None:
        """Extra-radar scan costs 10."""
        assert RADAR_COST == 10

    def test_mine_press_costs_10_flat(self) -> None:
        """A mine press costs 10 regardless of how many mines land
        (sniff-20260720-214329: 8 presses, each exactly -10)."""
        assert MINE_PRESS_COST == 10

    def test_block_ops_are_free(self) -> None:
        """Block pickup/drop costs nothing (stationary re-place pairs
        at zero fuel delta, 2026-07-20)."""
        assert BLOCK_OP_COST == 0


class TestTeleportCost:
    """``floor(6 * sqrt(dx^2 + dy^2))`` — 248/248 exact post-fix pairs."""

    def test_zero_distance_is_free(self) -> None:
        """Teleporting to the current tile costs nothing."""
        assert teleport_cost(10, 20, 10, 20) == 0

    def test_one_tile_axis_hop_costs_6(self) -> None:
        """A 1-tile axis hop costs exactly 6."""
        assert teleport_cost(0, 0, 1, 0) == 6

    def test_pythagorean_hop_is_exact(self) -> None:
        """A 3-4-5 hop costs exactly 30 (6 * 5, no rounding)."""
        assert teleport_cost(0, 0, 3, 4) == 30

    def test_diagonal_floor_rounds_down(self) -> None:
        """A 1,1 hop costs floor(6 * sqrt(2)) = 8, not 9."""
        assert teleport_cost(0, 0, 1, 1) == 8

    def test_negative_deltas_price_like_positive(self) -> None:
        """Direction is irrelevant: a -3,-4 hop is also 30."""
        assert teleport_cost(5, 5, 2, 1) == 30

    def test_long_hop_has_no_float_drift(self) -> None:
        """A 100,100 hop is isqrt(36 * 20000) = 848 exactly."""
        assert teleport_cost(0, 0, 100, 100) == 848
