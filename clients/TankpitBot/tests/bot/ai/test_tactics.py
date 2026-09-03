"""Tests for AI tactical decision functions."""

from __future__ import annotations

from tankpit_bot.bot.ai.tactics import (
    compute_desired_equipment,
)
from tankpit_bot.state.types import (
    SelfStateDict,
    WorldStateDict,
    make_self_state,
    make_viewport_state,
)


def _empty_world() -> WorldStateDict:
    """Create an empty world state."""
    return WorldStateDict(
        self_state=None,
        tanks={},
        containers={},
        mines={},
        terrain={},
        viewport=make_viewport_state(left=0, top=0, width=18, height=18),
        scanned_tiles={},
        timestamp_ms=0,
    )


def _self(x: int = 100, y: int = 100) -> SelfStateDict:
    """Create a self state at given position."""
    return make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=0,
        rank=4,
        fuel=800,
        leaderboard_position=1,
    )


# =============================================================================
# compute_desired_equipment
# =============================================================================


class TestComputeDesiredEquipment:
    """Tests for equipment desired-set computation."""

    def test_patrol_dual_and_radar(self) -> None:
        """PATROL mode has dual, homing, and radar always on."""
        result = compute_desired_equipment("PATROL", 800)
        assert result == {2, 4, 5}

    def test_hunt_dual_and_radar(self) -> None:
        """HUNT mode has dual, homing, and radar."""
        result = compute_desired_equipment("HUNT", 800)
        assert result == {2, 4, 5}

    def test_defend_dual_and_radar(self) -> None:
        """DEFEND mode has dual, homing, and radar (no shields)."""
        result = compute_desired_equipment("DEFEND", 800)
        assert result == {2, 4, 5}

    def test_collect_fuel_dual_and_radar(self) -> None:
        """COLLECT has dual, homing, and radar regardless of fuel level."""
        result = compute_desired_equipment("COLLECT", 100)
        assert result == {2, 4, 5}

    def test_dual_shots_depleted_drops_dual(self) -> None:
        """When dual shots count=0, slot 2 is not included."""
        result = compute_desired_equipment("HUNT", 800, dual_shots_count=0)
        assert result == {4, 5}

    def test_homing_shots_depleted_drops_homing(self) -> None:
        """When homing shots count=0, slot 4 is not included."""
        result = compute_desired_equipment("HUNT", 800, homing_shots_count=0)
        assert result == {2, 5}

    def test_no_shields_ever(self) -> None:
        """Shields are never included in desired equipment."""
        result = compute_desired_equipment("HUNT", 800)
        assert 1 not in result
