"""Tests for AI behavior action execution."""

from __future__ import annotations

from tankpit_bot.bot.ai.actions import execute_behavior
from tankpit_bot.bot.ai.types import make_behavior_score, make_initial_ai_state
from tankpit_bot.state.types import SelfStateDict, make_self_state


def _self(x: int = 100, y: int = 100, fuel: int = 800) -> SelfStateDict:
    """Create self state."""
    return make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=0,
        rank=4,
        fuel=fuel,
        leaderboard_position=1,
    )


# =============================================================================
# HUNT actions
# =============================================================================


class TestExecuteHunt:
    """Tests for HUNT behavior execution."""

    def test_shoot_when_cooldown_elapsed(self) -> None:
        """Issues shoot command when shoot cooldown has elapsed."""
        ai_state = make_initial_ai_state()
        # Target within viewport range (dist=5 <= _MAX_SHOOT_RANGE=8)
        behavior = make_behavior_score("HUNT", 800, 105, 100, "test")
        self_state = _self()

        new_state, cmd = execute_behavior(behavior, ai_state, self_state, 6000)
        assert cmd["cmd_type"] == "shoot"
        assert cmd["target_x"] == 105
        assert cmd["target_y"] == 100
        assert new_state["last_shoot_ms"] == 6000

    def test_move_when_target_beyond_viewport(self) -> None:
        """Falls back to move when target is beyond viewport range."""
        ai_state = make_initial_ai_state()
        # Target at (115, 100) from self (100, 100) = dist 15 > _MAX_SHOOT_RANGE=8
        behavior = make_behavior_score("HUNT", 800, 115, 100, "test")
        self_state = _self()

        _, cmd = execute_behavior(behavior, ai_state, self_state, 6000)
        assert cmd["cmd_type"] == "move"
        # Clamped step toward (115, 100): dx=15 clamped to 8 → 108
        assert cmd["target_x"] == 108

    def test_move_blocked_by_impassable_terrain(self) -> None:
        """Move stays in place when terrain at destination is impassable."""
        from tests.fakes import FakeTerrainMap

        ai_state = make_initial_ai_state()
        # Target at (115, 100) from self (100, 100) → clamped dest is (108, 100)
        behavior = make_behavior_score("HUNT", 800, 115, 100, "test")
        self_state = _self()
        terrain = FakeTerrainMap(terrain_data={(108, 100): "#"})

        _, cmd = execute_behavior(behavior, ai_state, self_state, 6000, terrain)
        assert cmd["cmd_type"] == "move"
        # Terrain blocked → stays at current position
        assert cmd["target_x"] == 100
        assert cmd["target_y"] == 100

    def test_move_when_shoot_on_cooldown(self) -> None:
        """Issues move command when shoot is on cooldown."""
        ai_state = make_initial_ai_state()
        ai_state["last_shoot_ms"] = 5500
        behavior = make_behavior_score("HUNT", 800, 110, 100, "test")
        self_state = _self()

        _, cmd = execute_behavior(behavior, ai_state, self_state, 6000)
        assert cmd["cmd_type"] == "move"
        # Target (110,100) from self (100,100): dx=10 clamped to 8 → 108
        assert cmd["target_x"] == 108

    def test_updates_active_mode(self) -> None:
        """Active mode is updated to HUNT."""
        ai_state = make_initial_ai_state()
        behavior = make_behavior_score("HUNT", 800, 110, 100, "test")
        self_state = _self()

        new_state, _ = execute_behavior(behavior, ai_state, self_state, 6000)
        assert new_state["active_mode"] == "HUNT"

    def test_resets_ticks_on_mode_change(self) -> None:
        """Ticks reset to 0 when mode changes."""
        ai_state = make_initial_ai_state()
        ai_state["active_mode"] = "COLLECT_FUEL"
        ai_state["ticks_in_mode"] = 50
        behavior = make_behavior_score("HUNT", 800, 110, 100, "test")
        self_state = _self()

        new_state, _ = execute_behavior(behavior, ai_state, self_state, 6000)
        assert new_state["ticks_in_mode"] == 0

    def test_increments_ticks_same_mode(self) -> None:
        """Ticks increment when staying in same mode."""
        ai_state = make_initial_ai_state()
        ai_state["active_mode"] = "HUNT"
        ai_state["ticks_in_mode"] = 5
        ai_state["last_scan_ms"] = 5000
        ai_state["last_shoot_ms"] = 5000
        behavior = make_behavior_score("HUNT", 800, 110, 100, "test")
        self_state = _self()

        new_state, _ = execute_behavior(behavior, ai_state, self_state, 6000)
        assert new_state["ticks_in_mode"] == 6


# =============================================================================
# COLLECT_FUEL actions
# =============================================================================


class TestExecuteCollectFuel:
    """Tests for COLLECT_FUEL behavior execution."""

    def test_issues_pickup_move(self) -> None:
        """Issues pickup_move command toward fuel container."""
        ai_state = make_initial_ai_state()
        behavior = make_behavior_score("COLLECT_FUEL", 950, 120, 110, "test")
        self_state = _self()

        new_state, cmd = execute_behavior(behavior, ai_state, self_state, 1000)
        assert cmd["cmd_type"] == "pickup_move"
        assert cmd["target_x"] == 120
        assert cmd["target_y"] == 110
        assert new_state["active_mode"] == "COLLECT_FUEL"


# =============================================================================
# COLLECT_EQUIPMENT actions
# =============================================================================


class TestExecuteCollectEquipment:
    """Tests for COLLECT_EQUIPMENT behavior execution."""

    def test_issues_pickup_move(self) -> None:
        """Issues pickup_move command toward equipment container."""
        ai_state = make_initial_ai_state()
        behavior = make_behavior_score("COLLECT_EQUIPMENT", 500, 115, 105, "test")
        self_state = _self()

        new_state, cmd = execute_behavior(behavior, ai_state, self_state, 1000)
        assert cmd["cmd_type"] == "pickup_move"
        assert cmd["target_x"] == 115
        assert new_state["active_mode"] == "COLLECT_EQUIPMENT"
