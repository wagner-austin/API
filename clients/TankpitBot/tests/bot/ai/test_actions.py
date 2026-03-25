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

    def test_radar_when_scan_ready(self) -> None:
        """Issues radar command when scan cooldown has elapsed."""
        ai_state = make_initial_ai_state()
        behavior = make_behavior_score("HUNT", 800, 110, 100, "test")
        self_state = _self()

        new_state, cmd = execute_behavior(behavior, ai_state, self_state, 6000)
        assert cmd["cmd_type"] == "radar"
        assert new_state["last_scan_ms"] == 6000

    def test_shoot_when_scan_not_ready(self) -> None:
        """Issues shoot command when scan is on cooldown but shoot is ready."""
        ai_state = make_initial_ai_state()
        ai_state["last_scan_ms"] = 5000  # Recent scan
        behavior = make_behavior_score("HUNT", 800, 110, 100, "test")
        self_state = _self()

        new_state, cmd = execute_behavior(behavior, ai_state, self_state, 6000)
        assert cmd["cmd_type"] == "shoot"
        assert cmd["target_x"] == 110
        assert cmd["target_y"] == 100
        assert new_state["last_shoot_ms"] == 6000

    def test_move_when_both_on_cooldown(self) -> None:
        """Issues move command when both scan and shoot are on cooldown."""
        ai_state = make_initial_ai_state()
        ai_state["last_scan_ms"] = 5500
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
        ai_state["active_mode"] = "PATROL"
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


# =============================================================================
# DEPOSIT_FUEL actions
# =============================================================================


class TestExecuteDepositFuel:
    """Tests for DEPOSIT_FUEL behavior execution."""

    def test_issues_pickup_move(self) -> None:
        """Issues pickup_move command toward deposit target."""
        ai_state = make_initial_ai_state()
        behavior = make_behavior_score("DEPOSIT_FUEL", 700, 130, 120, "test")
        self_state = _self()

        new_state, cmd = execute_behavior(behavior, ai_state, self_state, 1000)
        assert cmd["cmd_type"] == "pickup_move"
        assert cmd["target_x"] == 130
        assert new_state["active_mode"] == "DEPOSIT_FUEL"


# =============================================================================
# PATROL actions
# =============================================================================


class TestExecutePatrol:
    """Tests for PATROL behavior execution."""

    def test_issues_move_command(self) -> None:
        """Issues move command toward current waypoint."""
        ai_state = make_initial_ai_state()
        behavior = make_behavior_score("PATROL", 100, 64, 64, "test")
        self_state = _self()

        _, cmd = execute_behavior(behavior, ai_state, self_state, 1000)
        assert cmd["cmd_type"] == "move"
        # Target (64,64) from self (100,100): clamped to 8-step → (92,92)
        assert cmd["target_x"] == 92
        assert cmd["target_y"] == 92

    def test_advances_waypoint_when_close(self) -> None:
        """Waypoint index advances when within 3 tiles of target."""
        ai_state = make_initial_ai_state()
        ai_state["active_mode"] = "PATROL"
        behavior = make_behavior_score("PATROL", 100, 64, 64, "test")
        self_state = _self(x=63, y=64)  # 1 tile away

        new_state, _ = execute_behavior(behavior, ai_state, self_state, 1000)
        assert new_state["patrol_waypoint_index"] == 1

    def test_does_not_advance_when_far(self) -> None:
        """Waypoint index stays when far from target."""
        ai_state = make_initial_ai_state()
        ai_state["active_mode"] = "PATROL"
        behavior = make_behavior_score("PATROL", 100, 64, 64, "test")
        self_state = _self(x=100, y=100)  # Far from waypoint

        new_state, _ = execute_behavior(behavior, ai_state, self_state, 1000)
        assert new_state["patrol_waypoint_index"] == 0

    def test_wraps_waypoint_index(self) -> None:
        """Waypoint index wraps to 0 after reaching last waypoint."""
        ai_state = make_initial_ai_state()
        ai_state["active_mode"] = "PATROL"
        ai_state["patrol_waypoint_index"] = 3  # Last waypoint
        wp = ai_state["config"]["patrol_waypoints"][3]
        behavior = make_behavior_score("PATROL", 100, wp[0], wp[1], "test")
        self_state = _self(x=wp[0], y=wp[1])  # At waypoint

        new_state, _ = execute_behavior(behavior, ai_state, self_state, 1000)
        assert new_state["patrol_waypoint_index"] == 0


# =============================================================================
# DEFEND actions
# =============================================================================


class TestExecuteDefend:
    """Tests for DEFEND behavior execution."""

    def test_shoot_when_ready(self) -> None:
        """Issues shoot command when cooldown allows."""
        ai_state = make_initial_ai_state()
        behavior = make_behavior_score("DEFEND", 850, 105, 100, "test")
        self_state = _self()

        new_state, cmd = execute_behavior(behavior, ai_state, self_state, 5000)
        assert cmd["cmd_type"] == "shoot"
        assert cmd["target_x"] == 105
        assert new_state["last_shoot_ms"] == 5000

    def test_retreat_when_shoot_on_cooldown(self) -> None:
        """Issues move away from threat when shoot is on cooldown."""
        ai_state = make_initial_ai_state()
        ai_state["last_shoot_ms"] = 4500
        behavior = make_behavior_score("DEFEND", 850, 105, 100, "test")
        self_state = _self(x=100, y=100)

        _, cmd = execute_behavior(behavior, ai_state, self_state, 5000)
        assert cmd["cmd_type"] == "move"
        # Retreat: move away from (105, 100) -> (95, 100)
        assert cmd["target_x"] == 95
        assert cmd["target_y"] == 100

    def test_retreat_clamps_to_bounds(self) -> None:
        """Retreat coordinates are clamped to map bounds."""
        ai_state = make_initial_ai_state()
        ai_state["last_shoot_ms"] = 4500
        behavior = make_behavior_score("DEFEND", 850, 5, 5, "test")
        self_state = _self(x=2, y=2)

        _, cmd = execute_behavior(behavior, ai_state, self_state, 5000)
        assert cmd["cmd_type"] == "move"
        assert cmd["target_x"] == 0  # Clamped at 0, not -1
        assert cmd["target_y"] == 0
