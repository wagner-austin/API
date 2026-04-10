"""Tests for the durable HUNT owner."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


def _enemy_tank(
    *,
    tank_id: int = 50,
    x: int = 120,
    y: int = 100,
    name: str = "Enemy",
) -> TankStateDict:
    """Create a visible enemy tank for HUNT tests.

    Args:
        tank_id: Enemy tank id.
        x: Enemy x coordinate.
        y: Enemy y coordinate.
        name: Enemy display name.

    Returns:
        Visible enemy tank state.
    """
    return make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=2,
        rank=1,
        name=name,
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=100000,
    )


def test_hunt_acquire_searches_for_enemies_when_no_target_exists() -> None:
    """HUNT acquire falls back to enemy search when no threats are visible."""
    world, self_state = make_world(fuel=800)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason"] == "find_enemies"


def test_hunt_search_uses_regular_radar_when_extra_charges_are_empty() -> None:
    """Enemy search still uses the built-in radar without extra-radar stock."""
    world, self_state = make_world(fuel=800, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "radar"
    assert decision["behavior"]["reason"] == "radar_for_enemies"


def test_hunt_search_teleport_does_not_enter_confirm_kill_without_target() -> None:
    """Search teleports stay in acquisition instead of bogus confirm-kill."""
    world, self_state = make_world(fuel=800)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
            "last_scan_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    first_decision = decide_hunt_mode(ctx)
    if first_decision["command"]["cmd_type"] != "move":
        raise AssertionError("expected edge search move from enemy search path")

    next_ai_state = AIStateDict(
        **{
            **first_decision["updated_ai_state"],
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
        }
    )
    next_ctx = DecideCtx(world, self_state, next_ai_state, inventory, 102000, None, "")

    second_decision = decide_hunt_mode(next_ctx)

    assert second_decision["behavior"]["reason"] != "confirm_kill"


def test_hunt_acquire_uses_recent_map_snapshot_to_close_on_target() -> None:
    """Recent map intel lets HUNT acquire teleport directly toward a target."""
    tanks: dict[str, TankStateDict] = {
        "50": _enemy_tank(),
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason"] == "teleport Enemy"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_hunt_acquire_targets_enemy_between_break_and_resume_thresholds() -> None:
    """HUNT still acquires targets in the non-emergency reserve band."""
    tanks: dict[str, TankStateDict] = {
        "50": _enemy_tank(),
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    inventory = make_inventory(default_count=30, dual_count=18)
    inventory["homing_shots"]["count"] = 23
    inventory["extra_radars"]["count"] = 19
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason"] == "teleport Enemy"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_hunt_refresh_engages_visible_adjacent_locked_target() -> None:
    """Refresh state immediately engages a visible adjacent locked target."""
    tanks: dict[str, TankStateDict] = {
        "50": _enemy_tank(x=101, y=100),
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "REFRESH",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 101,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["behavior"]["reason"] == "shoot Enemy"
    assert decision["updated_ai_state"]["last_shot_target_id"] == 50


def test_hunt_refresh_returns_close_decision_for_visible_nonadjacent_target() -> None:
    """Refresh keeps progressing toward a visible target when closing is legal."""
    tanks: dict[str, TankStateDict] = {
        "50": _enemy_tank(),
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "REFRESH",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 120,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason"] == "teleport Enemy"


def test_hunt_acquire_opens_map_when_recent_snapshot_teleport_is_unaffordable() -> None:
    """Acquire falls back to map refresh when close teleport cannot be afforded."""
    tanks: dict[str, TankStateDict] = {
        "50": _enemy_tank(x=190, y=100, name="FarEnemy"),
    }
    world, self_state = make_world(fuel=520, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason"] == "find FarEnemy"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_hunt_refresh_reacquires_when_locked_target_is_missing() -> None:
    """Refresh falls back to acquisition when the locked target is gone."""
    world, self_state = make_world(fuel=800)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "REFRESH",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 120,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason"] == "find_enemies"


def test_hunt_refresh_opens_map_when_close_action_is_not_legal() -> None:
    """Refresh reopens the map when the target cannot be legally closed on."""
    tanks: dict[str, TankStateDict] = {
        "50": _enemy_tank(x=190, y=100, name="FarEnemy"),
    }
    world, self_state = make_world(fuel=520, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "REFRESH",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 190,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason"] == "find FarEnemy"


def test_hunt_close_enters_confirm_kill_when_locked_target_disappears() -> None:
    """Close state explicitly transitions through confirm-kill when target vanishes."""
    world, self_state = make_world(fuel=800)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "CLOSE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 120,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason"] == "confirm_kill"


def test_hunt_close_returns_close_decision_for_visible_target() -> None:
    """Close state continues issuing legal close actions while target is present."""
    tanks: dict[str, TankStateDict] = {
        "50": _enemy_tank(),
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "CLOSE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 120,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason"] == "teleport Enemy"


def test_hunt_close_opens_map_when_close_action_is_not_legal() -> None:
    """Close state reopens the map when no close action can legally run."""
    tanks: dict[str, TankStateDict] = {
        "50": _enemy_tank(x=190, y=100, name="FarEnemy"),
    }
    world, self_state = make_world(fuel=520, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "CLOSE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 190,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason"] == "find FarEnemy"


def test_hunt_engage_enters_confirm_kill_when_locked_target_disappears() -> None:
    """Engage state explicitly transitions through confirm-kill when target vanishes."""
    world, self_state = make_world(fuel=800)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ENGAGE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 120,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason"] == "confirm_kill"
    assert decision["updated_ai_state"]["combat_target_id"] == -1


def test_hunt_engage_shoots_visible_locked_target() -> None:
    """Engage state keeps shooting a visible locked target."""
    tanks: dict[str, TankStateDict] = {
        "50": _enemy_tank(x=101, y=100),
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ENGAGE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 101,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["behavior"]["reason"] == "shoot Enemy"


def test_hunt_engage_confirms_killed_target_with_explicit_reason() -> None:
    """Engage confirmation also handles targets already on the kill cooldown."""
    world, self_state = make_world(fuel=800)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ENGAGE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 120,
            "combat_target_y": 100,
            "killed_tank_ids": {"50": 99999},
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason"] == "confirm_kill"
    assert decision["updated_ai_state"]["combat_target_id"] == -1


def test_hunt_engage_without_locked_target_id_still_confirms_and_searches() -> None:
    """Engage handles a missing target id without crashing or reusing stale state."""
    world, self_state = make_world(fuel=800)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ENGAGE",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason"] == "confirm_kill"
    assert decision["updated_ai_state"]["combat_target_id"] == -1


def test_hunt_confirm_kill_reacquires_after_target_state_clears() -> None:
    """Confirm-kill is transient and then returns to normal acquisition."""
    tanks: dict[str, TankStateDict] = {
        "50": _enemy_tank(name="Respawned"),
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "CONFIRM_KILL",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason"] == "find Respawned"
    assert decision["updated_ai_state"]["combat_target_id"] == 50
    assert decision["updated_ai_state"]["combat_target_id"] == 50
