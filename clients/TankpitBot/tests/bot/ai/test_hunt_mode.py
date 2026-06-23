"""Tests for the durable HUNT owner."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
    viewport_covered_tiles,
)


def _enemy_tank(
    *,
    tank_id: int = 50,
    x: int = 120,
    y: int = 100,
    name: str = "Enemy",
) -> TankStateDict:
    """Create a visible enemy tank for HUNT tests.

    The tank is wire-present at the HUNT tests' tick clock (100000):
    ``last_wire_seen_ms`` is set equal to ``timestamp_ms`` so it passes
    the kill-shot wire-presence gate, modelling an enemy genuinely in
    view rather than a map-only afterimage.

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
        last_wire_seen_ms=100000,
        last_position_update_ms=100000,
        last_viewport_observation_ms=100000,
    )


def _pursuit_target(
    *,
    tank_id: int = 50,
    x: int = 120,
    y: int = 100,
    name: str = "Runner",
) -> TankStateDict:
    """Create an off-viewport but wire-fresh locked target.

    Models the case where a locked enemy teleported out of view:
    ``last_viewport_observation_ms`` is stale (so analyze_threats
    filters them out of the firing list) but ``timestamp_ms`` and
    ``last_wire_seen_ms`` are fresh (the global 0x2E broadcast or
    a recent map snapshot still vouches for them). HUNT must
    pursue this target via the world-registry path rather than
    enter CONFIRM_KILL.
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
        last_wire_seen_ms=100000,
        last_position_update_ms=100000,
        last_viewport_observation_ms=80000,
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


def test_hunt_search_dispatches_map_open_not_radar_during_acquire() -> None:
    """HUNT enemy search dispatches map_open, never radar.

    Behavior-contract guard: radar reveals hidden entities (fuel /
    equipment / mines), NOT enemies; enemies arrive via the wire
    stream. With no visible threat, HUNT acquire's only fallback is
    a global map snapshot -- the viewport-edge walk was deleted
    2026-06-22 because (a) viewport shifting is OFF in this game
    configuration, so walking to an edge reveals no new ground, and
    (b) the terrain-blocked teleport variant burned fuel without
    aiming at a known enemy.
    """
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

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason"] == "find_enemies"
    assert decision["behavior"]["mode"] == "HUNT"


def test_hunt_search_does_not_enter_confirm_kill_without_target() -> None:
    """Consecutive search ticks stay in ACQUIRE, never bogus confirm-kill.

    Each tick with no visible threat dispatches another ``map_open``;
    the mode-state derivation keeps the bot in ACQUIRE until a
    threat actually shows up. CONFIRM_KILL must only fire from a
    locked combat target disappearing -- not from enemy-search churn.
    """
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
    if first_decision["command"]["cmd_type"] != "map_open":
        raise AssertionError("expected map_open from enemy search path")

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


def test_hunt_acquire_uses_fresh_target_position_to_close_on_target() -> None:
    """Fresh wire-sourced target position lets HUNT acquire teleport directly."""
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
    """Refresh re-teleports to a visible non-adjacent target to close distance."""
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


def test_hunt_acquire_refuels_when_fresh_position_teleport_is_unaffordable() -> None:
    """Acquire delegates to fuel recovery when combat teleport is unaffordable.

    Run 20260611-025636: the old map-open fallback spun 115 map reopens
    without a single shot. Fuel collection is the only action that changes
    the affordability condition, so the bot refuels before re-acquiring.
    With the current viewport already swept by the bot's tile-coverage
    map the forager yields to the search hop, producing the teleport
    the regression guard expects.
    """
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
            "local_scan_tiles": viewport_covered_tiles(world),
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["mode"] == "COLLECT_FUEL"
    assert decision["updated_ai_state"]["combat_target_id"] == -1


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


def test_hunt_refresh_refuels_when_close_action_is_not_legal() -> None:
    """Refresh delegates to fuel recovery when combat teleport is unaffordable."""
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
            "local_scan_tiles": viewport_covered_tiles(world),
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["mode"] == "COLLECT_FUEL"


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
    """Close state re-teleports to a non-adjacent target to close distance."""
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


def test_hunt_close_refuels_when_close_action_is_not_legal() -> None:
    """Close state delegates to fuel recovery when combat teleport is unaffordable."""
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
            "local_scan_tiles": viewport_covered_tiles(world),
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["mode"] == "COLLECT_FUEL"


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
    """Confirm-kill is transient and then returns to normal acquisition.

    The respawned target carries a fresh wire-sourced position
    (``_enemy_tank`` defaults), so the reacquisition produces a direct
    teleport rather than a map_open refresh.
    """
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

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason"] == "teleport Respawned"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_scan_on_landing_reacquires_when_target_gone() -> None:
    """SCAN_ON_LANDING falls back to acquire when the locked target despawned."""
    world, self_state = make_world(fuel=800)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "SCAN_ON_LANDING",
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


def test_scan_on_landing_engages_when_target_present() -> None:
    """SCAN_ON_LANDING transitions to ENGAGE when target is still visible."""
    tanks: dict[str, TankStateDict] = {"50": _enemy_tank(x=101, y=100)}
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "SCAN_ON_LANDING",
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


def test_scan_on_landing_closes_when_target_not_adjacent() -> None:
    """SCAN_ON_LANDING re-closes when the target is visible but not adjacent."""
    tanks: dict[str, TankStateDict] = {"50": _enemy_tank(x=105, y=100)}
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "SCAN_ON_LANDING",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 105,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"


def test_hunt_engage_fires_homing_when_locked_target_left_viewport() -> None:
    """Locked target out of viewport but wire-fresh -> fire at last known pos.

    Behavior-contract guard (2026-06-22 user directive): when the
    locked target teleports out of view we DO NOT enter CONFIRM_KILL
    and DO NOT chase. The bot stays put and fires at the world-known
    position; the server picks a homing shot when the target is
    distant or in motion, and homing tracks. The lock holds until a
    deactivation signal arrives.
    """
    tanks: dict[str, TankStateDict] = {"50": _pursuit_target(x=150, y=150)}
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ENGAGE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["behavior"]["reason"] == "shoot Runner"


def test_hunt_close_fires_homing_when_locked_target_left_viewport() -> None:
    """CLOSE state pursues via homing fire when target leaves viewport."""
    tanks: dict[str, TankStateDict] = {"50": _pursuit_target(x=150, y=150)}
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "CLOSE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"


def test_hunt_refresh_fires_homing_when_locked_target_left_viewport() -> None:
    """REFRESH state pursues via homing fire when target leaves viewport."""
    tanks: dict[str, TankStateDict] = {"50": _pursuit_target(x=150, y=150)}
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "REFRESH",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"


def test_scan_on_landing_fires_homing_when_locked_target_left_viewport() -> None:
    """SCAN_ON_LANDING state pursues via homing fire when target leaves viewport."""
    tanks: dict[str, TankStateDict] = {"50": _pursuit_target(x=150, y=150)}
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "SCAN_ON_LANDING",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"
