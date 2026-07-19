"""Tests for the durable HUNT owner."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
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
    assert decision["behavior"]["reason_kind"] == "find_enemies"


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
            "last_map_open_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason_kind"] == "find_enemies"
    assert decision["behavior"]["mode"] == "HUNT"


def test_hunt_acquire_exits_when_fresh_map_has_no_viable_targets() -> None:
    """A fresh map snapshot with no viable enemy ends the session.

    User contract (2026-07-02): when the whole-map view is current and
    no enemy passes the acquisition gates (alive, unblocked, reachable,
    affordable), the bot must not loop on map refreshes -- it exits
    with ``no_viable_targets`` so the run is analyzable.
    """
    world, self_state = make_world(fuel=800)
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

    with pytest.raises(SessionExitError, match="no_viable_targets"):
        decide_hunt_mode(ctx)


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
            "last_map_open_ms": 90000,
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
    next_ctx = DecideCtx(world, self_state, next_ai_state, inventory, 106000, None, "")

    second_decision = decide_hunt_mode(next_ctx)

    assert second_decision["behavior"]["reason_kind"] != "confirm_kill"


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
    assert decision["behavior"]["reason_kind"] == "teleport_target"
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
    assert decision["behavior"]["reason_kind"] == "teleport_target"
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
    assert decision["behavior"]["reason_kind"] == "shoot_target"
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
    assert decision["behavior"]["reason_kind"] == "teleport_target"


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
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ((116, 100),))

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["mode"] == "COLLECT"
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
    assert decision["behavior"]["reason_kind"] == "find_enemies"


def test_hunt_acquire_resumes_visible_locked_target() -> None:
    """ACQUIRE shoots a still-visible locked target instead of re-acquiring fresh.

    Recovery cycles (fuel + equipment) preserve ``combat_target_id``
    so HUNT can resume the same engagement after restocking. When
    ACQUIRE runs and the held lock is still in the threat list, the
    bot engages directly -- no map_open, no fresh
    ``select_new_combat_target``.
    """
    tanks: dict[str, TankStateDict] = {
        "50": _enemy_tank(x=101, y=100),
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
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
    assert decision["behavior"]["reason_kind"] == "shoot_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_hunt_acquire_resumes_visible_locked_target_with_close_when_not_adjacent() -> None:
    """ACQUIRE closes distance on a visible-but-distant locked target.

    If the lock is in the threat list but not in a cardinal-fire
    position, ACQUIRE returns a close (teleport) decision -- same as
    the REFRESH state's close branch.
    """
    tanks: dict[str, TankStateDict] = {
        "50": _enemy_tank(),  # default position (115, 100), distance 15 from (100,100)
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 115,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_hunt_acquire_releases_stale_lock_and_teleports_back_when_affordable() -> None:
    """ACQUIRE releases an off-viewport lock and re-acquires by teleport.

    User contract (2026-07-02): a lock that reaches ACQUIRE with its
    target off-viewport is a stale engagement resumed after a mode
    interrupt. The bot never fires from stand-off range on resume --
    it releases the lock and re-acquires fresh. When the same enemy is
    still the nearest affordable candidate, acquisition teleports back
    to it (the recorded human behavior: purple-1 was resumed by
    map-teleporting onto it, session 2026-07-01).
    """
    tanks: dict[str, TankStateDict] = {"50": _pursuit_target(x=115, y=115)}
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 115,
            "combat_target_y": 115,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_hunt_acquire_releases_stale_lock_when_target_unaffordable() -> None:
    """ACQUIRE drops an off-viewport lock whose re-engagement is unaffordable.

    The stale lock is released and the unaffordable enemy is rejected
    by acquisition (teleport cost + kill budget + reserve exceeds
    fuel), so the bot falls through to a map refresh with no lock --
    instead of firing at a target it cannot legally hit (live run
    2026-07-01 20:48: eleven server-rejected shots at a target 92
    tiles away).
    """
    tanks: dict[str, TankStateDict] = {"50": _pursuit_target(x=150, y=150)}
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["updated_ai_state"]["combat_target_id"] == -1


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
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ((116, 100),))

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["mode"] == "COLLECT"


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
    assert decision["behavior"]["reason_kind"] == "confirm_kill"


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
    assert decision["behavior"]["reason_kind"] == "teleport_target"


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
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ((116, 100),))

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["mode"] == "COLLECT"


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
    assert decision["behavior"]["reason_kind"] == "confirm_kill"
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
    assert decision["behavior"]["reason_kind"] == "shoot_target"


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
    assert decision["behavior"]["reason_kind"] == "confirm_kill"
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
    assert decision["behavior"]["reason_kind"] == "confirm_kill"
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
    assert decision["behavior"]["reason_kind"] == "teleport_target"
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
    assert decision["behavior"]["reason_kind"] == "find_enemies"


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
            "last_shot_target_id": 50,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["behavior"]["reason_kind"] == "shoot_target"


def test_hunt_close_fires_homing_when_locked_target_left_viewport() -> None:
    """CLOSE state pursues via homing fire when target leaves viewport after engagement."""
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
            "last_shot_target_id": 50,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"


def test_hunt_refresh_fires_homing_when_locked_target_left_viewport() -> None:
    """REFRESH state pursues via homing fire when target leaves viewport after engagement."""
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
            "last_shot_target_id": 50,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"


def test_hunt_close_re_teleports_when_lock_was_never_engaged() -> None:
    """CLOSE state re-teleports when lock was set but no shot ever fired at it.

    Regression guard for live run 2026-06-23 21:36:31: bot was at
    (46,100), planner emitted teleport to red-4 at (56,177); the
    executor swapped the teleport for a pre-teleport ``map_open``
    (precondition not met). The map_open took 6116ms and the next
    decision ran in HUNT/CLOSE -- which previously fell straight to
    pursuit and fired ``shoot`` at (56,177) from dist=87, looping 19
    times. ``last_shot_target_id != combat_target_id`` is the
    discriminator: we set the lock but never fired, so the lock is
    the pre-engagement intent, not a mid-fight chase. Re-issue the
    teleport instead of firing into the void.
    """
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

    assert decision["command"]["cmd_type"] == "teleport"


def test_hunt_engage_re_teleports_when_lock_was_never_engaged() -> None:
    """ENGAGE substate re-teleports when lock was set but no shot ever fired."""
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

    assert decision["command"]["cmd_type"] == "teleport"


def test_hunt_refresh_re_teleports_when_lock_was_never_engaged() -> None:
    """REFRESH substate re-teleports when lock was set but no shot ever fired."""
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

    assert decision["command"]["cmd_type"] == "teleport"


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


def _map_known_enemy(
    *,
    tank_id: int = 60,
    x: int = 240,
    y: int = 100,
    name: str = "FarEnemy",
    timestamp_ms: int = 99800,
) -> TankStateDict:
    """Create a map-known enemy with no viewport confirmation.

    The tank carries a fresh map ``timestamp_ms`` (within the map-open
    cooldown) but no viewport observation, so it is invisible to
    ``analyze_threats`` and reachable only through the acquisition /
    relay paths.

    Args:
        tank_id: Enemy tank id.
        x: Enemy x coordinate.
        y: Enemy y coordinate.
        name: Enemy display name.
        timestamp_ms: Map snapshot observation timestamp.

    Returns:
        Map-known enemy tank state.
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
        timestamp_ms=timestamp_ms,
    )


def test_hunt_acquire_relays_via_dot_toward_unaffordable_enemy() -> None:
    """An unaffordable enemy triggers a dot-relay hop instead of an exit.

    User contract (2026-07-03): yellow-dot teleport while en route to
    the opponent. The enemy at 140 tiles costs 840 fuel to reach --
    unaffordable end-to-end at fuel 700 -- so the bot hops to the dot
    that makes affordable progress. The dot behind the bot (no
    progress) and the near-enemy dot that would dip below the
    fuel-low reserve are both skipped.
    """
    tanks: dict[str, TankStateDict] = {
        "60": _map_known_enemy(),
        # Stale map entry: rejected for a non-affordability reason, so
        # the relay must not travel toward it.
        "70": _map_known_enemy(tank_id=70, x=110, y=100, name="Ghost", timestamp_ms=10),
    }
    world, self_state = make_world(fuel=700, tanks=tanks)
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
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        inventory,
        100000,
        None,
        "",
        # (50,100) is behind the bot (no progress); (230,100) makes the
        # most progress but costs 780 + 200 reserve > 700 fuel;
        # (150,100) is the affordable progress dot.
        ((50, 100), (230, 100), (150, 100)),
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 150
    assert decision["command"]["target_y"] == 100
    assert decision["behavior"]["reason_kind"] == "dot_relay"
    assert decision["behavior"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["combat_target_id"] == -1


def test_hunt_relay_prefers_dot_nearest_the_enemy() -> None:
    """Among affordable progress dots, the one closest to the enemy wins.

    The nearer-to-enemy dot is listed first so the second qualifying
    dot exercises the not-better-than-incumbent branch. An allied tank
    in the registry exercises the relay's non-enemy filter.
    """
    tanks: dict[str, TankStateDict] = {
        "60": _map_known_enemy(),
        "80": make_tank_state(
            tank_id=80,
            x=105,
            y=100,
            team=1,
            rank=1,
            name="Ally",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=99800,
        ),
    }
    world, self_state = make_world(fuel=700, tanks=tanks)
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
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        inventory,
        100000,
        None,
        "",
        ((180, 100), (130, 100)),
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 180
    assert decision["command"]["target_y"] == 100


def test_hunt_relay_tie_breaks_on_cheaper_hop() -> None:
    """Dots equidistant from the enemy keep the cheaper teleport."""
    tanks: dict[str, TankStateDict] = {"60": _map_known_enemy()}
    world, self_state = make_world(fuel=1100, tanks=tanks)
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
    # Both dots sit 20 tiles from the enemy at (240,100); the second is
    # the cheaper hop from (100,100) and must replace the first.
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        inventory,
        100000,
        None,
        "",
        ((240, 120), (220, 100)),
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 220
    assert decision["command"]["target_y"] == 100


def test_hunt_relay_exits_when_only_dot_is_impassable() -> None:
    """A relay with no passable progress dot still exits the session."""
    from tests.in_memory_terrain_map import InMemoryTerrainMap

    tanks: dict[str, TankStateDict] = {"60": _map_known_enemy()}
    world, self_state = make_world(fuel=700, tanks=tanks)
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
    terrain = InMemoryTerrainMap(terrain_data={(150, 100): "W"})
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        inventory,
        100000,
        terrain,
        "",
        ((150, 100),),
    )

    with pytest.raises(SessionExitError, match="no_viable_targets"):
        decide_hunt_mode(ctx)


def test_hunt_refuels_in_place_when_no_dot_makes_progress() -> None:
    """With no strict-progress dot, the bot refuels in ANY direction.

    User ruling 2026-07-19 after run 14:49: rejoined at fuel 653 with
    an enemy 26.6 tiles away, 622 usable dots around it, and only
    water-locked dots strictly closer -- the strict-progress relay
    starved the bot amid plenty and exited at tick 4. The deficit was
    fuel, not distance: hop to the best fresh dot regardless of
    direction, get richer, then pounce. Here the only dot (50,100) is
    BEHIND the bot relative to the enemy at (240,100), so the relay
    declines it but the refuel fallback takes it.
    """
    tanks: dict[str, TankStateDict] = {"60": _map_known_enemy()}
    world, self_state = make_world(fuel=700, tanks=tanks)
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
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        inventory,
        100000,
        None,
        "",
        ((50, 100),),
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 50
    assert decision["command"]["target_y"] == 100
    assert decision["behavior"]["reason_kind"] == "hunt_refuel"
    assert decision["behavior"]["mode"] == "HUNT"


def test_hunt_refuel_exits_at_fuel_capacity() -> None:
    """At fuel capacity a still-unaffordable enemy is genuinely out of range.

    Refueling cannot help a tank already at its cap (rank 2 -> 1200),
    so the refuel fallback declines and the fail-hard session exit is
    correct: the enemy at 140 tiles needs 840 + 650 = 1490 fuel
    end-to-end, beyond what this rank can ever carry.
    """
    tanks: dict[str, TankStateDict] = {"60": _map_known_enemy()}
    world, self_state = make_world(fuel=1200, tanks=tanks)
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
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        inventory,
        100000,
        None,
        "",
        ((50, 100),),
    )

    with pytest.raises(SessionExitError, match="no_viable_targets"):
        decide_hunt_mode(ctx)


def test_hunt_pursuit_aim_is_clamped_into_viewport() -> None:
    """Pursuit fires at a viewport-legal tile, never the raw off-viewport coords.

    The server rejects any shoot aim outside the visible viewport
    (0x52 code 0, live run 2026-07-03 20:34: five rejections aiming
    at a pursuit target 5 rows below the viewport). The aim is only a
    hint -- the server picks homing from the target_id and the seeker
    tracks -- so the dispatch clamps the registry coordinate onto the
    viewport bounds. Registry truth (combat_target_x/y) keeps the real
    position for the stationary-miss comparison.
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
            "last_shot_target_id": 50,
            "last_shot_target_name": "Runner",
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"
    # Viewport is (92,92)-(107,107); the raw registry coords (150,150)
    # are clamped onto the boundary.
    assert decision["command"]["target_x"] == 107
    assert decision["command"]["target_y"] == 107
    assert decision["command"]["target_id"] == 50
    # Registry truth is preserved on the lock.
    assert decision["updated_ai_state"]["combat_target_x"] == 150
    assert decision["updated_ai_state"]["combat_target_y"] == 150
