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
        # Left the viewport 8 s ago -- inside the ~12 s homing trace
        # ([[shoot-event-format]]#reroute-ttl-ms), so pursuit fire is
        # still live; the trace-expired behavior has its own pin.
        last_viewport_observation_ms=92000,
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


def test_hunt_acquire_teleports_back_to_an_affordable_off_viewport_lock() -> None:
    """ACQUIRE returns to an off-viewport lock by teleport, keeping it.

    User contract (2026-07-25): a lock that reaches ACQUIRE with its
    target off-viewport is an engagement resumed after a mode
    interrupt, and the restock cycle does not abandon the target.
    The bot never fires from stand-off range on resume (user contract
    2026-07-02) -- with a fresh position and an affordable return
    (~127 + 650 against 800 here) it teleports straight back to the
    locked tank (the recorded human behavior: purple-1 was resumed by
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
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_hunt_acquire_refuels_with_lock_held_when_return_unaffordable() -> None:
    """ACQUIRE keeps an off-viewport lock whose return is unaffordable.

    Refuel-then-RESUME (user ruling 2026-07-27): when the return
    teleport plus the kill budget plus the fuel-low reserve exceeds
    the tank (here ~424 + 650 against 800), the tick delegates to
    fuel recovery WITH the lock held, so the 2026-07-25 resume
    machinery returns to this exact target once the trip is fundable.
    The old release-and-reacquire at this branch lost run 183703's
    red-1 to a fresh distance race. Firing from stand-off range stays
    forbidden (live run 2026-07-01 20:48: eleven rejected shots at a
    target 92 tiles away) -- the bot collects, it does not shoot.

    2026-07-29: this shape is now BOT-specific -- a locked HUMAN
    beyond funds relays toward them instead (unlimited-distance
    human pursuit), covered by
    ``test_locked_human_beyond_funds_relays_with_lock_held``.
    """
    tanks: dict[str, TankStateDict] = {"50": _pursuit_target(x=150, y=150, name="red-4")}
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
    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["combat_target_id"] == 50
    assert decision["updated_ai_state"]["blocked_combat_targets"] == {}


def test_hunt_acquire_returns_to_the_locked_target_after_a_mode_interrupt() -> None:
    """ACQUIRE teleports back to an affordable off-viewport lock.

    User contract (2026-07-25): the restock cycle does not abandon
    the target -- damage persists, so a bot resuming HUNT at full
    stock returns to the same tank it was fighting. The pursuit
    position is fresh and the return (cost ~120 + 650 reserve) fits
    the 1200-fuel tank, so the decision is a teleport at the locked
    target with the lock retained.
    """
    tanks: dict[str, TankStateDict] = {"50": _pursuit_target(x=120, y=100)}
    world, self_state = make_world(fuel=1200, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
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
    assert decision["behavior"]["target_x"] == 120
    assert decision["behavior"]["target_y"] == 100
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_hunt_acquire_refreshes_a_stale_locked_position_via_map() -> None:
    """A resumed lock with a stale position opens the map, keeping the lock.

    The pursuit target's last observation predates the map-open
    cooldown window, so teleporting at those coordinates would commit
    fuel to a tile the enemy may have left. The resume path refreshes
    via map_open first; the lock survives for the post-refresh tick.
    """
    stale = make_tank_state(
        tank_id=50,
        x=120,
        y=100,
        team=2,
        rank=1,
        name="Runner",
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=90000,
        last_wire_seen_ms=90000,
        last_position_update_ms=90000,
        last_viewport_observation_ms=80000,
    )
    world, self_state = make_world(fuel=1200, tanks={"50": stale})
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
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
    assert decision["updated_ai_state"]["combat_target_id"] == 50


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


def test_scan_on_landing_shoots_when_target_in_range() -> None:
    """SCAN_ON_LANDING fires from position at an in-range visible target.

    User ruling 2026-07-29: in-view + in-range means shoot from the
    current tile; the close teleport is reserved for beyond
    ``SHOT_RANGE_TILES``.
    """
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

    assert decision["command"]["cmd_type"] == "shoot"


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
    world, self_state = make_world(fuel=1090, tanks=tanks)
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
    world, self_state = make_world(fuel=1090, tanks=tanks)
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
    world, self_state = make_world(fuel=1090, tanks=tanks)
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


def test_pursuit_fire_stops_when_the_homing_trace_expires() -> None:
    """A departed target past the ~12 s trace gets the map chase, not a shot.

    Flags 4 and 5 of run bot-20260730-01xx: seven pursuit homings hit,
    then one shot always resolved after the reroute wall as a booked
    miss and a wasted tick ("couldnt we avoid the missed shot
    entirely? and save a tick"). Past ``PURSUIT_TRACE_TTL_MS`` the
    pursuit goes straight to the map refresh the miss would have
    bought anyway, with the lock held.
    """
    stale_target = make_tank_state(
        tank_id=50,
        x=150,
        y=150,
        team=2,
        rank=1,
        name="Runner",
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=100000,
        last_wire_seen_ms=100000,
        last_position_update_ms=100000,
        last_viewport_observation_ms=80000,
    )
    world, self_state = make_world(fuel=800, tanks={"50": stale_target})
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
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


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


def test_hunt_acquire_teleports_at_an_affordable_map_known_enemy() -> None:
    """A map-fresh enemy inside the affordability gate is teleport-acquired.

    No viewport-confirmed threat and no lock exist; the enemy is known
    only from the map snapshot (fresh ``timestamp_ms``, no viewport
    observation), close enough that teleport + kill budget + reserve
    fits inside the tank -- the acquisition path teleports at it.
    """
    tanks: dict[str, TankStateDict] = {
        "60": _map_known_enemy(x=130, y=100),
    }
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
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 60


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
        ((170, 100), (130, 100)),
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 170
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


def test_unaffordable_human_outranks_affordable_bot_at_acquisition() -> None:
    """A rank-window human beyond the horizon preempts nearby bot farming.

    User ruling 2026-07-29 ("unlimited distance for humans... this is
    the real deal"), born from the Yuppler encounter: Yuppler at dist
    95 was rejected ``unaffordable`` while the bot farmed red-3 at
    dist 19. With an affordable practice bot AND an unaffordable
    human on the fresh map, the decision must be a dot-relay leg
    toward the HUMAN, not a teleport at the bot.
    """
    tanks: dict[str, TankStateDict] = {
        "60": _map_known_enemy(tank_id=60, x=115, y=100, name="red-5"),
        "90": _map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler"),
    }
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
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        None,
        "",
        # (150,100) closes distance to Yuppler and is affordable.
        ((150, 100),),
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 150
    assert decision["command"]["target_y"] == 100
    assert decision["behavior"]["reason_kind"] == "dot_relay"


def test_human_pursuit_falls_back_to_bot_when_no_leg_helps() -> None:
    """With no progress dot and a full tank, the bot farms while waiting.

    The pursuit must not deadlock the session: when no dot closes
    distance to the human and refuel-in-place is pointless (already
    at capacity), the affordable bot is engaged and the next map
    re-evaluates the pursuit.
    """
    tanks: dict[str, TankStateDict] = {
        "60": _map_known_enemy(tank_id=60, x=115, y=100, name="red-5"),
        "90": _map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler"),
    }
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
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 60


def test_recruit_human_is_not_pursued() -> None:
    """A rank-0 human stays protected -- no relay chain toward them.

    The rank window rejects recruits BEFORE the affordability gate
    (reason ``protected_human_rank``, not ``unaffordable``), so the
    pursuit helper can never travel toward one and the affordable bot
    is farmed normally.
    """
    tanks: dict[str, TankStateDict] = {
        "60": _map_known_enemy(tank_id=60, x=115, y=100, name="red-5"),
        "90": make_tank_state(
            tank_id=90,
            x=240,
            y=100,
            team=2,
            rank=0,
            name="Yuppler",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=99800,
        ),
    }
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
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        None,
        "",
        ((150, 100),),
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 60


def test_locked_human_beyond_funds_relays_with_lock_held() -> None:
    """A locked human who teleported beyond funds is chased leg by leg.

    User ruling 2026-07-29: "even if they teleport super far away."
    The return costs 840 + the 650 engagement floor at fuel 700, so
    the plain resume cannot fund it -- the decision must be a relay
    leg toward the human with ``combat_target_id`` retained
    (never-drop rides through every leg).
    """
    tanks: dict[str, TankStateDict] = {
        "90": _map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler"),
    }
    world, self_state = make_world(fuel=700, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
            "combat_target_id": 90,
            "combat_target_x": 240,
            "combat_target_y": 100,
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        None,
        "",
        ((150, 100),),
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 150
    assert decision["command"]["target_y"] == 100
    assert decision["behavior"]["reason_kind"] == "dot_relay"
    assert decision["updated_ai_state"]["combat_target_id"] == 90


def test_locked_bot_beyond_funds_still_refuels_in_place() -> None:
    """The relay-resume is human-only: a bot lock keeps the plain refuel.

    Practice bots never flee across the map, so the 2026-07-27
    refuel-then-resume (get richer in place, return when fundable)
    remains the right shape for them -- guards the ``is_human_name``
    gate on the new relay branch.
    """
    tanks: dict[str, TankStateDict] = {
        "90": _map_known_enemy(tank_id=90, x=240, y=100, name="red-9"),
    }
    world, self_state = make_world(fuel=700, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
            "combat_target_id": 90,
            "combat_target_x": 240,
            "combat_target_y": 100,
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        None,
        "",
        ((150, 100),),
    )

    decision = decide_hunt_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["combat_target_id"] == 90


def test_locked_human_with_no_relay_leg_falls_back_to_refuel() -> None:
    """When no relay leg helps, the locked-human resume uses plain refuel.

    At fuel capacity with no progress dot, ``_relay_toward`` returns
    ``None`` (refuel-in-place is pointless at a full tank), so the
    resume falls through to the 2026-07-27 refuel-for-hunt path --
    whose collect cascade also declines at capacity and terminates
    via the blocked-target replan rather than deadlocking the tick.
    """
    tanks: dict[str, TankStateDict] = {
        "90": _map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler"),
    }
    world, self_state = make_world(fuel=1100, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
            "combat_target_id": 90,
            "combat_target_x": 240,
            "combat_target_y": 100,
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

    decision = decide_hunt_mode(ctx)

    # The tick advances (no relay teleport was possible); the exact
    # fallback shape is refuel_for_hunt's contract, not re-tested here.
    assert decision["behavior"]["reason_kind"] != "dot_relay"


def test_relay_leg_cost_is_capped_at_the_engagement_budget() -> None:
    """A max-progress dot costing more than one kill budget is skipped.

    Regression for the 2026-07-29 21:17:40 broke-arrival: the uncapped
    picker paid 787 fuel in one leg (1100 -> 313) and landed next to
    Yuppler unable to fight, stranding the pursuit in a minutes-long
    restock. The near-enemy dot here costs ~780 (affordable under the
    old floor-only rule at a full tank) and must lose to the cheaper
    progress dot at ~300.
    """
    tanks: dict[str, TankStateDict] = {
        "90": _map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler"),
    }
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
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        None,
        "",
        # (230,100): most progress, cost ~780 -- beyond the 450 leg cap.
        # (150,100): cost 300 -- the correct capped leg.
        ((230, 100), (150, 100)),
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 150
    assert decision["command"]["target_y"] == 100
    assert decision["behavior"]["reason_kind"] == "dot_relay"


def test_stale_known_human_forces_a_map_refresh_over_bot_farming() -> None:
    """A map-stale rank-window human is refreshed before settling for bots.

    The freshness asymmetry that hid Yuppler (2026-07-29 21:19):
    practice bots stay wire-fresh by moving; a quiet human goes stale
    ``map_open_cooldown_ms`` after every map open, and with a fresh
    bot always available acquisition never reopened the map. Here the
    bot is map-fresh via the wire, the human's timestamp has aged
    out, and the map itself is older than the cooldown -- the
    decision must be a map refresh, not a teleport at red-5.
    """
    tanks: dict[str, TankStateDict] = {
        "60": _map_known_enemy(tank_id=60, x=115, y=100, name="red-5"),
        "90": _map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler", timestamp_ms=80000),
    }
    world, self_state = make_world(fuel=1100, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 90000,
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"


def test_fresh_map_showing_stale_human_farms_normally() -> None:
    """A FRESH map that still shows the human stale means they left.

    No refresh can cure a human absent from the latest snapshot, so
    the affordable bot is engaged -- the refresh rule cannot loop.
    """
    tanks: dict[str, TankStateDict] = {
        "60": _map_known_enemy(tank_id=60, x=115, y=100, name="red-5"),
        "90": _map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler", timestamp_ms=80000),
    }
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
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 60


def test_stale_human_exists_filters_and_reasons() -> None:
    """The stale-human predicate skips allies, bots, and non-stale humans.

    Direct contract test: an allied human (same team), a wire-fresh
    practice bot, and a BLOCKED human must all fail to trigger the
    refresh; only a human whose sole curable defect is stale map data
    returns True.
    """
    from tankpit_bot.bot.ai.threats import stale_human_exists

    ally_human = make_tank_state(
        tank_id=70,
        x=110,
        y=100,
        team=1,
        rank=2,
        name="FriendlyHuman",
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=80000,
    )
    fresh_bot = _map_known_enemy(tank_id=60, x=115, y=100, name="red-5")
    blocked_human = _map_known_enemy(tank_id=80, x=200, y=100, name="Blocked")
    stale_human = _map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler", timestamp_ms=80000)

    def check(tanks: dict[str, TankStateDict], blocked: dict[str, int]) -> bool:
        world, self_state = make_world(fuel=1100, tanks=tanks)
        return stale_human_exists(
            world,
            self_state,
            blocked,
            {},
            None,
            100000,
            5000,
            engagement_reserve_fuel=650,
        )

    assert check({"70": ally_human, "60": fresh_bot}, {}) is False
    assert check({"80": blocked_human}, {"80": 100000}) is False
    assert check({"90": stale_human}, {}) is True


def test_relay_skips_progress_dot_below_the_fuel_floor() -> None:
    """A capped-cost dot that would dip below the reserve is skipped.

    At fuel 400, the 300-cost progress dot passes the 450 leg cap but
    would leave 100 < the 200 floor -- the cheaper dot wins instead.
    """
    tanks: dict[str, TankStateDict] = {
        "90": _map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler"),
    }
    world, self_state = make_world(fuel=400, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        None,
        "",
        ((150, 100), (110, 100)),
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 110
    assert decision["command"]["target_y"] == 100
    assert decision["behavior"]["reason_kind"] == "dot_relay"


def test_scan_on_landing_pursuit_past_the_trace_wall_chases_via_map() -> None:
    """SCAN_ON_LANDING's pursuit branch also respects the homing-trace wall.

    The ENGAGE-path wall has its own pin; this covers the second
    call-site (audit 2026-07-30): a locked target that left the
    viewport more than PURSUIT_TRACE_TTL_MS ago gets the map chase
    instead of a guaranteed-miss shot.
    """
    stale = _pursuit_target(x=150, y=150)
    stale["last_viewport_observation_ms"] = 87000  # 13 s > the 12 s wall
    tanks: dict[str, TankStateDict] = {"50": stale}
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

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason_kind"] == "find_target"
