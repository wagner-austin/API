"""HUNT phase routing: lock resume, close/engage, confirm-kill, landing scan."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._support import (
    make_enemy_tank,
    make_inventory,
    make_pursuit_target,
    make_scanned_ai_state,
    make_world,
)


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
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": make_pursuit_target(x=115, y=115)}
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

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
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": make_pursuit_target(x=150, y=150, name="red-4")}
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

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
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": make_pursuit_target(x=120, y=100)}
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

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
    ws = WorldService()
    stale = make_tank_state(
        tank_id=50,
        x=120,
        y=100,
        team=2,
        rank=1,
        name="red-9",
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_hunt_refresh_refuels_when_close_action_is_not_legal() -> None:
    """Refresh delegates to fuel recovery when combat teleport is unaffordable."""
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "50": make_enemy_tank(x=190, y=100, name="red-50"),
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
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        inventory,
        100000,
        None,
        "",
        ((140, 100),),
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["mode"] == "COLLECT"


def test_hunt_close_enters_confirm_kill_when_locked_target_disappears() -> None:
    """Close state explicitly transitions through confirm-kill when target vanishes."""
    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason_kind"] == "confirm_kill"


def test_hunt_close_returns_close_decision_for_visible_target() -> None:
    """Close state re-teleports to a non-adjacent target to close distance."""
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "50": make_enemy_tank(),
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"


def test_hunt_close_refuels_when_close_action_is_not_legal() -> None:
    """Close state delegates to fuel recovery when combat teleport is unaffordable."""
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "50": make_enemy_tank(x=190, y=100, name="red-50"),
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
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        inventory,
        100000,
        None,
        "",
        ((140, 100),),
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["mode"] == "COLLECT"


def test_hunt_engage_enters_confirm_kill_when_locked_target_disappears() -> None:
    """Engage state explicitly transitions through confirm-kill when target vanishes."""
    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason_kind"] == "confirm_kill"
    assert decision["updated_ai_state"]["combat_target_id"] == -1


def test_hunt_engage_shoots_visible_locked_target() -> None:
    """Engage state keeps shooting a visible locked target."""
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "50": make_enemy_tank(x=101, y=100),
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["behavior"]["reason_kind"] == "shoot_target"


def test_hunt_engage_confirms_killed_target_with_explicit_reason() -> None:
    """Engage confirmation also handles targets already on the kill cooldown."""
    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason_kind"] == "confirm_kill"
    assert decision["updated_ai_state"]["combat_target_id"] == -1


def test_hunt_engage_without_locked_target_id_still_confirms_and_searches() -> None:
    """Engage handles a missing target id without crashing or reusing stale state."""
    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

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
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "50": make_enemy_tank(name="red-8"),
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_scan_on_landing_reacquires_when_target_gone() -> None:
    """SCAN_ON_LANDING falls back to acquire when the locked target despawned."""
    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason_kind"] == "find_enemies"


def test_scan_on_landing_engages_when_target_present() -> None:
    """SCAN_ON_LANDING transitions to ENGAGE when target is still visible."""
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": make_enemy_tank(x=101, y=100)}
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"


def test_scan_on_landing_shoots_when_target_in_range() -> None:
    """SCAN_ON_LANDING fires from position at an in-range visible target.

    User ruling 2026-07-29: in-view + in-range means shoot from the
    current tile; the close teleport is reserved for beyond
    ``SHOT_RANGE_TILES``.
    """
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": make_enemy_tank(x=105, y=100)}
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"
