"""Dot-relay travel and refuel-in-place toward out-of-range enemies."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._support import (
    make_inventory,
    make_map_known_enemy,
    make_pursuit_target,
    make_scanned_ai_state,
    make_world,
)


def test_hunt_acquire_teleports_at_an_affordablemake_map_known_enemy() -> None:
    """A map-fresh enemy inside the affordability gate is teleport-acquired.

    No viewport-confirmed threat and no lock exist; the enemy is known
    only from the map snapshot (fresh ``timestamp_ms``, no viewport
    observation), close enough that teleport + kill budget + reserve
    fits inside the tank -- the acquisition path teleports at it.
    """
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "60": make_map_known_enemy(x=130, y=100),
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
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "", ws=ws)

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
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "60": make_map_known_enemy(),
        # Stale map entry: rejected for a non-affordability reason, so
        # the relay must not travel toward it.
        "70": make_map_known_enemy(tank_id=70, x=110, y=100, name="red-26", timestamp_ms=10),
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
        ws=ws,
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
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "60": make_map_known_enemy(),
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
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 170
    assert decision["command"]["target_y"] == 100


def test_hunt_relay_tie_breaks_on_cheaper_hop() -> None:
    """Dots equidistant from the enemy keep the cheaper teleport."""
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"60": make_map_known_enemy()}
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
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 220
    assert decision["command"]["target_y"] == 100


def test_hunt_relay_exits_when_only_dot_is_impassable() -> None:
    """A relay with no passable progress dot still exits the session."""
    from tests.in_memory_terrain_map import InMemoryTerrainMap

    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"60": make_map_known_enemy()}
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
        ws=ws,
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
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"60": make_map_known_enemy()}
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
        ws=ws,
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
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"60": make_map_known_enemy()}
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
        ws=ws,
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
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": make_pursuit_target(x=150, y=150)}
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
            "last_shot_target_name": "red-9",
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

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
