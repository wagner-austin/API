"""Tests for the durable fuel recovery owner."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.recover_fuel_mode import (
    can_use_fuel_radar,
    decide_recover_fuel_mode,
    minimum_recovery_fuel_volume,
    select_fuel_target,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import (
    make_inventory,
    make_post_radar_ai_state,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def test_recover_fuel_mode_continues_locked_fuel_target() -> None:
    """The durable owner keeps an executable locked fuel target."""
    world, self_state = make_world(
        fuel=250,
        containers={
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "resource_target_kind": "fuel",
            "resource_target_x": 105,
            "resource_target_y": 105,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT_FUEL"
    assert decision["behavior"]["reason"] == "fuel=700"
    assert decision["command"]["cmd_type"] == "pickup_fuel"


def test_recover_fuel_mode_clears_combat_lock_when_fuel_mode_owns_tick() -> None:
    """Fuel recovery drops stale combat ownership when taking control."""
    world, self_state = make_world(
        fuel=250,
        containers={
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 120,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["updated_ai_state"]["combat_target_id"] == -1
    assert decision["updated_ai_state"]["combat_target_x"] == 0
    assert decision["updated_ai_state"]["combat_target_y"] == 0


def test_recover_fuel_mode_grabs_adjacent_equipment_opportunistically() -> None:
    """Fuel recovery picks up visible equipment before chasing fuel.

    Regression guard for live run 20260610-011x: the bot walked past
    equipment containers because the mode only looked for fuel.
    """
    world, self_state = make_world(
        fuel=400,
        containers={
            "101,100": make_container_state(
                x=101,
                y=100,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["reason"] == "opportunistic_equipment"
    assert decision["command"]["cmd_type"] == "pickup_equipment"
    assert decision["behavior"]["target_x"] == 101
    assert decision["behavior"]["target_y"] == 100


def test_recover_fuel_mode_releases_lock_for_markedly_closer_fuel() -> None:
    """A locked far container yields to abundant nearby fuel.

    Regression guard for live run 20260610-011x: the bot walked across
    the map to a locked container while ignoring closer fuel the whole
    way.
    """
    world, self_state = make_world(
        fuel=250,
        containers={
            "107,107": make_container_state(
                x=107,
                y=107,
                is_fuel=True,
                volume=900,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
            "160,160": make_container_state(
                x=160,
                y=160,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "resource_target_kind": "fuel",
            "resource_target_x": 160,
            "resource_target_y": 160,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["reason"] == "fuel=900"
    assert decision["behavior"]["target_x"] == 107
    assert decision["behavior"]["target_y"] == 107
    assert decision["updated_ai_state"]["resource_target_x"] == 107


def test_recover_fuel_mode_keeps_lock_against_marginally_closer_fuel() -> None:
    """A candidate inside the anti-churn threshold does not break the lock."""
    world, self_state = make_world(
        fuel=250,
        containers={
            "104,104": make_container_state(
                x=104,
                y=104,
                is_fuel=True,
                volume=900,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "resource_target_kind": "fuel",
            "resource_target_x": 105,
            "resource_target_y": 105,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["reason"] == "fuel=700"
    assert decision["behavior"]["target_x"] == 105
    assert decision["behavior"]["target_y"] == 105


def test_recover_fuel_mode_uses_radar_when_viewport_needs_authoritative_scan() -> None:
    """The durable owner senses before repositioning in an unscanned viewport."""
    world, self_state = make_world(fuel=250, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT_FUEL"
    assert decision["behavior"]["reason"] == "forage_radar"
    assert decision["command"]["cmd_type"] == "radar"


def test_recover_fuel_mode_uses_regular_radar_when_extra_charges_are_empty() -> None:
    """Fuel recovery still scans with free radar when extras are depleted."""
    world, self_state = make_world(fuel=250, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 1
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["reason"] == "forage_radar"
    assert decision["command"]["cmd_type"] == "radar"


def test_recover_fuel_mode_raises_when_genuinely_boxed_in() -> None:
    """The durable fuel owner raises when no productive recovery exists.

    With the current viewport already scanned, every tile water,
    no affordable search hop, and no atlas-known fuel dot, the bot
    has nothing legal to do. The map_intel terminal was removed
    2026-06-22; the recovery owner raises to make the stuck state
    loud instead of silently spamming map_open.
    """
    world, self_state = make_world(fuel=140, scanned=True)
    ai_state = AIStateDict(
        **{
            **make_post_radar_ai_state(world),
            "mode": "RECOVER_FUEL",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 1
    terrain_data: dict[tuple[int, int], str] = {}
    for x in range(92, 108):
        for y in range(92, 108):
            terrain_data[(x, y)] = "W"
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

    with pytest.raises(ValueError, match="RECOVER_FUEL owner produced no decision"):
        decide_recover_fuel_mode(ctx)


def test_select_fuel_target_returns_none_for_unreachable_off_viewport_target() -> None:
    """Out-of-viewport fuel with no walkable approach and no fuel returns None."""
    from tankpit_bot.sniffer.world_state import mark_move_target_failed, reset_world_state

    reset_world_state()
    world, self_state = make_world(
        self_x=100,
        self_y=100,
        fuel=800,
        scanned=True,
        containers={
            "103,100": make_container_state(
                x=103,
                y=100,
                is_fuel=True,
                volume=500,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    mark_move_target_failed(103, 100, 99000)
    terrain = InMemoryTerrainMap(terrain_data={})
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        terrain,
        "",
    )

    assert select_fuel_target(ctx, allow_unreachable=True) is None
    reset_world_state()


def test_select_fuel_target_dispatches_pickup_for_in_viewport_target() -> None:
    """In-viewport fuel dispatches pickup_fuel regardless of walkability.

    Pre-2026-06-21 the planner gave up when the only path was a
    teleport the bot couldn't afford. The new logic dispatches
    ``pickup_fuel`` directly: it's a single server-routed command,
    no teleport needed, and the server walks the bot toward the
    container.
    """
    world, self_state = make_world(
        fuel=0,
        containers={
            "103,100": make_container_state(
                x=103,
                y=100,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    terrain_data: dict[tuple[int, int], str] = {(101, y): "#" for y in range(92, 109)}
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        terrain,
        "",
    )

    selected = select_fuel_target(ctx, allow_unreachable=True)
    if selected is None:
        raise AssertionError("expected select_fuel_target to dispatch a pickup")
    _container, command = selected
    assert command["cmd_type"] == "pickup_fuel"
    assert command["target_x"] == 103
    assert command["target_y"] == 100


def test_selects_low_volume_fuel_when_critically_low() -> None:
    """Critical fuel recovery accepts small visible fuel containers."""
    world, self_state = make_world(
        fuel=40,
        containers={
            "101,100": make_container_state(
                x=101,
                y=100,
                is_fuel=True,
                volume=57,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    inventory = make_inventory()
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["behavior"]["reason"] == "fuel=57"


def test_recover_fuel_mode_walks_to_unscanned_tile_when_radar_too_costly() -> None:
    """Fuel recovery walks within the viewport when the radar fuel cost is unaffordable.

    With fuel below the radar + operating-reserve floor the forager
    cannot fire a radar, but it CAN walk -- moves are free, so the
    bot walks toward the nearest unscanned tile so the next tick's
    free radar (or a paid radar once refueled) reveals new ground.
    The OLD edge-walk fallback fired only because the legacy gate
    skipped the forager entirely; the tile-aware forager prefers
    in-viewport sweeping over a directionless edge step.
    """
    world, self_state = make_world(fuel=5, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["command"]["cmd_type"] == "move"
    assert decision["behavior"]["reason"] == "forage_sweep"
    assert decision["behavior"]["mode"] == "COLLECT_FUEL"


def test_can_use_fuel_radar_keeps_operating_reserve() -> None:
    """Fuel radar needs cost plus reserve; inventory never matters.

    Regression guard for live run 20260612-131003: 64 unreserved scans
    burned fuel to 7 and stranded the session for 28 minutes.
    """
    world, self_state = make_world(fuel=110, scanned=False)
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")

    assert can_use_fuel_radar(ctx) is True
    assert minimum_recovery_fuel_volume(ctx) == 1

    low_fuel_world, low_fuel_self_state = make_world(fuel=109, scanned=False)
    low_fuel_ctx = DecideCtx(
        low_fuel_world,
        low_fuel_self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        None,
        "",
    )

    assert can_use_fuel_radar(low_fuel_ctx) is False
    assert minimum_recovery_fuel_volume(low_fuel_ctx) == 1

    healthy_world, healthy_self_state = make_world(fuel=800, scanned=False)
    healthy_ctx = DecideCtx(
        healthy_world,
        healthy_self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        None,
        "",
    )

    assert minimum_recovery_fuel_volume(healthy_ctx) == 100

    inventory["extra_radars"]["count"] = 1
    uncharged_ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        None,
        "",
    )

    assert can_use_fuel_radar(uncharged_ctx) is True


def test_fuel_recovery_sweeps_equipment_before_search_hop() -> None:
    """When no fuel target exists but equipment is in the viewport, sweep it."""
    containers = {
        "102,100": make_container_state(
            x=102,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(fuel=400, scanned=True, containers=containers)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["dual_shots"]["count"] = 3
    inventory["homing_shots"]["count"] = 3
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["reason"] == "opportunistic_equipment"
    assert decision["command"]["cmd_type"] == "pickup_equipment"
    assert decision["command"]["target_x"] == 102
    assert decision["command"]["target_y"] == 100


def test_fuel_recovery_sweeps_equipment_at_critical_fuel() -> None:
    """Sweep fires when fuel is critical and no fuel containers exist."""
    containers = {
        "103,100": make_container_state(
            x=103,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(fuel=200, scanned=True, containers=containers)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["dual_shots"]["count"] = 3
    inventory["homing_shots"]["count"] = 3
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["reason"] == "sweep_equipment"
    assert decision["behavior"]["target_x"] == 103


def test_locked_fuel_clears_when_water_locked() -> None:
    """A locked fuel target on water clears when fully boxed in."""
    terrain_data: dict[tuple[int, int], str] = {}
    for x in range(92, 108):
        for y in range(92, 108):
            terrain_data[(x, y)] = "W"
    terrain_data[(100, 100)] = InMemoryTerrainMap.GROUND
    terrain_data[(120, 100)] = "W"
    terrain_data[(121, 100)] = "W"
    terrain_data[(119, 100)] = "W"
    terrain_data[(120, 101)] = "W"
    terrain_data[(120, 99)] = "W"
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    world, self_state = make_world(
        self_x=100,
        self_y=100,
        fuel=300,
        scanned=True,
        containers={
            "120,100": make_container_state(
                x=120,
                y=100,
                is_fuel=True,
                volume=500,
                timestamp_ms=100000,
            )
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "",
            "mode_started_ms": 90000,
            "resource_target_kind": "fuel",
            "resource_target_x": 120,
            "resource_target_y": 100,
        }
    )
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["updated_ai_state"]["resource_target_kind"] == ""
