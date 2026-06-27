"""Tests for the durable fuel recovery owner."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.collect_mode import (
    decide_collect_mode,
    select_fuel_target,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import (
    make_inventory,
    make_post_radar_ai_state,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def test_collect_mode_continues_locked_fuel_target() -> None:
    """The durable owner keeps an executable locked fuel target."""
    world, self_state = make_world(
        fuel=150,
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
            "mode": "COLLECT",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "resource_target_kind": "fuel",
            "resource_target_x": 105,
            "resource_target_y": 105,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["behavior"]["reason"] == "fuel=700"
    assert decision["command"]["cmd_type"] == "pickup_fuel"


def test_collect_mode_preserves_combat_lock_across_recovery() -> None:
    """Fuel recovery does NOT clear ``combat_target_id``.

    Recovery preserves the held lock so the bot can resume the same
    engagement after refueling rather than re-acquiring a fresh target.
    Equipment recovery has always preserved the lock; fuel recovery was
    flipped to match 2026-06-23 so that "bail to refuel mid-fight,
    finish the kill after" is a single coherent behaviour. HUNT's
    resume path (``_decide_hunt_acquire``) reads the persisted
    ``combat_target_id`` and continues the engagement when the lock is
    still viable.
    """
    world, self_state = make_world(
        fuel=150,
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
            "mode": "COLLECT",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 120,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["updated_ai_state"]["combat_target_id"] == 50
    assert decision["updated_ai_state"]["combat_target_x"] == 120
    assert decision["updated_ai_state"]["combat_target_y"] == 100


def test_collect_mode_grabs_adjacent_equipment_before_fuel() -> None:
    """COLLECT picks up visible equipment before chasing visible fuel.

    Regression guard for live run 20260610-011x: the bot walked past
    equipment containers because the old fuel-recovery mode only looked
    for fuel. Under the unified cascade equipment ranks ahead of fuel.
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
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    assert decision["behavior"]["reason"] == "equipment_restock"
    assert decision["command"]["cmd_type"] == "pickup_equipment"
    assert decision["behavior"]["target_x"] == 101
    assert decision["behavior"]["target_y"] == 100


def test_collect_mode_releases_lock_for_markedly_closer_fuel() -> None:
    """A locked far container yields to abundant nearby fuel.

    Regression guard for live run 20260610-011x: the bot walked across
    the map to a locked container while ignoring closer fuel the whole
    way.
    """
    world, self_state = make_world(
        fuel=150,
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
            "mode": "COLLECT",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "resource_target_kind": "fuel",
            "resource_target_x": 160,
            "resource_target_y": 160,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    assert decision["behavior"]["reason"] == "fuel=900"
    assert decision["behavior"]["target_x"] == 107
    assert decision["behavior"]["target_y"] == 107
    assert decision["updated_ai_state"]["resource_target_x"] == 107


def test_collect_mode_keeps_lock_against_marginally_closer_fuel() -> None:
    """A candidate inside the anti-churn threshold does not break the lock."""
    world, self_state = make_world(
        fuel=150,
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
            "mode": "COLLECT",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "resource_target_kind": "fuel",
            "resource_target_x": 105,
            "resource_target_y": 105,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    assert decision["behavior"]["reason"] == "fuel=700"
    assert decision["behavior"]["target_x"] == 105
    assert decision["behavior"]["target_y"] == 105


def test_collect_mode_uses_radar_when_viewport_needs_authoritative_scan() -> None:
    """The durable owner senses before repositioning in an unscanned viewport."""
    world, self_state = make_world(fuel=150, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["behavior"]["reason"] == "forage_radar"
    assert decision["command"]["cmd_type"] == "radar"


def test_collect_mode_uses_regular_radar_when_extra_charges_are_empty() -> None:
    """Fuel recovery still scans with free radar when extras are depleted."""
    world, self_state = make_world(fuel=150, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 1
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    assert decision["behavior"]["reason"] == "forage_radar"
    assert decision["command"]["cmd_type"] == "radar"


def test_collect_mode_raises_when_genuinely_boxed_in() -> None:
    """The durable fuel owner raises when no productive recovery exists.

    With the current viewport already scanned, every tile water,
    and fuel below even the short-hop cost (8 tiles * 6 = 48), the
    bot has nothing legal to do. The ``hunt_min_fuel`` reserve
    drop (2026-06-24) means the bot can now short-hop at very low
    fuel, so the genuine-stranding threshold is fuel < short-hop
    cost rather than fuel < cost + reserve.
    """
    world, self_state = make_world(fuel=30, scanned=True)
    ai_state = AIStateDict(
        **{
            **make_post_radar_ai_state(world),
            "mode": "COLLECT",
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

    with pytest.raises(ValueError, match="COLLECT owner produced no decision"):
        decide_collect_mode(ctx)


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

    assert select_fuel_target(ctx) is None
    reset_world_state()


def test_select_fuel_target_rejects_walk_unreachable_in_viewport() -> None:
    """Walk-unreachable in-viewport fuel is not selected (walkable-only).

    The server's long-press pickup walks the tank in a straight
    line; if water or rocks block that path it returns CANT_GO and
    one rejection flags the container ``failed_pickups`` for the
    whole session. The fix shipped 2026-06-24 is to never select
    containers the bot cannot walk to in the first place.
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

    assert select_fuel_target(ctx) is None


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
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["behavior"]["reason"] == "fuel=57"


def test_collect_mode_walks_to_unscanned_tile_when_radar_too_costly() -> None:
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
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    assert decision["command"]["cmd_type"] == "move"
    assert decision["behavior"]["reason"] == "forage_sweep"
    assert decision["behavior"]["mode"] == "COLLECT"


def test_collect_takes_visible_equipment_before_search_hop() -> None:
    """When equipment is in the viewport, COLLECT grabs it before hopping."""
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
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["dual_shots"]["count"] = 3
    inventory["homing_shots"]["count"] = 3
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    assert decision["behavior"]["reason"] == "equipment_restock"
    assert decision["command"]["cmd_type"] == "pickup_equipment"
    assert decision["command"]["target_x"] == 102
    assert decision["command"]["target_y"] == 100


def test_collect_takes_visible_equipment_at_critical_fuel() -> None:
    """Visible equipment is still grabbed at critical fuel (equipment ranks first)."""
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
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["dual_shots"]["count"] = 3
    inventory["homing_shots"]["count"] = 3
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    assert decision["behavior"]["reason"] == "equipment_restock"
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
        fuel=150,
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
            "mode": "COLLECT",
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

    decision = decide_collect_mode(ctx)

    assert decision["updated_ai_state"]["resource_target_kind"] == ""
