"""Cascade-level fuel collection behavior of the COLLECT owner."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.collect_pickups import (
    _first_walkworthy_fuel,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def test_collect_mode_continues_locked_fuel_target() -> None:
    """The durable owner keeps an executable locked fuel target."""
    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["behavior"]["reason_kind"] in ("fuel_locked", "fuel_collect")
    assert decision["behavior"]["reason_context"]["volume"] == 700
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
    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
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
    ws = WorldService()
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
    # dual below capacity: the all-slots-full pickup gate must permit
    # (user mechanic 2026-07-18 -- at full inventory the pickup would
    # be a guaranteed code-7 rejection and is skipped).
    inventory = make_inventory(dual_count=20)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "equipment_restock"
    assert decision["command"]["cmd_type"] == "pickup_equipment"
    assert decision["behavior"]["target_x"] == 101
    assert decision["behavior"]["target_y"] == 100


def test_collect_mode_uses_radar_when_viewport_needs_authoritative_scan() -> None:
    """The durable owner senses before repositioning in an unscanned viewport."""
    ws = WorldService()
    world, self_state = make_world(fuel=150, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(landing_scan_viewport=""),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["behavior"]["reason_kind"] == "scan_on_landing"
    assert decision["command"]["cmd_type"] == "radar"


def test_collect_mode_uses_regular_radar_when_extra_charges_are_empty() -> None:
    """Fuel recovery still scans with free radar when extras are depleted."""
    ws = WorldService()
    world, self_state = make_world(fuel=150, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(landing_scan_viewport=""),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 1
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "scan_on_landing"
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
    ws = WorldService()
    # The recent open was ANSWERED -- without the ingestion stamp
    # the exit correctly defers behind the in-flight map answer
    # (the islet fix, 2026-08-26).
    ws.map_data_ingested_ms = 96500
    world, self_state = make_world(fuel=30, scanned=True)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            # Recent map open: the dot atlas is empty and a re-open
            # inside the cooldown teaches nothing, so the hop declines.
            "last_map_open_ms": 96000,
        }
    )
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 1
    terrain_data: dict[tuple[int, int], str] = {}
    for x in range(92, 108):
        for y in range(92, 108):
            terrain_data[(x, y)] = "W"
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "", ws=ws)

    with pytest.raises(SessionExitError, match="COLLECT owner produced no decision"):
        decide_collect_mode(ctx)


def test_select_fuel_target_returns_none_for_unreachable_off_viewport_target() -> None:
    """Out-of-viewport fuel with no walkable approach and no fuel returns None."""

    ws = WorldService()
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
    ws.mark_move_target_failed(103, 100, 99000)
    terrain = InMemoryTerrainMap(terrain_data={})
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        terrain,
        "",
        ws=ws,
    )

    assert _first_walkworthy_fuel(ctx) is None


def test_select_fuel_target_rejects_walk_unreachable_in_viewport() -> None:
    """Walk-unreachable in-viewport fuel is not selected (walkable-only).

    The server's long-press pickup walks the tank in a straight
    line; if water or rocks block that path it returns CANT_GO and
    one rejection flags the container ``failed_pickups`` for the
    whole session. The fix shipped 2026-06-24 is to never select
    containers the bot cannot walk to in the first place.
    """
    ws = WorldService()
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
        ws=ws,
    )

    assert _first_walkworthy_fuel(ctx) is None


def test_selects_low_volume_fuel_when_critically_low() -> None:
    """Critical fuel recovery accepts small visible fuel containers."""
    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["behavior"]["reason_kind"] in ("fuel_locked", "fuel_collect")
    assert decision["behavior"]["reason_context"]["volume"] == 57


def test_collect_mode_walks_when_no_extras_and_local_5x5_already_covered() -> None:
    """Fuel recovery walks instead of radaring when the next free radar would reveal nothing.

    Radar is always affordable (the wire never denies the action), but
    a free radar only marks the 5x5 around the tank. When extras are
    exhausted AND those 25 tiles are already covered, firing again
    would mark zero new tiles -- the tank has to walk first so a
    later free radar reaches new ground. Without this gate the bot
    loops radaring from the same spot forever (post-unconditional-
    radar regression caught in design 2026-06-26).
    """
    ws = WorldService()
    world, self_state = make_world(fuel=5, scanned=False)
    # Pre-mark the 5x5 around the tank (self at default (100,100)) so
    # the next free radar would reveal nothing more.
    world["scanned_tiles"] = {f"{x},{y}": 100000 for y in range(98, 103) for x in range(98, 103)}
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["command"]["cmd_type"] == "move"
    assert decision["behavior"]["reason_kind"] == "forage_sweep"
    assert decision["behavior"]["mode"] == "COLLECT"


def test_collect_takes_visible_equipment_before_search_hop() -> None:
    """When equipment is in the viewport, COLLECT grabs it before hopping."""
    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "equipment_restock"
    assert decision["command"]["cmd_type"] == "pickup_equipment"
    assert decision["command"]["target_x"] == 102
    assert decision["command"]["target_y"] == 100


def test_collect_takes_visible_equipment_at_critical_fuel() -> None:
    """Visible equipment is still grabbed at critical fuel (equipment ranks first)."""
    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "equipment_restock"
    assert decision["behavior"]["target_x"] == 103
