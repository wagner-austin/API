"""Cascade-level equipment collection behavior of the COLLECT owner."""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.physics.capacity import inventory_capacity
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import WorldStateDict, make_container_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def test_full_inventory_skips_equipment_pickup() -> None:
    """At all-slots-full, visible equipment is not dispatched.

    User mechanic (2026-07-18): containers fill whatever is empty and
    the server rejects with code 7 only at all-slots-full -- a pickup
    at full inventory is a guaranteed wasted tick (8 of them in the
    2026-07-18 5-minute run before this gate).
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
    rank_cap = inventory_capacity(self_state["rank"])
    inventory = make_inventory(dual_count=rank_cap, default_count=rank_cap)

    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)
    decision = decide_collect_mode(ctx)

    assert ctx.inventory["dual_shots"]["count"] == rank_cap
    if decision is not None:
        assert decision["command"]["cmd_type"] != "pickup_equipment"


def test_collect_mode_forages_radar_when_search_hop_is_unaffordable() -> None:
    """The durable owner forages built-in radar when no search hop can be afforded.

    Regression guard for live run 20260610-000x: the owner used to raise
    here, killing the bot process mid-game. An unaffordable hop with no
    extra radar must degrade to the free built-in radar forage, never an
    exception. The viewport has unscanned ground so the forager fires
    the free radar instead of falling through to the unaffordable hop.
    """
    ws = WorldService()
    world, self_state = make_world(fuel=800, scanned=False)
    base_state = make_scanned_ai_state(landing_scan_viewport="")
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
            },
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["behavior"]["reason_kind"] == "scan_on_landing"
    assert decision["command"]["cmd_type"] == "radar"


def test_collect_mode_forages_radar_when_fully_boxed_in() -> None:
    """A fully boxed-in owner forages built-in radar instead of crashing.

    Every viewport tile is water, radar is exhausted, and at fuel=140
    neither the search hop nor any exploration teleport is affordable --
    the terminal action must be the free built-in radar forage so the
    process keeps running. Viewport not yet scanned so the forager
    has work to do.
    """
    ws = WorldService()
    world, self_state = make_world(fuel=140, scanned=False)
    base_state = make_scanned_ai_state(landing_scan_viewport="")
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
            },
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 0
    terrain_data: dict[tuple[int, int], str] = {}
    for x in range(92, 108):
        for y in range(92, 108):
            terrain_data[(x, y)] = "W"
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["behavior"]["reason_kind"] == "scan_on_landing"
    assert decision["command"]["cmd_type"] == "radar"


def test_collect_mode_raises_when_genuinely_boxed_in() -> None:
    """Boxed-in recovery raises instead of spamming map_open.

    Both walking-to-edge and the always-on map_intel terminal were
    removed 2026-06-22 because they wasted fuel without changing the
    bot's state in any productive way. When the forager is
    exhausted AND no teleport hop is affordable AND no known
    equipment exists, the bot has nothing legal to do; raising
    surfaces the stuck state loudly instead of silently looping.
    """
    import pytest

    # Fuel below the short-hop cost (8 tiles * 6 = 48). The
    # ``hunt_min_fuel`` reserve was dropped 2026-06-24, so genuine
    # stranding requires fuel below the raw short-hop cost.
    ws = WorldService()
    # The recent open was ANSWERED -- without the ingestion stamp
    # the exit correctly defers behind the in-flight map answer
    # (the islet fix, 2026-08-26).
    ws.map_data_ingested_ms = 96500
    world, self_state = make_world(fuel=30, scanned=True)
    # "Genuinely boxed in" must also mean no frontier: the 2026-08-14
    # lawnmower continuation walks toward any unscanned adjacent band,
    # which genuinely rescues this state -- so the boxed-in pin needs
    # the surroundings covered wall to wall.
    padded = dict(world["scanned_tiles"])
    for x in range(84, 116):
        for y in range(84, 116):
            padded[f"{x},{y}"] = 100000
    world = WorldStateDict(**{**world, "scanned_tiles": padded})
    base_state = make_scanned_ai_state()
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
            },
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            # Recent map open: the dot atlas is empty and a re-open
            # inside the cooldown teaches nothing, so the hop declines.
            "last_map_open_ms": 96000,
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    with pytest.raises(SessionExitError, match="COLLECT owner produced no decision"):
        decide_collect_mode(ctx)


def test_collect_mode_raises_when_fully_boxed_in() -> None:
    """A fully boxed-in owner raises so the stuck state is loud.

    Every viewport tile is water, the forage map is swept, radar is
    exhausted, and the search hop is unaffordable. The bot has no
    productive action; the silent map_intel fallback was deleted
    2026-06-22 in favour of a loud raise so the wedged state can't
    be missed.
    """
    import pytest

    # Fuel below the short-hop cost so no teleport is affordable.
    ws = WorldService()
    # The recent open was ANSWERED -- without the ingestion stamp
    # the exit correctly defers behind the in-flight map answer
    # (the islet fix, 2026-08-26).
    ws.map_data_ingested_ms = 96500
    world, self_state = make_world(fuel=30, scanned=True)
    # "Genuinely boxed in" must also mean no frontier: the 2026-08-14
    # lawnmower continuation walks toward any unscanned adjacent band,
    # which genuinely rescues this state -- so the boxed-in pin needs
    # the surroundings covered wall to wall.
    padded = dict(world["scanned_tiles"])
    for x in range(84, 116):
        for y in range(84, 116):
            padded[f"{x},{y}"] = 100000
    world = WorldStateDict(**{**world, "scanned_tiles": padded})
    base_state = make_scanned_ai_state()
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
            },
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            # Recent map open: the dot atlas is empty and a re-open
            # inside the cooldown teaches nothing, so the hop declines.
            "last_map_open_ms": 96000,
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 0
    terrain_data: dict[tuple[int, int], str] = {}
    for x in range(92, 108):
        for y in range(92, 108):
            terrain_data[(x, y)] = "W"
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "", ws=ws)

    with pytest.raises(SessionExitError, match="COLLECT owner produced no decision"):
        decide_collect_mode(ctx)


def test_collect_mode_picks_equipment_before_adjacent_fuel() -> None:
    """COLLECT picks visible equipment first, even when fuel sits adjacent.

    Under the unified cascade equipment ranks ahead of fuel: the
    gameplay loop drains all in-viewport equipment first, then the
    fuel-pickup step considers what remains. The old fuel-mode rule
    that opportunistically grabbed equipment, and the equipment-mode
    rule that opportunistically grabbed adjacent fuel, are collapsed
    into this single ordering.
    """
    ws = WorldService()
    world, self_state = make_world(
        fuel=800,
        scanned=True,
        containers={
            "100,101": make_container_state(
                x=100,
                y=101,
                is_fuel=True,
                volume=500,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
            "106,106": make_container_state(
                x=106,
                y=106,
                is_fuel=False,
                volume=0,
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
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "equipment_restock"
    assert decision["command"]["cmd_type"] == "pickup_equipment"
    assert decision["behavior"]["target_x"] == 106
    assert decision["behavior"]["target_y"] == 106


def test_collect_mode_walks_to_biggest_viewport_fuel_when_no_equipment() -> None:
    """No equipment in viewport + a fuel container exists -> walk to it before teleporting.

    User's hand-played loop (2026-06-23): teleport into a fresh
    viewport, scan, pick up all equipment, then optionally grab the
    biggest fuel container before hopping to the next clean
    viewport. The bot previously ignored non-adjacent fuel during
    equipment recovery and immediately teleport-searched, leaving a
    pickup-eligible fuel container behind every viewport. Live run
    2026-06-23 tick 21: F at (167,251), bot at (165,254), bot
    ignored it and hopped away. This test pins that the bot now
    walks to the in-viewport fuel before bailing.
    """
    # Fuel + volume chosen so the projected pickup fits under cap:
    # corporal cap is 1200, fuel 800, walk 10 tiles, volume 300 -->
    # 800 + 10 + min(300, 400) = 1110 <= 1200. Overflow-refusal is
    # covered by ``pickup_not_worth_walk`` tests in test_collect_mode_fuel.py.
    ws = WorldService()
    world, self_state = make_world(
        fuel=800,
        scanned=True,
        containers={
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=300,
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
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["behavior"]["reason_kind"] in ("fuel_locked", "fuel_collect")
    assert decision["behavior"]["reason_context"]["volume"] == 300
    assert decision["behavior"]["target_x"] == 105
    assert decision["behavior"]["target_y"] == 105


def test_collect_mode_skips_opportunistic_fuel_at_rank_capacity() -> None:
    """When fuel is at ``fuel_capacity(rank)``, the fuel pickup is skipped.

    Picking up at capacity wastes the action (wire ``0x52`` code-5
    ``Tank full``), so the opportunistic-viewport-fuel branch must
    defer. Capacity here is rank-derived
    (:func:`tankpit_bot.physics.capacity.fuel_capacity`), not a
    learned watermark: at corporal (``rank=2``) capacity is 1200 and
    the tank is at exactly 1200, so ``select_and_pickup_fuel``
    returns ``None`` and the cascade falls through to the no-equipment
    search-hop path.
    """
    ws = WorldService()
    world, self_state = make_world(
        fuel=1200,
        scanned=True,
        containers={
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=900,
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
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] != "fuel_collect"


def test_collect_mode_falls_through_when_fuel_walk_unreachable() -> None:
    """Fuel in viewport but walk_or_teleport blocked -> fall through, no fuel pickup.

    Covers the ``fuel_command is not None`` false branch. The fuel
    container's coords are marked as a previously-failed move, so
    ``walk_or_teleport`` returns ``None`` and the bot skips the
    opportunistic fuel pickup, falling through to the no-equipment
    search-hop path.
    """

    ws = WorldService()
    world, self_state = make_world(
        fuel=800,
        scanned=True,
        containers={
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=900,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    ws.mark_move_target_failed(105, 105, 99000)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] != "fuel_collect"


def test_collect_mode_releases_lock_for_markedly_closer_equipment() -> None:
    """A locked far container yields to markedly closer equipment.

    Mirrors the fuel-mode rule; regression guard for live run
    20260610-011x lock stickiness.
    """
    ws = WorldService()
    world, self_state = make_world(
        fuel=800,
        scanned=True,
        containers={
            "106,106": make_container_state(
                x=106,
                y=106,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
            "160,100": make_container_state(
                x=160,
                y=100,
                is_fuel=False,
                volume=0,
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
            "resource_target_kind": "equipment",
            "resource_target_x": 160,
            "resource_target_y": 100,
        }
    )
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["target_x"] == 106
    assert decision["behavior"]["target_y"] == 106
    assert decision["updated_ai_state"]["resource_target_x"] == 106


def test_collect_mode_keeps_lock_against_marginally_closer_equipment() -> None:
    """A candidate inside the anti-churn threshold does not break the lock."""
    ws = WorldService()
    world, self_state = make_world(
        fuel=800,
        scanned=True,
        containers={
            "104,104": make_container_state(
                x=104,
                y=104,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=False,
                volume=0,
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
            "resource_target_kind": "equipment",
            "resource_target_x": 105,
            "resource_target_y": 105,
        }
    )
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "equipment_locked"
    assert decision["behavior"]["target_x"] == 105
    assert decision["behavior"]["target_y"] == 105
