"""Tests for the durable fuel recovery owner."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.collect_mode import (
    _continue_or_release_fuel_lock,
    _pickup_not_worth_walk,
    _select_and_pickup_fuel,
    decide_collect_mode,
    select_fuel_target,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.state.types import SelfStateDict, make_container_state
from tests.bot.ai._support import (
    make_inventory,
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "equipment_restock"
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

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] in ("fuel_locked", "fuel_collect")
    assert decision["behavior"]["reason_context"]["volume"] == 900
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

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] in ("fuel_locked", "fuel_collect")
    assert decision["behavior"]["reason_context"]["volume"] == 700
    assert decision["behavior"]["target_x"] == 105
    assert decision["behavior"]["target_y"] == 105


def test_collect_mode_uses_radar_when_viewport_needs_authoritative_scan() -> None:
    """The durable owner senses before repositioning in an unscanned viewport."""
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["behavior"]["reason_kind"] == "scan_on_landing"
    assert decision["command"]["cmd_type"] == "radar"


def test_collect_mode_uses_regular_radar_when_extra_charges_are_empty() -> None:
    """Fuel recovery still scans with free radar when extras are depleted."""
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

    with pytest.raises(SessionExitError, match="COLLECT owner produced no decision"):
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["command"]["cmd_type"] == "move"
    assert decision["behavior"]["reason_kind"] == "forage_sweep"
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

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "equipment_restock"
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

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "equipment_restock"
    assert decision["behavior"]["target_x"] == 103


def test_locked_fuel_holds_when_water_locked() -> None:
    """A water-boxed fuel plan survives the tick it cannot execute.

    Committed-intent law ([[committed-intent]]): transient
    inexecutability holds the plan — the tick goes to the rest of
    the cascade, and only a genuine release gate (superior
    candidate, validity, the move-failed mark) drops it.
    """
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

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] != "fuel_locked"
    assert decision["updated_ai_state"]["resource_target_kind"] == "fuel"


def test_select_fuel_returns_none_at_rank_derived_capacity() -> None:
    """``_select_and_pickup_fuel`` refuses to dispatch at capacity.

    Sergeant (rank 3) has fuel capacity 1300 per
    :func:`tankpit_bot.physics.capacity.fuel_capacity`. A full tank
    at exactly 1300 must skip fuel selection so the cascade falls
    through instead of dispatching a wasted ``pickup_fuel`` that the
    server rejects with ``0x52`` code-5.
    """

    base_world, base_self = make_world(
        fuel=1300,
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
    self_state = SelfStateDict(**{**base_self, "rank": 3})
    world = base_world
    world["self_state"] = self_state
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

    decision = _select_and_pickup_fuel(ctx, ctx.base)

    assert decision is None


def test_locked_fuel_released_at_rank_derived_capacity() -> None:
    """A held fuel lock is dropped when the tank hits ``fuel_capacity(rank)``.

    Regression guard for the live-run 2026-07-06 tank-full pickup loop:
    the lock-continuation path had no capacity gate, so a held fuel
    lock kept re-dispatching ``pickup_fuel`` at capacity, each dispatch
    draws wire ``0x52`` code-5 ``Tank full``, and the lock survives to
    next tick. With rank-derived capacity, the lock is released and
    ``resource_target_kind`` cleared before the cascade can produce a
    fresh pickup command.
    """

    base_world, base_self = make_world(
        fuel=1600,
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
    self_state = SelfStateDict(**{**base_self, "rank": 6})
    world = base_world
    world["self_state"] = self_state
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
    locked_target = world["containers"]["105,105"]

    decision, updated_state = _continue_or_release_fuel_lock(
        ctx,
        ctx.base,
        locked_target,
    )

    assert decision is None
    assert updated_state["resource_target_kind"] == ""
    assert updated_state["resource_target_x"] == 0
    assert updated_state["resource_target_y"] == 0


def test_pickup_refused_when_clamped_gain_not_worth_the_walk() -> None:
    """A distant cap-clamped sliver is refused: the 2026-07-06 waste class.

    Private (rank 1, cap 1100) at fuel 1054: headroom 46, so a
    386-volume container transfers only 46. At a 2-tile walk the
    threshold is ``25 * 2 = 50 > 46`` -- not worth it, refuse.
    """

    base_world, base_self = make_world(fuel=1054)
    self_state = SelfStateDict(**{**base_self, "rank": 1})
    world = base_world
    world["self_state"] = self_state
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")
    container = make_container_state(
        x=102,
        y=100,
        is_fuel=True,
        volume=386,
        timestamp_ms=100000,
        failed_pickups=0,
    )

    assert _pickup_not_worth_walk(ctx, container) is True


def test_adjacent_clamped_sliver_is_worth_taking() -> None:
    """The same sliver one tile away IS taken: 46 >= 25 * 1.

    Under the old binary gate this exact geometry (the 2026-07-06
    canonical shape) was refused; with code=5 handled cleanly, +46
    fuel for one 2-second walk tile is the same rate a good dot hop
    pays, so the pickup is worth it.
    """

    base_world, base_self = make_world(fuel=1054)
    self_state = SelfStateDict(**{**base_self, "rank": 1})
    world = base_world
    world["self_state"] = self_state
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")
    container = make_container_state(
        x=101,
        y=100,
        is_fuel=True,
        volume=386,
        timestamp_ms=100000,
        failed_pickups=0,
    )

    assert _pickup_not_worth_walk(ctx, container) is False


def test_big_clamped_container_is_worth_a_long_walk() -> None:
    """Fuel 600 vs a 1000-volume container: 500 effective gain wins.

    The falsifying case for the old binary gate (2026-07-19): because
    ``volume >= headroom``, the old formula refused this pickup at ANY
    walk distance -- walking past half a tank of fuel one tile away.
    The rate predicate scores the actual 500-fuel transfer: worth up
    to a 20-tile walk at 25/tile, so a 12-tile walk clears easily.
    """

    base_world, base_self = make_world(fuel=600)
    self_state = SelfStateDict(**{**base_self, "rank": 1})
    world = base_world
    world["self_state"] = self_state
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")
    container = make_container_state(
        x=106,
        y=106,
        is_fuel=True,
        volume=1000,
        timestamp_ms=100000,
        failed_pickups=0,
    )

    assert _pickup_not_worth_walk(ctx, container) is False


def test_unclamped_pickup_is_worth_it_at_the_exact_rate_boundary() -> None:
    """Gain exactly equal to the walk threshold is taken, not refused.

    Fuel 500 (headroom 600), 100-volume container 4 tiles away:
    effective gain 100 == ``25 * 4`` -- the predicate refuses only
    strictly-below-rate pickups, so the boundary case dispatches.
    """

    base_world, base_self = make_world(fuel=500)
    self_state = SelfStateDict(**{**base_self, "rank": 1})
    world = base_world
    world["self_state"] = self_state
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")
    container = make_container_state(
        x=102,
        y=102,
        is_fuel=True,
        volume=100,
        timestamp_ms=100000,
        failed_pickups=0,
    )

    assert _pickup_not_worth_walk(ctx, container) is False


def test_critical_fuel_takes_any_reachable_sliver() -> None:
    """Below the fuel-low break the worth-the-walk rule is suspended.

    Run bot-20260728-090813 exited ``out_of_fuel`` at fuel 98 with a
    pickable 39-fuel container two tiles away, refused by the rate
    predicate as "not worth 2-tile walk". The predicate is efficiency
    logic for a healthy tank; at critical fuel the alternative to the
    walk is ending the session, so any reachable fuel dispatches.
    """

    base_world, base_self = make_world(
        fuel=98,
        scanned=True,
        containers={
            "102,100": make_container_state(
                x=102,
                y=100,
                is_fuel=True,
                volume=39,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    self_state = SelfStateDict(**{**base_self, "rank": 1})
    world = base_world
    world["self_state"] = self_state
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

    decision = _select_and_pickup_fuel(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected critical-fuel pickup decision")
    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["behavior"]["target_x"] == 102
    assert decision["behavior"]["reason_context"]["volume"] == 39


def test_select_and_pickup_fuel_refuses_when_projected_pickup_overflows() -> None:
    """``_select_and_pickup_fuel`` returns None on a not-worth-it sliver.

    Wire the 2026-07-06 waste class end-to-end: private at fuel 1054
    (headroom 46) with a single visible 386-volume fuel container 3
    tiles east -- effective gain 46 against a ``25 * 3 = 75``
    threshold. The at-cap gate passes (fuel below cap), the fuel
    target is selected successfully, but ``_pickup_not_worth_walk``
    fires and the planner returns None instead of dispatching. The
    container is left untouched -- not blacklisted -- so a later
    tick with more headroom (or from an adjacent tile) can still
    consume it.
    """

    base_world, base_self = make_world(
        fuel=1054,
        scanned=True,
        containers={
            "103,100": make_container_state(
                x=103,
                y=100,
                is_fuel=True,
                volume=386,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    self_state = SelfStateDict(**{**base_self, "rank": 1})
    world = base_world
    world["self_state"] = self_state
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

    decision = _select_and_pickup_fuel(ctx, ctx.base)

    assert decision is None
    assert world["containers"]["103,100"]["failed_pickups"] == 0
