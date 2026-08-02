"""Held fuel locks: continuation, release rules, capacity, water-locked holds."""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_locks import continue_or_release_fuel_lock
from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.collect_pickups import (
    select_and_pickup_fuel,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.state.types import SelfStateDict, make_container_state
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


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


def test_locked_fuel_holds_when_water_locked() -> None:
    """A water-boxed fuel plan survives the tick it cannot execute.

    Committed-intent law ([[committed-intent]]): transient
    inexecutability holds the plan â€” the tick goes to the rest of
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
    """``select_and_pickup_fuel`` refuses to dispatch at capacity.

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

    decision = select_and_pickup_fuel(ctx, ctx.base)

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

    decision, updated_state = continue_or_release_fuel_lock(
        ctx,
        ctx.base,
        locked_target,
    )

    assert decision is None
    assert updated_state["resource_target_kind"] == ""
    assert updated_state["resource_target_x"] == 0
    assert updated_state["resource_target_y"] == 0
