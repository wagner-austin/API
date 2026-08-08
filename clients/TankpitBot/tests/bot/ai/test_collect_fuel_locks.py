"""Held fuel locks: continuation, release rules, capacity, water-locked holds."""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_locks import continue_or_release_fuel_lock
from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.collect_pickups import (
    select_and_pickup_fuel,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.ferry import FerryAwareTerrain
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import SelfStateDict, make_container_state, make_mine_state
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
    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

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
    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

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
    candidate, validity, the move-failed mark, unservability) drops
    it. The fixture keeps a fresh ferry ON the container's pond so
    the target stays SERVABLE (the ride lane exists) and the hold is
    genuinely transient.
    """
    ws = WorldService()
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
    from tankpit_bot.state.types import make_terrain_tile
    from tankpit_bot.types.constants import TERRAIN_FERRY

    world["terrain"]["121,101"] = make_terrain_tile(121, 101, TERRAIN_FERRY, observed_ms=100000)
    terrain_data[(121, 101)] = "W"
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "", ws=ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] != "fuel_locked"
    assert decision["updated_ai_state"]["resource_target_kind"] == "fuel"


def test_inexecutable_lock_without_terrain_holds() -> None:
    """No terrain map means no unservability verdict: the lock holds.

    Without the static map the pond and landing questions cannot be
    answered, so the structural release must not guess — transient
    hold, per the committed-intent law. The locked target sits
    outside the visible viewport, so the pickup path yields no
    command this tick.
    """
    ws = WorldService()
    world, self_state = make_world(
        self_x=100,
        self_y=100,
        fuel=400,
        scanned=True,
        containers={
            "130,100": make_container_state(
                x=130,
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
            "resource_target_kind": "fuel",
            "resource_target_x": 130,
            "resource_target_y": 100,
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "", ws=ws)

    decision, held_state = continue_or_release_fuel_lock(
        ctx, ai_state, world["containers"]["130,100"]
    )

    assert decision is None
    assert held_state["resource_target_kind"] == "fuel"


def test_unservable_water_locked_fuel_releases_the_lock() -> None:
    """No landing, no ferry on the pond: the lock releases as unservable.

    Run bot-20260804-234008 (2026-08-05 00:04) held exactly this lock
    for 11 minutes: a water-boxed container with the only known ferry
    floating on a DIFFERENT water body. No lane -- walk, hop, or ride
    -- could ever serve it, and no move-failed mark could ever arrive
    because nothing was ever dispatched. The enumerated release law
    now carries the structural verdict the selectors already compute.
    """
    ws = WorldService()
    terrain_data: dict[tuple[int, int], str] = {
        (120, 100): "W",
        (121, 100): "W",
        (119, 100): "W",
        (120, 101): "W",
        (120, 99): "W",
    }
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
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, terrain, "", ws=ws)

    decision, released_state = continue_or_release_fuel_lock(
        ctx, ai_state, world["containers"]["120,100"]
    )

    assert decision is None
    assert released_state["resource_target_kind"] == ""


def test_mine_denied_locked_fuel_releases_when_no_shot_exists() -> None:
    """Mined-only service tiles with no shootable mine: unservable.

    The bot-20260805-173034 class with the clearance arm closed: the
    sole ground service tile carries a hostile mine (teleports
    displace, walks refuse), but the mine sits outside the visible
    viewport so no clearance shot can be aimed, and no ferry floats
    on the pond. Nothing in the cascade can ever serve it — release.
    """
    ws = WorldService()
    terrain_data: dict[tuple[int, int], str] = {
        (130, 100): "W",
        (129, 100): "W",
        (130, 101): "W",
        (130, 99): "W",
    }
    terrain = FerryAwareTerrain(
        InMemoryTerrainMap(terrain_data=terrain_data),
        {},
        riding=False,
        hostile_mine_keys=frozenset({"131,100"}),
        occupied_tank_keys=frozenset(),
    )
    world, self_state = make_world(
        self_x=100,
        self_y=100,
        fuel=150,
        scanned=True,
        containers={
            "130,100": make_container_state(
                x=130,
                y=100,
                is_fuel=True,
                volume=500,
                timestamp_ms=100000,
            )
        },
    )
    world["mines"]["131,100"] = make_mine_state(x=131, y=100, mine_type=0, tank_id=-1, team=2)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "",
            "mode_started_ms": 90000,
            "resource_target_kind": "fuel",
            "resource_target_x": 130,
            "resource_target_y": 100,
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, terrain, "", ws=ws)

    decision, released_state = continue_or_release_fuel_lock(
        ctx, ai_state, world["containers"]["130,100"]
    )

    assert decision is None
    assert released_state["resource_target_kind"] == ""


def test_mine_denied_locked_fuel_holds_while_the_clearance_shot_exists() -> None:
    """A shootable service mine keeps the lock alive: servable via the free single.

    Same pocket but inside the viewport with a clear shot line: the
    clearance step runs before the hop lanes, so the verdict must
    HOLD — the shot reopens the landing next tick and the lock then
    serves normally. Releasing here would re-create the session-4
    churn one layer up.
    """
    ws = WorldService()
    terrain_data: dict[tuple[int, int], str] = {
        (104, 100): "W",
        (103, 100): "W",
        (104, 101): "W",
        (104, 99): "W",
    }
    terrain = FerryAwareTerrain(
        InMemoryTerrainMap(terrain_data=terrain_data),
        {},
        riding=False,
        hostile_mine_keys=frozenset({"105,100"}),
        occupied_tank_keys=frozenset(),
    )
    world, self_state = make_world(
        self_x=100,
        self_y=100,
        fuel=150,
        scanned=True,
        containers={
            "104,100": make_container_state(
                x=104,
                y=100,
                is_fuel=True,
                volume=500,
                timestamp_ms=100000,
            )
        },
    )
    world["mines"]["105,100"] = make_mine_state(x=105, y=100, mine_type=0, tank_id=-1, team=2)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "",
            "mode_started_ms": 90000,
            "resource_target_kind": "fuel",
            "resource_target_x": 104,
            "resource_target_y": 100,
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, terrain, "", ws=ws)

    _decision, held_state = continue_or_release_fuel_lock(
        ctx, ai_state, world["containers"]["104,100"]
    )

    assert held_state["resource_target_kind"] == "fuel"
    assert held_state["resource_target_x"] == 104
    assert held_state["resource_target_y"] == 100


def test_out_of_window_locked_fuel_holds_so_the_hop_can_fire() -> None:
    """A lock outside the command window HOLDS -- it must not decide.

    Run bot-20260805-075502 07:57 proved the law by breaking it: an
    "approach the window edge" leg inserted here returned a decision
    every tick, short-circuited the cascade, and starved the
    already-planned equipment-hop teleport into a one-tile-walk
    treadmill (autoscroll OFF walking can never shift the window,
    [[viewport-shift-protocol]]). The hold returns no decision so the
    cascade falls through to the hop lane, whose teleport recenters
    the window on the target.
    """
    ws = WorldService()
    world, self_state = make_world(
        self_x=100,
        self_y=100,
        fuel=400,
        scanned=True,
        containers={
            "130,100": make_container_state(
                x=130,
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
            "resource_target_x": 130,
            "resource_target_y": 100,
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        "",
        ws=ws,
    )

    decision, held_state = continue_or_release_fuel_lock(
        ctx, ai_state, world["containers"]["130,100"]
    )

    assert decision is None
    assert held_state["resource_target_kind"] == "fuel"
    assert held_state["resource_target_x"] == 130


def test_select_fuel_returns_none_at_rank_derived_capacity() -> None:
    """``select_and_pickup_fuel`` refuses to dispatch at capacity.

    Sergeant (rank 3) has fuel capacity 1300 per
    :func:`tankpit_bot.physics.capacity.fuel_capacity`. A full tank
    at exactly 1300 must skip fuel selection so the cascade falls
    through instead of dispatching a wasted ``pickup_fuel`` that the
    server rejects with ``0x52`` code-5.
    """

    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

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

    ws = WorldService()
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)
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
