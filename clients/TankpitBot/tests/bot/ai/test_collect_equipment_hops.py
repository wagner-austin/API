"""The equipment hop: tracked-container teleports and the held-lock bar."""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_hops import hop_toward_equipment
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.ferry import FerryAwareTerrain
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def test_hop_toward_equipment_picks_nearest_of_multiple_external_candidates() -> None:
    """The equipment hop step picks the nearest external equipment container.

    Two equipment containers sit outside the current viewport
    (default 92-107 around bot at 100,100): one at (130,100) with
    teleport cost 180, one at (150,100) with teleport cost 300. Bot
    at fuel 1200 (corporal cap) and under-armed inventory forces the
    hop step to fire; both teleports leave the 650 engagement
    reserve behind so both are affordable candidates. The step ranks
    by teleport cost and picks (130,100) -- exercising the
    ``best_container is not None AND cost >= best_cost`` branch when
    the (150,100) candidate is considered and rejected as more
    expensive.
    """
    ws = WorldService()
    containers = {
        "130,100": make_container_state(
            x=130,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
        "150,100": make_container_state(
            x=150,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    inventory = make_inventory(default_count=15)
    terrain = InMemoryTerrainMap()
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        terrain,
        "",
        ws=ws,
    )

    decision = hop_toward_equipment(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected equipment-hop decision")
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 130
    assert decision["command"]["target_y"] == 100
    assert decision["behavior"]["reason_kind"] == "equipment_hop"


def test_hop_toward_equipment_declines_during_a_held_lock_above_break() -> None:
    """F21 fix (2026-07-31): no mid-fight hop while reserves clear the break bar.

    Same fixture as the nearest-candidate pin (under the entry cap at
    15 of 25, two tracked containers) but with a combat lock held:
    the hunt-ENTRY bar must not schedule travel during a fight -- the
    s7-3/4 receipts show an 85-tile round trip at duals 22/25 with
    Yuppler locked. Above the break thresholds the hop declines.
    """
    ws = WorldService()
    containers = {
        "130,100": make_container_state(
            x=130,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    inventory = make_inventory(default_count=15)
    terrain = InMemoryTerrainMap()
    ai_state = AIStateDict(**{**make_scanned_ai_state(), "combat_target_id": 50})
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "", ws=ws)

    assert hop_toward_equipment(ctx, ctx.base) is None


def test_hop_toward_equipment_fires_during_a_held_lock_at_weapon_break() -> None:
    """A genuine weapon break still tops up mid-fight (the resume floor)."""
    ws = WorldService()
    containers = {
        "130,100": make_container_state(
            x=130,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    inventory = make_inventory(default_count=3)
    terrain = InMemoryTerrainMap()
    ai_state = AIStateDict(**{**make_scanned_ai_state(), "combat_target_id": 50})
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "", ws=ws)

    decision = hop_toward_equipment(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected equipment-hop decision at weapon break")
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "equipment_hop"


def test_hop_toward_equipment_takes_in_viewport_walk_blocked_containers() -> None:
    """In-viewport equipment the walk step cannot reach is teleport fair game.

    The 2026-07-30 flag-4 fix ([[flag-triage-20260729]]): this step
    runs only after the walk-pickup step declined, so any tracked
    container left â€” including one INSIDE the current viewport whose
    walk path is terrain-blocked â€” is hopped to instead of being
    invisible to both steps (run bot-20260730-000038 ticks 121-126
    dot-hopped away from three identified containers and paid a
    return trip later).
    """
    ws = WorldService()
    containers = {
        "105,105": make_container_state(
            x=105,
            y=105,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    inventory = make_inventory(default_count=15)
    terrain = InMemoryTerrainMap()
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        terrain,
        "",
        ws=ws,
    )

    decision = hop_toward_equipment(ctx, ctx.base)

    if decision is None:
        raise AssertionError("walk-blocked in-viewport equipment must draw a teleport hop")
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 105
    assert decision["command"]["target_y"] == 105
    assert decision["behavior"]["reason_kind"] == "equipment_hop"


def test_hop_toward_equipment_skips_when_teleport_unaffordable() -> None:
    """The equipment hop step declines when every teleport leaves under-reserve.

    Engagement reserve is ``engagement_fuel_budget(450) +
    fuel_low_threshold(200) = 650``. A teleport from (100,100) to
    (200,100) costs 600; at fuel 1000 the post-teleport residual is
    400, which is below the 650 reserve. The step considers the
    candidate, rejects the affordability check, and returns None.
    """
    ws = WorldService()
    containers = {
        "200,100": make_container_state(
            x=200,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1000, containers=containers)
    inventory = make_inventory(default_count=15)
    terrain = InMemoryTerrainMap()
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        terrain,
        "",
        ws=ws,
    )

    decision = hop_toward_equipment(ctx, ctx.base)

    assert decision is None


def test_hop_toward_equipment_skips_when_landing_tile_impassable() -> None:
    """The equipment hop step skips containers with no legal landing tile.

    Container at (150,100) with the container tile and all four
    cardinal neighbors marked water: ``find_teleport_landing_tile``
    returns None for this container, the loop continues to the next
    candidate (of which there are none), and the step returns None.
    """
    ws = WorldService()
    containers = {
        "150,100": make_container_state(
            x=150,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    inventory = make_inventory(default_count=15)
    terrain_data: dict[tuple[int, int], str] = {
        (150, 100): "W",
        (149, 100): "W",
        (151, 100): "W",
        (150, 99): "W",
        (150, 101): "W",
    }
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        terrain,
        "",
        ws=ws,
    )

    decision = hop_toward_equipment(ctx, ctx.base)

    assert decision is None


def test_hop_toward_equipment_never_teleports_to_its_own_tile() -> None:
    """A landing equal to the current position is not travel.

    Flag s8-2 (run bot-20260730-025337, 03:00:00): the escape hop
    landed ON its target and the next derivation selected a fresh
    teleport to the tile the tank was standing on, deferring a
    map open for a zero-distance jump. Ground the tank already owns
    belongs to the pickup steps, so the sole own-tile candidate
    declines the hop entirely.
    """
    ws = WorldService()
    containers = {
        "100,100": make_container_state(
            x=100,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    inventory = make_inventory(default_count=15)
    terrain = InMemoryTerrainMap()
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        terrain,
        "",
        ws=ws,
    )

    assert hop_toward_equipment(ctx, ctx.base) is None


def test_hop_toward_equipment_skips_own_tile_for_a_real_candidate() -> None:
    """The zero-cost own-tile candidate never outbids actual travel.

    Cost ranking structurally favors the own-tile candidate (cost 0),
    so without the own-ground gate the (100,100) container would win
    over the external (130,100) one and produce the s8-2 self-teleport.
    """
    ws = WorldService()
    containers = {
        "100,100": make_container_state(
            x=100,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
        "130,100": make_container_state(
            x=130,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    inventory = make_inventory(default_count=15)
    terrain = InMemoryTerrainMap()
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        terrain,
        "",
        ws=ws,
    )

    decision = hop_toward_equipment(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected equipment-hop decision")
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 130
    assert decision["command"]["target_y"] == 100


def test_hop_toward_equipment_boards_a_ferry_for_water_locked_drop() -> None:
    """A fresh believed ferry serves a water-locked equipment drop.

    Radar-situation receipt (2026-07-30 session 7): all 8 tracked
    equipment containers were water drops with every neighbor
    impassable, so the radar-broke tank had no reachable restock. The
    equipment hop now mirrors the fuel larder's F5 ferry landing: the
    boarding tile becomes the teleport target and the held lock rides
    to the pickup.
    """
    from tankpit_bot.state.types import make_terrain_tile
    from tankpit_bot.types.constants import TERRAIN_FERRY

    ws = WorldService()
    containers = {
        "150,100": make_container_state(
            x=150,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    world["terrain"]["148,101"] = make_terrain_tile(148, 101, TERRAIN_FERRY, observed_ms=100000)
    inventory = make_inventory(default_count=15)
    terrain_data: dict[tuple[int, int], str] = {
        (150, 100): "W",
        (149, 100): "W",
        (151, 100): "W",
        (150, 99): "W",
        (150, 101): "W",
        # The boarding tile floats on the SAME pond as the drop — the
        # ride-exists gate (2026-08-05) requires water connectivity,
        # and this fixture's ferry originally sat on unconnected
        # ground by accident (the exact live-deadlock geometry).
        (149, 101): "W",
        (148, 101): "W",
    }
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        terrain,
        "",
        ws=ws,
    )

    decision = hop_toward_equipment(ctx, ctx.base)

    if decision is None:
        raise AssertionError("ferry-served equipment hop must produce a decision")
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 148
    assert decision["command"]["target_y"] == 101
    assert decision["behavior"]["reason_kind"] == "equipment_hop"


def test_hop_skips_a_mine_denied_nearest_and_takes_the_next_candidate() -> None:
    """The bot-20260805-173034 loop-killer: mined access is not a landing.

    The nearer container's only legal service tile carries a known
    hostile mine -- a teleport there displaces every time, so the hop
    must not aim at it. The selector skips it (no_landing) and takes
    the farther clean candidate instead of re-aiming forever.
    """
    from tankpit_bot.state.types import make_mine_state

    ws = WorldService()
    containers = {
        "130,100": make_container_state(
            x=130,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
        "150,100": make_container_state(
            x=150,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    # (130,100) sits on water; three cardinals are water, the sole
    # ground neighbor (131,100) carries a hostile mine.
    world["mines"]["131,100"] = make_mine_state(x=131, y=100, mine_type=0, tank_id=-1, team=2)
    terrain = FerryAwareTerrain(
        InMemoryTerrainMap(
            {
                (130, 100): InMemoryTerrainMap.WATER,
                (129, 100): InMemoryTerrainMap.WATER,
                (130, 99): InMemoryTerrainMap.WATER,
                (130, 101): InMemoryTerrainMap.WATER,
            }
        ),
        {},
        riding=False,
        hostile_mine_keys=frozenset({"131,100"}),
        occupied_tank_keys=frozenset(),
    )
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        terrain,
        "",
        ws=ws,
    )

    decision = hop_toward_equipment(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected the clean farther candidate to win the hop")
    command = decision["command"]
    assert command["cmd_type"] == "teleport"
    assert command["target_x"] == 150
    assert command["target_y"] == 100


def test_hop_toward_equipment_declines_when_terrain_unknown() -> None:
    """Unknown terrain declines the equipment hop even with a live candidate.

    The candidate list is deliberately non-empty and affordable, so the
    terrain guard is the ONLY thing that can produce None here. Against
    a world with no tracked equipment the ``not candidates`` branch
    returns None anyway, and that is how this guard survived mutation
    (2026-08-08). ``_equipment_hop_landing`` takes a caller-narrowed
    non-None terrain, so dropping the guard raises instead of declining.
    """
    ws = WorldService()
    containers = {
        "130,100": make_container_state(
            x=130,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(default_count=15),
        100000,
        None,
        "",
        ws=ws,
    )

    assert hop_toward_equipment(ctx, ctx.base) is None
