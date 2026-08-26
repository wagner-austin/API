"""Ferry-served equipment hops: boarding-tile landings and their loop-breaker.

Split from ``test_collect_equipment_hops`` at the 600-line ceiling.
The F5 boarding landing (radar-situation receipt, 2026-07-30) and the
islet-loop ride-dead guard (2026-08-26) live here together.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_hops import hop_toward_equipment
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


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


def test_hop_never_re_boards_from_beside_the_boarding_tile() -> None:
    """Standing beside the boarding tile, the same hop never re-derives.

    The islet loop's equipment mirror (arterial 2026-08-26): a
    boarding hop that lands and cannot ride re-derives itself forever
    because the hop SUCCEEDS every lap. Adjacency to the boarding
    tile is the ride-failed receipt; the candidate is dead until the
    tank moves away.
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
    world, self_state = make_world(self_x=148, self_y=102, fuel=1200, containers=containers)
    world["terrain"]["148,101"] = make_terrain_tile(148, 101, TERRAIN_FERRY, observed_ms=100000)
    inventory = make_inventory(default_count=15)
    terrain = InMemoryTerrainMap(
        terrain_data={
            (150, 100): "W",
            (149, 100): "W",
            (151, 100): "W",
            (150, 99): "W",
            (150, 101): "W",
            (149, 101): "W",
            (148, 101): "W",
        }
    )
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
