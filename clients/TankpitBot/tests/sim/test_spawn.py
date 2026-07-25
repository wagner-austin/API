"""Deterministic open-tile placement (seeding + reactivation).

The runtime respawn law this module once tested was FALSIFIED
2026-07-25 (every observed "spawn" was an exposure of a pre-existing
container); the sim spawns nothing at runtime and these pickers now
serve world seeding and bot reactivation only.
"""

from __future__ import annotations

from tankpit_bot.sim.spawn import (
    _tile_occupied,
    find_open_tile,
    find_open_tile_near,
)
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimEquipmentDict,
    SimMineDict,
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _world() -> SimWorldDict:
    """One tank at (10, 10) with one stocked container and one drained."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    world["containers"].append(SimContainerDict(x=20, y=20, volume=300, dotted=True))
    world["containers"].append(SimContainerDict(x=30, y=30, volume=0, dotted=True))
    world["equipment"].append(SimEquipmentDict(x=40, y=40))
    return world


def test_tile_occupancy_blocks_every_entity_kind() -> None:
    """Tanks, stocked containers, equipment, mines, and blocks all block."""
    from tankpit_bot.sim.world import SimBlockDict

    world = _world()
    world["mines"].append(SimMineDict(x=50, y=50, team=1))
    world["blocks"].append(SimBlockDict(x=55, y=55))
    assert _tile_occupied(world, 55, 55) is True
    assert _tile_occupied(world, 10, 10) is True
    assert _tile_occupied(world, 20, 20) is True
    assert _tile_occupied(world, 30, 30) is False
    assert _tile_occupied(world, 40, 40) is True
    assert _tile_occupied(world, 50, 50) is True
    assert _tile_occupied(world, 60, 60) is False


def test_global_scan_walks_past_an_occupied_first_probe() -> None:
    """A mine on the tick-derived first candidate forces the stride on."""
    world = _world()
    seed = (30 * 97) % 65536
    first = (seed % 256, seed // 256)
    world["mines"].append(SimMineDict(x=first[0], y=first[1], team=1))
    position = find_open_tile(world, InMemoryTerrainMap(), 30)
    if position is None:
        raise AssertionError("an open map must yield a placement tile")
    assert position != first


def test_sealed_map_yields_no_tile() -> None:
    """A world with no open tile returns None instead of guessing."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 0, 0, 1000)
    sealed = InMemoryTerrainMap(terrain_data={(x, y): "#" for x in range(256) for y in range(256)})
    assert find_open_tile(world, sealed, 30) is None
    assert find_open_tile_near(world, sealed, 2, 2, tick=1, min_radius=0, max_radius=3) is None


def test_ring_search_skips_bounds_and_occupied_rings() -> None:
    """Near a map corner with a mined inner ring, the band widens."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 2, 2, 1000)
    for dx in range(-6, 7):
        for dy in range(-6, 7):
            if max(abs(dx), abs(dy)) == 6 and 0 <= 2 + dx < 256 and 0 <= 2 + dy < 256:
                world["mines"].append(SimMineDict(x=2 + dx, y=2 + dy, team=1))
    position = find_open_tile_near(
        world, InMemoryTerrainMap(), 2, 2, tick=5, min_radius=6, max_radius=7
    )
    if position is None:
        raise AssertionError("the widened ring band must yield a tile")
    assert max(abs(position[0] - 2), abs(position[1] - 2)) == 7
