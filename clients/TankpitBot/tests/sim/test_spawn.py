"""The archive-mined replenishment law: fresh-position respawns."""

from __future__ import annotations

from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.spawn import (
    RESPAWN_INTERVAL_TICKS,
    SPAWN_VOLUME,
    _tile_occupied,
    find_open_tile,
    find_open_tile_near,
    respawn_containers,
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
    world["containers"].append(SimContainerDict(x=20, y=20, volume=300))
    world["containers"].append(SimContainerDict(x=30, y=30, volume=0))
    world["equipment"].append(SimEquipmentDict(x=40, y=40))
    return world


def test_below_target_spawns_one_fresh_dot_on_the_minute_beat() -> None:
    """A drained population respawns exactly one dot at a fresh tile."""
    world = _world()
    world["tick"] = RESPAWN_INTERVAL_TICKS
    respawn_containers(world, InMemoryTerrainMap(), fuel_target=2, equipment_target=1)
    stocked = [c for c in world["containers"] if c["volume"] > 0]
    assert len(stocked) == 2
    spawned = stocked[-1]
    assert spawned["volume"] == SPAWN_VOLUME
    assert (spawned["x"], spawned["y"]) != (30, 30)
    assert (spawned["x"], spawned["y"]) != (20, 20)


def test_at_target_population_spawns_nothing() -> None:
    """The 12-minute idle session at high population spawned zero."""
    world = _world()
    world["tick"] = RESPAWN_INTERVAL_TICKS
    respawn_containers(world, InMemoryTerrainMap(), fuel_target=1, equipment_target=1)
    assert len([c for c in world["containers"] if c["volume"] > 0]) == 1
    world["tick"] = RESPAWN_INTERVAL_TICKS + RESPAWN_INTERVAL_TICKS // 2
    respawn_containers(world, InMemoryTerrainMap(), fuel_target=1, equipment_target=1)
    assert len(world["equipment"]) == 1


def test_off_beat_ticks_spawn_nothing() -> None:
    """Spawns land only on the minute and half-minute beats."""
    world = _world()
    world["tick"] = RESPAWN_INTERVAL_TICKS + 1
    respawn_containers(world, InMemoryTerrainMap(), fuel_target=9, equipment_target=9)
    assert len(world["containers"]) == 2
    assert len(world["equipment"]) == 1


def test_equipment_respawns_on_the_offset_beat() -> None:
    """Equipment mirrors the fuel law on the half-minute beat."""
    world = _world()
    world["equipment"] = []
    world["tick"] = RESPAWN_INTERVAL_TICKS + RESPAWN_INTERVAL_TICKS // 2
    respawn_containers(world, InMemoryTerrainMap(), fuel_target=2, equipment_target=1)
    assert len(world["equipment"]) == 1


def test_spawns_avoid_occupied_and_impassable_tiles() -> None:
    """The deterministic scan skips tanks, mines, stock, and rock."""
    world = _world()
    world["mines"].append(SimMineDict(x=50, y=50, team=1))
    world["tick"] = RESPAWN_INTERVAL_TICKS
    respawn_containers(world, InMemoryTerrainMap(), fuel_target=2, equipment_target=1)
    spawned = [c for c in world["containers"] if c["volume"] > 0][-1]
    position = (spawned["x"], spawned["y"])
    assert position not in {(10, 10), (20, 20), (40, 40), (50, 50)}


def test_sealed_map_spawns_nothing() -> None:
    """A world with no open tile stays silent instead of guessing."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 0, 0, 1000)
    world["tick"] = RESPAWN_INTERVAL_TICKS
    sealed = InMemoryTerrainMap(terrain_data={(x, y): "#" for x in range(256) for y in range(256)})
    respawn_containers(world, sealed, fuel_target=5, equipment_target=5)
    world["tick"] = RESPAWN_INTERVAL_TICKS + RESPAWN_INTERVAL_TICKS // 2
    respawn_containers(world, sealed, fuel_target=5, equipment_target=5)
    assert world["containers"] == []
    assert world["equipment"] == []


def test_tile_occupancy_blocks_every_entity_kind() -> None:
    """Tanks, stocked containers, equipment, and mines all block."""
    world = _world()
    world["mines"].append(SimMineDict(x=50, y=50, team=1))
    assert _tile_occupied(world, 10, 10) is True
    assert _tile_occupied(world, 20, 20) is True
    assert _tile_occupied(world, 30, 30) is False
    assert _tile_occupied(world, 40, 40) is True
    assert _tile_occupied(world, 50, 50) is True
    assert _tile_occupied(world, 60, 60) is False


def test_global_scan_walks_past_an_occupied_first_probe() -> None:
    """A mine on the tick-derived first candidate forces the stride on."""
    world = _world()
    seed = (RESPAWN_INTERVAL_TICKS * 97) % 65536
    first = (seed % 256, seed // 256)
    world["mines"].append(SimMineDict(x=first[0], y=first[1], team=1))
    position = find_open_tile(world, InMemoryTerrainMap(), RESPAWN_INTERVAL_TICKS)
    if position is None:
        raise AssertionError("an open map must yield a spawn tile")
    assert position != first


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


def test_server_ticks_apply_the_law_toward_the_seeded_equilibrium() -> None:
    """A drained world refills across minute beats, then stops.

    The target is fixed at SERVER INIT from the seeded stock — later
    consumption creates deficit, it never lowers the equilibrium.
    """
    world = _world()
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    assert server._fuel_target == 1
    world["containers"][0]["volume"] = 0
    hold = ClientCommandDict(kind="map_open", command=108, x=0, y=0, target_id=0, slot=0)
    for _ in range(RESPAWN_INTERVAL_TICKS * 2):
        server.queue_command(9, hold)
        server.advance_tick()
    stocked = [c for c in world["containers"] if c["volume"] > 0]
    assert len(stocked) == 1
