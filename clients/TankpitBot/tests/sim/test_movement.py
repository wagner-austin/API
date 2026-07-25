"""Law 2 — instant movement, billing, and arrival effects."""

from __future__ import annotations

from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.sim.movement import MINE_WALK_COST, process_move
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimMineDict,
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _world_with_tank(fuel: int = 1000) -> SimWorldDict:
    """One private-rank tank at (10, 10) on an empty world."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, fuel)
    return world


def test_move_relocates_bills_full_path_instantly() -> None:
    """Route, relocation, and billing all land in one call."""
    world = _world_with_tank()
    outcome = process_move(world, InMemoryTerrainMap(), 9, 13, 12)
    assert outcome["kind"] == "moved"
    assert outcome["path"] == "sseee"
    assert world["tanks"][9]["x"] == 13
    assert world["tanks"][9]["y"] == 12
    assert world["tanks"][9]["fuel"] == 995


def test_impassable_destination_is_cant_go() -> None:
    """Rock at the destination yields cant_go and no mutation."""
    world = _world_with_tank()
    terrain = InMemoryTerrainMap(terrain_data={(13, 12): "#"})
    outcome = process_move(world, terrain, 9, 13, 12)
    assert outcome["kind"] == "cant_go"
    assert world["tanks"][9]["fuel"] == 1000
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (10, 10)


def test_route_cost_above_fuel_is_insufficient() -> None:
    """A path longer than the tank's fuel is refused, not partial."""
    world = _world_with_tank(fuel=3)
    outcome = process_move(world, InMemoryTerrainMap(), 9, 15, 10)
    assert outcome["kind"] == "insufficient_fuel"
    assert world["tanks"][9]["fuel"] == 3


def test_arrival_pickup_respects_capacity() -> None:
    """The destination container drains only what the tank can hold."""
    world = _world_with_tank(fuel=fuel_capacity(1) - 50)
    world["containers"].append(SimContainerDict(x=11, y=10, volume=200, dotted=True))
    outcome = process_move(world, InMemoryTerrainMap(), 9, 11, 10)
    assert outcome["kind"] == "moved"
    assert world["tanks"][9]["fuel"] == fuel_capacity(1)
    assert world["containers"][0]["volume"] == 149
    assert outcome["pickups"] == [{"x": 11, "y": 10, "remaining_volume": 149}]


def test_empty_container_is_not_a_pickup() -> None:
    """A drained container at the destination produces no record."""
    world = _world_with_tank()
    world["containers"].append(SimContainerDict(x=11, y=10, volume=0, dotted=True))
    outcome = process_move(world, InMemoryTerrainMap(), 9, 11, 10)
    assert outcome["kind"] == "moved"
    assert outcome["pickups"] == []


def test_walking_into_enemy_mine_detonates_and_bills_45() -> None:
    """The destination's enemy mine costs 45 and disappears."""
    world = _world_with_tank()
    world["mines"].append(SimMineDict(x=11, y=10, team=2))
    outcome = process_move(world, InMemoryTerrainMap(), 9, 11, 10)
    assert outcome["kind"] == "moved"
    assert outcome["mine_positions"] == [(11, 10)]
    assert world["mines"] == []
    assert world["tanks"][9]["fuel"] == 1000 - 1 - MINE_WALK_COST


def test_enemy_mine_interior_forces_detour_own_mine_does_not() -> None:
    """Routing avoids enemy mines but walks own-color mines."""
    world = _world_with_tank()
    world["mines"].append(SimMineDict(x=10, y=11, team=2))
    detoured = process_move(world, InMemoryTerrainMap(), 9, 10, 12)
    assert detoured["kind"] == "moved"
    assert len(detoured["path"]) == 4
    world["mines"][0] = SimMineDict(x=10, y=11, team=0)
    world["tanks"][9]["x"], world["tanks"][9]["y"] = 10, 10
    direct = process_move(world, InMemoryTerrainMap(), 9, 10, 12)
    assert direct["path"] == "ss"


def test_other_tank_blocks_interior_and_destination() -> None:
    """Living tanks block routing; dead tanks do not."""
    world = _world_with_tank()
    world["tanks"][11] = make_sim_tank(11, 1, 1, 11, 10, 500)
    occupied = process_move(world, InMemoryTerrainMap(), 9, 11, 10)
    assert occupied["kind"] == "cant_go"
    detoured = process_move(world, InMemoryTerrainMap(), 9, 12, 10)
    assert detoured["kind"] == "moved"
    assert len(detoured["path"]) == 4
    world["tanks"][9]["x"], world["tanks"][9]["y"] = 10, 10
    world["tanks"][11]["alive"] = False
    through = process_move(world, InMemoryTerrainMap(), 9, 12, 10)
    assert through["path"] == "ee"
