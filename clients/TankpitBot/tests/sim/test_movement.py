"""Law 2 — instant movement, billing, and arrival effects."""

from __future__ import annotations

from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.sim.actions import process_radar
from tankpit_bot.sim.combat import SLOT_RADAR
from tankpit_bot.sim.movement import MINE_WALK_COST, process_move
from tankpit_bot.sim.world import (
    SimBlockDict,
    SimContainerDict,
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
    place_mine,
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


def test_walk_debit_clamps_to_remaining_fuel() -> None:
    """Fuel never rejects a walk: the full path executes, billed
    min(cost, fuel) — fuel-0 walks were repeatedly accepted live
    (density runs 2-3, 2026-07-25)."""
    world = _world_with_tank(fuel=3)
    outcome = process_move(world, InMemoryTerrainMap(), 9, 15, 10)
    assert outcome["kind"] == "moved"
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (15, 10)
    assert world["tanks"][9]["fuel"] == 0

    zero = _world_with_tank(fuel=0)
    outcome = process_move(zero, InMemoryTerrainMap(), 9, 13, 10)
    assert outcome["kind"] == "moved"
    assert (zero["tanks"][9]["x"], zero["tanks"][9]["y"]) == (13, 10)
    assert zero["tanks"][9]["fuel"] == 0


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
    place_mine(world, 11, 10, 2)
    outcome = process_move(world, InMemoryTerrainMap(), 9, 11, 10)
    assert outcome["kind"] == "moved"
    assert outcome["mine_positions"] == [(11, 10)]
    assert world["mines"] == {}
    assert world["tanks"][9]["fuel"] == 1000 - 1 - MINE_WALK_COST


def test_revealed_enemy_mine_forces_detour_own_mine_does_not() -> None:
    """Routing avoids REVEALED enemy mines but walks own-color mines.

    Visibility is team-scoped (user contract 2026-08-04): the detour
    only happens once the mover's team has been shown the mine.
    """
    world = _world_with_tank()
    place_mine(world, 10, 11, 2)
    world["revealed_mine_keys_by_team"]["0"] = ["10,11"]
    detoured = process_move(world, InMemoryTerrainMap(), 9, 10, 12)
    assert detoured["kind"] == "moved"
    assert len(detoured["path"]) == 4
    place_mine(world, 10, 11, 0)
    world["tanks"][9]["x"], world["tanks"][9]["y"] = 10, 10
    direct = process_move(world, InMemoryTerrainMap(), 9, 10, 12)
    assert direct["path"] == "ss"


def test_hidden_enemy_mine_is_walked_into_and_arrests_the_move() -> None:
    """An unscanned enemy mine mid-route detonates and stops the walk.

    The server auto-paths around VISIBLE mines only (user contract
    2026-08-04, [[walk-mechanics]]): with no teammate scan on record
    the mine does not exist to the router, the walk steps onto it,
    pays the 45, and arrests there — no code 1.
    """
    world = _world_with_tank()
    place_mine(world, 10, 11, 2)
    outcome = process_move(world, InMemoryTerrainMap(), 9, 10, 12)
    assert outcome["kind"] == "moved"
    assert outcome["path"] == "s"
    assert outcome["mine_positions"] == [(10, 11)]
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (10, 11)
    assert world["tanks"][9]["fuel"] == 1000 - 1 - MINE_WALK_COST
    assert world["mines"] == {}


def test_teammate_scan_reveals_for_the_whole_color() -> None:
    """A DIFFERENT tank's scan makes the mover's route mine-aware.

    Team-scoped visibility (user contract 2026-08-04): teammate 12
    scans the mine with an extra radar; mover 9 — who never scanned —
    then routes around it.
    """
    world = _world_with_tank()
    world["tanks"][12] = make_sim_tank(12, 0, 1, 12, 11, 500)
    world["tanks"][12]["counts"][SLOT_RADAR] = 1
    place_mine(world, 10, 11, 2)
    scan = process_radar(world, 12, None)
    assert scan["mines"] != []
    assert "10,11" in world["revealed_mine_keys_by_team"]["0"]
    detoured = process_move(world, InMemoryTerrainMap(), 9, 10, 12)
    assert detoured["kind"] == "moved"
    assert len(detoured["path"]) == 4
    assert world["mines"] != {}


def test_other_tank_blocks_interior_and_destination() -> None:
    """Living tanks block routing; dead tanks do not.

    A click on an ADJACENT occupied tile is the zero-tile pure
    refusal: the first step is already blocked, nothing walks, and
    the bare code 1 answers (live 2026-08-02 20:58:45 — the one
    code-1 of twelve with no echo).
    """
    world = _world_with_tank()
    world["tanks"][11] = make_sim_tank(11, 1, 1, 11, 10, 500)
    occupied = process_move(world, InMemoryTerrainMap(), 9, 11, 10)
    assert occupied["kind"] == "cant_go"
    assert occupied["path"] == ""
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (10, 10)
    detoured = process_move(world, InMemoryTerrainMap(), 9, 12, 10)
    assert detoured["kind"] == "moved"
    assert len(detoured["path"]) == 4
    world["tanks"][9]["x"], world["tanks"][9]["y"] = 10, 10
    world["tanks"][11]["alive"] = False
    through = process_move(world, InMemoryTerrainMap(), 9, 12, 10)
    assert through["path"] == "ee"


def _corridor_terrain() -> InMemoryTerrainMap:
    """Rock-walled single corridor from (10,10) east to (15,10)."""
    walls = {(x, y): InMemoryTerrainMap.ROCK for x in range(9, 17) for y in (9, 11)}
    walls.update({(9, 10): InMemoryTerrainMap.ROCK, (16, 10): InMemoryTerrainMap.ROCK})
    return InMemoryTerrainMap(walls)


def test_corked_corridor_walks_to_the_blocker_and_reports_cant_go() -> None:
    """A tank corking the only corridor yields the partial-walk receipt.

    The measured choreography (2026-08-04, 12 live code-1s): the
    server plans the route AS IF clear, walks it, and stops one tile
    before the body — live 18:12:35 stopped at (16,24) with Belton on
    (16,23). The outcome is cant_go WITH the walked prefix, and the
    world reflects the movement and its billing.
    """
    world = _world_with_tank()
    world["tanks"][11] = make_sim_tank(11, 1, 1, 13, 10, 500)
    outcome = process_move(world, _corridor_terrain(), 9, 15, 10)
    assert outcome["kind"] == "cant_go"
    assert outcome["path"] == "ee"
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (12, 10)
    assert world["tanks"][9]["fuel"] == 1000 - 2


def test_revealed_mine_severing_the_corridor_stops_the_walk_before_it() -> None:
    """A scanned enemy mine walls the corridor: stop before, code 1.

    Revealed mines BLOCK ([[walk-mechanics]], user contract
    2026-08-04: "terrain blocking or mines blocking") — the walk
    stops before the mine rather than detonating it.
    """
    world = _world_with_tank()
    place_mine(world, 13, 10, 2)
    world["revealed_mine_keys_by_team"]["0"] = ["13,10"]
    outcome = process_move(world, _corridor_terrain(), 9, 15, 10)
    assert outcome["kind"] == "cant_go"
    assert outcome["path"] == "ee"
    assert world["mines"] != {}
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (12, 10)


def test_block_obstacle_severing_the_corridor_stops_the_walk_before_it() -> None:
    """A movable block corking the corridor is hit, not routed through."""
    world = _world_with_tank()
    world["blocks"].append(SimBlockDict(x=13, y=10))
    outcome = process_move(world, _corridor_terrain(), 9, 15, 10)
    assert outcome["kind"] == "cant_go"
    assert outcome["path"] == "ee"
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (12, 10)


def test_terrain_severed_destination_is_the_pure_refusal() -> None:
    """Rock all the way around the destination: no walk, bare cant_go."""
    world = _world_with_tank()
    walls = {(x, y): InMemoryTerrainMap.ROCK for x in range(19, 22) for y in range(9, 12)}
    del walls[(20, 10)]
    outcome = process_move(world, InMemoryTerrainMap(walls), 9, 20, 10)
    assert outcome["kind"] == "cant_go"
    assert outcome["path"] == ""
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (10, 10)
    assert world["tanks"][9]["fuel"] == 1000


def test_hidden_mine_on_the_severed_fallback_walk_detonates_instead() -> None:
    """An unscanned mine met during the fallback walk is the walk-over.

    The corridor is corked by a tank further east, but before the
    walk reaches it an unrevealed enemy mine sits on the route: the
    tank steps onto the mine, pays the 45, and arrests there — the
    walk-over law, not a code 1.
    """
    world = _world_with_tank()
    world["tanks"][11] = make_sim_tank(11, 1, 1, 14, 10, 500)
    place_mine(world, 12, 10, 2)
    outcome = process_move(world, _corridor_terrain(), 9, 15, 10)
    assert outcome["kind"] == "moved"
    assert outcome["path"] == "ee"
    assert outcome["mine_positions"] == [(12, 10)]
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (12, 10)
    assert world["tanks"][9]["fuel"] == 1000 - 2 - MINE_WALK_COST
