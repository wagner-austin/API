"""Laws 5-8: teleport displacement, radar, map, and mine placement."""

from __future__ import annotations

from tankpit_bot.physics.capacity import free_radar_radius
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.sim.actions import (
    build_map_data,
    process_mine_press,
    process_radar,
    process_teleport,
)
from tankpit_bot.sim.combat import SLOT_RADAR
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimMineDict,
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _world() -> SimWorldDict:
    """Client tank 9 (team 0) at (100, 100) with 800 fuel."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 100, 100, 800)
    return world


def test_teleport_lands_on_clear_target_and_bills_actual_distance() -> None:
    """A clear target lands exactly and costs floor(6 x euclid)."""
    world = _world()
    outcome = process_teleport(world, InMemoryTerrainMap(), 9, 130, 140)
    assert outcome["kind"] == "landed"
    assert (outcome["landed_x"], outcome["landed_y"]) == (130, 140)
    assert outcome["cost"] == teleport_cost(100, 100, 130, 140)
    assert world["tanks"][9]["fuel"] == 800 - outcome["cost"]


def test_teleport_displacement_prefers_east_then_north_then_west() -> None:
    """Ring-1 displacement follows the measured E -> N -> W order."""
    world = _world()
    world["mines"].append(SimMineDict(x=130, y=140, team=1))
    east = process_teleport(world, InMemoryTerrainMap(), 9, 130, 140)
    assert (east["landed_x"], east["landed_y"]) == (131, 140)
    assert east["cost"] == teleport_cost(100, 100, 131, 140)

    world = _world()
    world["mines"].append(SimMineDict(x=130, y=140, team=1))
    terrain = InMemoryTerrainMap(terrain_data={(131, 140): "#"})
    north = process_teleport(world, terrain, 9, 130, 140)
    assert (north["landed_x"], north["landed_y"]) == (130, 139)

    world = _world()
    world["mines"].append(SimMineDict(x=130, y=140, team=1))
    terrain = InMemoryTerrainMap(terrain_data={(131, 140): "#", (130, 139): "#"})
    west = process_teleport(world, terrain, 9, 130, 140)
    assert (west["landed_x"], west["landed_y"]) == (129, 140)


def test_teleport_south_is_the_last_resort_and_full_ring_blocks() -> None:
    """South lands only when E/N/W are blocked; a sealed ring rejects."""
    walls = {(131, 140): "#", (130, 139): "#", (129, 140): "#"}
    world = _world()
    world["mines"].append(SimMineDict(x=130, y=140, team=1))
    south = process_teleport(world, InMemoryTerrainMap(terrain_data=dict(walls)), 9, 130, 140)
    assert (south["landed_x"], south["landed_y"]) == (130, 141)

    sealed = dict(walls)
    sealed[(130, 141)] = "#"
    world = _world()
    world["mines"].append(SimMineDict(x=130, y=140, team=1))
    blocked = process_teleport(world, InMemoryTerrainMap(terrain_data=sealed), 9, 130, 140)
    assert blocked["kind"] == "blocked"
    assert (world["tanks"][9]["x"], world["tanks"][9]["y"]) == (100, 100)
    assert world["tanks"][9]["fuel"] == 800


def test_teleport_blockers_tank_blocks_own_mine_does_not() -> None:
    """Another tank displaces the landing; an own-color mine does not."""
    world = _world()
    world["tanks"][11] = make_sim_tank(11, 1, 1, 130, 140, 500)
    displaced = process_teleport(world, InMemoryTerrainMap(), 9, 130, 140)
    assert (displaced["landed_x"], displaced["landed_y"]) == (131, 140)

    world = _world()
    world["mines"].append(SimMineDict(x=130, y=140, team=0))
    direct = process_teleport(world, InMemoryTerrainMap(), 9, 130, 140)
    assert (direct["landed_x"], direct["landed_y"]) == (130, 140)


def test_teleport_insufficient_fuel_and_landing_pickup() -> None:
    """Cost above fuel rejects; a landing container auto-drains."""
    world = _world()
    world["tanks"][9]["fuel"] = 5
    rejected = process_teleport(world, InMemoryTerrainMap(), 9, 130, 140)
    assert rejected["kind"] == "insufficient_fuel"

    world = _world()
    world["containers"].append(SimContainerDict(x=130, y=140, volume=100, dotted=True))
    landed = process_teleport(world, InMemoryTerrainMap(), 9, 130, 140)
    assert landed["pickups"] == [{"x": 130, "y": 140, "remaining_volume": 0}]
    assert world["tanks"][9]["fuel"] == 800 - landed["cost"] + 100


def test_radar_consumes_extra_for_the_viewport_window() -> None:
    """An available extra covers exactly the 16x16 window and is consumed.

    The window spans [x-8, x+8) for a centered tank — column x+8 is
    OUTSIDE it (the acceptance-boundary measurement,
    [[viewport-shift-protocol]]). Volume-0 containers ARE reported —
    the wire's cache value 0 is the client's "tile is empty" removal
    signal (323 zero-volume reveals in the archive).
    """
    world = _world()
    world["tanks"][9]["counts"][SLOT_RADAR] = 2
    world["containers"].append(SimContainerDict(x=108, y=100, volume=50, dotted=True))
    world["containers"].append(SimContainerDict(x=107, y=100, volume=0, dotted=True))
    world["containers"].append(SimContainerDict(x=92, y=100, volume=70, dotted=True))
    world["containers"].append(SimContainerDict(x=91, y=100, volume=70, dotted=True))
    world["mines"].append(SimMineDict(x=100, y=107, team=1))
    outcome = process_radar(world, 9)
    assert outcome["consumed_extra"] is True
    assert world["tanks"][9]["counts"][SLOT_RADAR] == 1
    assert [(c["x"], c["y"], c["volume"]) for c in outcome["containers"]] == [
        (107, 100, 0),
        (92, 100, 70),
    ]
    assert [(m["x"], m["y"], m["team"]) for m in outcome["mines"]] == [(100, 107, 1)]
    assert outcome["enemy_found"] is False


def test_radar_window_override_covers_the_stored_window() -> None:
    """A drifted window (the tank walked; autoscroll OFF never
    recenters) scans the WINDOW's tiles, not the tank's surroundings."""
    world = _world()
    world["tanks"][9]["counts"][SLOT_RADAR] = 1
    world["containers"].append(SimContainerDict(x=85, y=100, volume=60, dotted=True))
    world["containers"].append(SimContainerDict(x=108, y=100, volume=60, dotted=True))
    outcome = process_radar(world, 9, (84, 92))
    assert outcome["consumed_extra"] is True
    assert [(c["x"], c["y"], c["volume"]) for c in outcome["containers"]] == [(85, 100, 60)]


def test_radar_exposure_dots_large_hidden_fuel() -> None:
    """A reveal at >= 500 volume joins the atlas; smaller never does.

    The measured 2026-07-25 exposure law: the dot threshold is
    exactly ``MAP_DOT_MIN_VOLUME`` (500), and dotting is permanent —
    a later drain does not undo it.
    """
    world = _world()
    world["tanks"][9]["counts"][SLOT_RADAR] = 1
    world["containers"].append(SimContainerDict(x=105, y=100, volume=499, dotted=False))
    world["containers"].append(SimContainerDict(x=106, y=100, volume=500, dotted=False))
    process_radar(world, 9)
    by_pos = {(c["x"], c["y"]): c for c in world["containers"]}
    assert by_pos[(105, 100)]["dotted"] is False
    assert by_pos[(106, 100)]["dotted"] is True
    by_pos[(106, 100)]["volume"] = 0
    assert (106, 100) in set(build_map_data(world)["fuel_dots"])


def test_radar_without_extras_uses_rank_radius_and_finds_enemies() -> None:
    """No extras (or a disabled slot) falls back to the built-in radius."""
    world = _world()
    radius = free_radar_radius(1)
    world["tanks"][11] = make_sim_tank(11, 1, 1, 100 + radius, 100, 500)
    outcome = process_radar(world, 9)
    assert outcome["consumed_extra"] is False
    assert outcome["enemy_found"] is True

    world["tanks"][9]["counts"][SLOT_RADAR] = 5
    world["tanks"][9]["enabled"][SLOT_RADAR] = False
    disabled = process_radar(world, 9)
    assert disabled["consumed_extra"] is False
    assert world["tanks"][9]["counts"][SLOT_RADAR] == 5


def test_map_data_sorts_dots_and_lists_living_tanks() -> None:
    """The map snapshot: atlas-ordered DOTTED containers, live blips.

    Dots are exposure memory (2026-07-25): a drained dotted container
    stays on the map, and an unexposed stocked container never
    appears. Mines are absent by law.
    """
    world = _world()
    world["containers"].append(SimContainerDict(x=5, y=200, volume=10, dotted=True))
    world["containers"].append(SimContainerDict(x=200, y=3, volume=10, dotted=True))
    world["containers"].append(SimContainerDict(x=1, y=1, volume=0, dotted=True))
    world["containers"].append(SimContainerDict(x=2, y=1, volume=900, dotted=False))
    world["tanks"][11] = make_sim_tank(11, 1, 2, 50, 60, 500)
    world["tanks"][11]["alive"] = False
    snapshot = build_map_data(world)
    assert snapshot["fuel_dots"] == [(1, 1), (200, 3), (5, 200)]
    assert [entry["tank_id"] for entry in snapshot["tanks"]] == [9]


def test_mine_press_places_skips_and_trades_one_to_one() -> None:
    """The 3x3 press: clear tiles filled, blockers skipped, 1:1 trades."""
    world = _world()
    world["tanks"][11] = make_sim_tank(11, 1, 1, 101, 100, 500)
    world["mines"].append(SimMineDict(x=99, y=100, team=1))
    world["mines"].append(SimMineDict(x=100, y=99, team=0))
    terrain = InMemoryTerrainMap(terrain_data={(99, 99): "#"})
    outcome = process_mine_press(world, terrain, 9)
    assert outcome["detonated"] == [(99, 100)]
    assert (101, 100) not in outcome["placed"]
    assert (99, 99) not in outcome["placed"]
    assert (100, 99) not in outcome["placed"]
    assert (100, 100) in outcome["placed"]
    assert len(outcome["placed"]) == 5
    own_positions = {(m["x"], m["y"]) for m in world["mines"] if m["team"] == 0}
    assert set(outcome["placed"]) | {(100, 99)} == own_positions
