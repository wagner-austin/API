"""Direct tests for the ferry boarding-tile search and its pond gate."""

from __future__ import annotations

from tankpit_bot.bot.ai.ferry_landing import find_ferry_boarding_tile, goal_water_pond
from tankpit_bot.state.types import TerrainTileDict, make_empty_world_state, make_terrain_tile
from tankpit_bot.types.constants import TERRAIN_FERRY
from tests.in_memory_terrain_map import InMemoryTerrainMap

_NOW = 100000


def _ferry_at(world_terrain: dict[str, TerrainTileDict], x: int, y: int) -> None:
    """Record a fresh ferry belief at the given tile."""
    world_terrain[f"{x},{y}"] = make_terrain_tile(x, y, TERRAIN_FERRY, observed_ms=_NOW)


def test_goal_not_afloat_yields_no_boarding_tile() -> None:
    """A goal on solid ground has no pond, so no ferry can serve it.

    The boarding search only exists for water-locked goals; a
    rock-pocket container that reaches it by mistake must decline
    even with a fresh ferry two tiles away.
    """
    world = make_empty_world_state()
    _ferry_at(world["terrain"], 52, 50)
    terrain = InMemoryTerrainMap(terrain_data={(52, 50): "W", (52, 51): "W"})

    assert find_ferry_boarding_tile(world, terrain, 50, 50) is None


def test_pond_at_the_map_edge_still_serves_the_goal() -> None:
    """The flood clamps at the map border instead of walking off it.

    Goal in the corner pond at (0, 0) with the ferry afloat on the
    same water: the boarding tile is found and the border tiles do
    not leak the fill out of bounds.
    """
    world = make_empty_world_state()
    _ferry_at(world["terrain"], 2, 0)
    water = {(x, y): "W" for x in range(0, 4) for y in range(0, 3)}
    terrain = InMemoryTerrainMap(terrain_data=water)

    assert find_ferry_boarding_tile(world, terrain, 0, 0) == (2, 0)


def _two_ponds() -> InMemoryTerrainMap:
    """Two water bodies on one row, split by a single land ridge.

    field01's own geometry in miniature: the wiki records a container
    pond of 4,456 tiles that does NOT contain the ferry tile at
    (112,15), because one land ridge at (111,15) separates the two
    bodies ([[ferry-mechanics]]). A ferry in the far pond is
    unreachable however near it looks by distance.
    """
    water = {(x, 10): InMemoryTerrainMap.WATER for x in (10, 11, 12)}
    water.update({(x, 10): InMemoryTerrainMap.WATER for x in (14, 15)})
    return InMemoryTerrainMap(terrain_data=water)


def test_a_land_ridge_splits_the_pond_and_excludes_the_far_water() -> None:
    """The gate: distance does not make a ferry reachable, connectivity does.

    Proved DIRECTLY here rather than as an emergent property of a
    60-round session. The session test that used to be this law's only
    proof broke when an unrelated law changed, because its assertion
    depends on the bot reaching the decision at all — which is how a
    scenario ends up encoding whatever the sim did at the time.
    """
    pond = goal_water_pond(_two_ponds(), 10, 10)

    assert pond == {(10, 10), (11, 10), (12, 10)}
    assert (14, 10) not in pond
    assert (15, 10) not in pond


def test_a_goal_on_land_has_no_pond_at_all() -> None:
    """A container that is not afloat yields an empty set, not a guess."""
    assert goal_water_pond(_two_ponds(), 13, 10) == set()


def test_a_live_ferry_tile_conducts_the_pond() -> None:
    """A ferry renders OVER its lake, so it must not split the body.

    Treating the ferry as a wall would both cut the pond in two and
    exclude the very boarding tile the search is validating.
    """
    tiles = {(x, 10): InMemoryTerrainMap.WATER for x in (10, 11, 13, 14)}
    tiles[(12, 10)] = "~"
    pond = goal_water_pond(InMemoryTerrainMap(terrain_data=tiles), 10, 10)

    assert (12, 10) in pond
    assert (14, 10) in pond


def test_the_flood_stops_at_the_map_edge() -> None:
    """Coordinates run 0-255; the fill never walks off the field."""
    edge = {(0, 0): InMemoryTerrainMap.WATER, (1, 0): InMemoryTerrainMap.WATER}
    assert goal_water_pond(InMemoryTerrainMap(terrain_data=edge), 0, 0) == {(0, 0), (1, 0)}
