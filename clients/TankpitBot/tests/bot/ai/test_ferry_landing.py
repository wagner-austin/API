"""Direct tests for the ferry boarding-tile search and its pond gate."""

from __future__ import annotations

from tankpit_bot.bot.ai.ferry_landing import find_ferry_boarding_tile
from tankpit_bot.state.types import TerrainTileDict, make_empty_world_state, make_terrain_tile
from tankpit_bot.state.types.constants import TERRAIN_FERRY
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

    assert find_ferry_boarding_tile(world, terrain, 50, 50, _NOW) is None


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

    assert find_ferry_boarding_tile(world, terrain, 0, 0, _NOW) == (2, 0)
