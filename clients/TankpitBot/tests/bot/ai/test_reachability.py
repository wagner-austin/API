"""Tests for viewport-bounded reachability helpers.

Mines are not a separate reachability parameter anymore -- hostile
mines are composed into the terrain view (2026-07-20), so these tests
build the composed view where mine-blocking matters.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.ferry import FerryAwareTerrain
from tankpit_bot.bot.ai.pathfinding import find_path
from tankpit_bot.bot.ai.reachability import (
    is_collection_reachable_in_viewport,
    is_move_reachable_in_viewport,
)
from tests.bot.ai._support import make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _mined_terrain(base: InMemoryTerrainMap, keys: frozenset[str]) -> FerryAwareTerrain:
    """Compose hostile-mine keys over a static map for reachability tests.

    Args:
        base: Static terrain data.
        keys: "x,y" hostile-mine coordinate keys.

    Returns:
        Composed terrain view with the mines impassable.
    """
    return FerryAwareTerrain(
        base,
        {},
        riding=False,
        hostile_mine_keys=keys,
        occupied_tank_keys=frozenset(),
    )


def _occupied_terrain(base: InMemoryTerrainMap, keys: frozenset[str]) -> FerryAwareTerrain:
    """Compose tank-body keys over a static map for reachability tests.

    Args:
        base: Static terrain data.
        keys: "x,y" keys of tiles holding another tank's body.

    Returns:
        Composed terrain view with the occupied tiles impassable.
    """
    return FerryAwareTerrain(
        base,
        {},
        riding=False,
        hostile_mine_keys=frozenset(),
        occupied_tank_keys=keys,
    )


class TestTankBodyReachability:
    """Reachability against tank bodies -- the F6 live failure class.

    Live origin: run bot-20260803-180918 drew ten ``error_code=1``
    stops between 18:12 and 18:40. Nine of the ten moved the tank
    before stopping, which is the server's documented behaviour --
    it walks up to the blocker, stops, and reports ([[walk-mechanics]],
    user contract 2026-08-04). The composed view had no notion of a
    tank body, so every one of those routes looked open.
    """

    def test_body_in_the_only_corridor_makes_the_target_unreachable(self) -> None:
        """A body corking a one-tile gap severs the route."""
        walls = {(102, y): "#" for y in range(92, 108) if y != 100}
        base = InMemoryTerrainMap(walls)
        world, self_state = make_world(self_x=100, self_y=100)

        assert (
            is_move_reachable_in_viewport(
                world,
                _occupied_terrain(base, frozenset()),
                self_state["x"],
                self_state["y"],
                104,
                100,
            )
            is True
        )
        assert (
            is_move_reachable_in_viewport(
                world,
                _occupied_terrain(base, frozenset({"102,100"})),
                self_state["x"],
                self_state["y"],
                104,
                100,
            )
            is False
        )

    def test_body_beside_the_corridor_leaves_the_route_open(self) -> None:
        """Occupancy blocks only the tile the body stands on."""
        walls = {(102, y): "#" for y in range(92, 108) if y != 100}
        base = InMemoryTerrainMap(walls)
        world, self_state = make_world(self_x=100, self_y=100)

        result = is_move_reachable_in_viewport(
            world,
            _occupied_terrain(base, frozenset({"103,101"})),
            self_state["x"],
            self_state["y"],
            104,
            100,
        )

        assert result is True

    def test_collection_routes_around_a_body_onto_an_adjacent_tile(self) -> None:
        """A body on open ground is detoured, not treated as a dead end."""
        base = InMemoryTerrainMap()
        world, self_state = make_world(self_x=100, self_y=100)

        result = is_collection_reachable_in_viewport(
            world,
            _occupied_terrain(base, frozenset({"101,100"})),
            self_state["x"],
            self_state["y"],
            103,
            100,
        )

        assert result is True

    def test_collection_target_under_a_body_is_served_from_a_neighbour(self) -> None:
        """A container beneath a tank is collected from an adjacent tile."""
        base = InMemoryTerrainMap()
        world, self_state = make_world(self_x=100, self_y=100)

        result = is_collection_reachable_in_viewport(
            world,
            _occupied_terrain(base, frozenset({"103,100"})),
            self_state["x"],
            self_state["y"],
            103,
            100,
        )

        assert result is True

    def test_collection_target_ringed_by_bodies_is_unreachable(self) -> None:
        """No landing tile survives when every neighbour holds a body."""
        base = InMemoryTerrainMap()
        world, self_state = make_world(self_x=100, self_y=100)
        ring = frozenset({"103,100", "102,100", "104,100", "103,99", "103,101"})

        result = is_collection_reachable_in_viewport(
            world,
            _occupied_terrain(base, ring),
            self_state["x"],
            self_state["y"],
            103,
            100,
        )

        assert result is False


class TestViewportReachability:
    """Tests for viewport-bounded reachability rules."""

    def test_move_reachable_when_detour_stays_inside_viewport(self) -> None:
        """In-viewport detours are accepted."""
        world, self_state = make_world(self_x=100, self_y=100)
        terrain = InMemoryTerrainMap({(102, 100): "#"})

        result = is_move_reachable_in_viewport(
            world,
            terrain,
            self_state["x"],
            self_state["y"],
            104,
            100,
        )

        assert result is True

    def test_move_reachable_rejects_detour_outside_viewport(self) -> None:
        """A full-map detour is rejected when it leaves the visible viewport."""
        terrain_data = {(92, y): "#" for y in range(92, 108)}
        terrain = InMemoryTerrainMap(terrain_data)
        world, self_state = make_world(self_x=100, self_y=100)

        unbounded_path = find_path(terrain, self_state["x"], self_state["y"], 91, 100)
        assert unbounded_path != []

        result = is_move_reachable_in_viewport(
            world,
            terrain,
            self_state["x"],
            self_state["y"],
            91,
            100,
        )

        assert result is False

    def test_collection_reachable_from_adjacent_landing_tile(self) -> None:
        """Collection can succeed from a safe adjacent landing tile.

        The container's own tile carries a hostile mine (impassable in
        the composed view), so the pickup is serviced from a cardinal
        neighbor -- exactly how the bot drained the mined fuel dot at
        (37,153) live on 2026-07-20.
        """
        world, self_state = make_world(self_x=100, self_y=100)
        terrain = _mined_terrain(InMemoryTerrainMap(), frozenset({"104,100"}))

        result = is_collection_reachable_in_viewport(
            world,
            terrain,
            self_state["x"],
            self_state["y"],
            104,
            100,
        )

        assert result is True

    def test_collection_reachable_skips_mined_landing_tile(self) -> None:
        """Collection skips blocked adjacent landings and uses another safe tile."""
        world, self_state = make_world(self_x=100, self_y=100)
        terrain = _mined_terrain(InMemoryTerrainMap(), frozenset({"104,100", "105,100"}))

        result = is_collection_reachable_in_viewport(
            world,
            terrain,
            self_state["x"],
            self_state["y"],
            104,
            100,
        )

        assert result is True

    def test_collection_reachable_rejects_detour_outside_viewport(self) -> None:
        """Collection is rejected when only an out-of-viewport path exists."""
        terrain_data = {(92, y): "#" for y in range(92, 108)}
        terrain = InMemoryTerrainMap(terrain_data)
        world, self_state = make_world(self_x=100, self_y=100)

        unbounded_path = find_path(terrain, self_state["x"], self_state["y"], 91, 100)
        assert unbounded_path != []

        result = is_collection_reachable_in_viewport(
            world,
            terrain,
            self_state["x"],
            self_state["y"],
            91,
            100,
        )

        assert result is False

    def test_collection_reachable_handles_map_edge_candidates(self) -> None:
        """Collection ignores out-of-map adjacent candidates at the map edge."""
        world, self_state = make_world(self_x=254, self_y=10)
        terrain = _mined_terrain(InMemoryTerrainMap(), frozenset({"255,10"}))

        result = is_collection_reachable_in_viewport(
            world,
            terrain,
            self_state["x"],
            self_state["y"],
            255,
            10,
        )

        assert result is True
