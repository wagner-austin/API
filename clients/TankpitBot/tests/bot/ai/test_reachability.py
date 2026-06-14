"""Tests for viewport-bounded reachability helpers."""

from __future__ import annotations

from tankpit_bot.bot.ai.pathfinding import find_path
from tankpit_bot.bot.ai.reachability import (
    is_collection_reachable_in_viewport,
    is_move_reachable_in_viewport,
)
from tankpit_bot.state.types import make_mine_state
from tests.bot.ai._support import make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


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
            world["mines"],
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
            world["mines"],
        )

        assert result is False

    def test_collection_reachable_from_adjacent_landing_tile(self) -> None:
        """Collection can succeed from a safe adjacent landing tile."""
        world, self_state = make_world(self_x=100, self_y=100)
        terrain = InMemoryTerrainMap()
        world["mines"] = {"104,100": make_mine_state(x=104, y=100, mine_type=0, tank_id=-1, team=1)}

        result = is_collection_reachable_in_viewport(
            world,
            terrain,
            self_state["x"],
            self_state["y"],
            104,
            100,
            world["mines"],
        )

        assert result is True

    def test_collection_reachable_skips_mined_landing_tile(self) -> None:
        """Collection skips blocked adjacent landings and uses another safe tile."""
        world, self_state = make_world(self_x=100, self_y=100)
        terrain = InMemoryTerrainMap()
        world["mines"] = {
            "104,100": make_mine_state(x=104, y=100, mine_type=0, tank_id=-1, team=1),
            "105,100": make_mine_state(x=105, y=100, mine_type=0, tank_id=-1, team=1),
        }

        result = is_collection_reachable_in_viewport(
            world,
            terrain,
            self_state["x"],
            self_state["y"],
            104,
            100,
            world["mines"],
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
            world["mines"],
        )

        assert result is False

    def test_collection_reachable_handles_map_edge_candidates(self) -> None:
        """Collection ignores out-of-map adjacent candidates at the map edge."""
        world, self_state = make_world(self_x=254, self_y=10)
        terrain = InMemoryTerrainMap()
        world["mines"] = {"255,10": make_mine_state(x=255, y=10, mine_type=0, tank_id=-1, team=1)}

        result = is_collection_reachable_in_viewport(
            world,
            terrain,
            self_state["x"],
            self_state["y"],
            255,
            10,
            world["mines"],
        )

        assert result is True
