"""Tests for AI pathfinding."""

from __future__ import annotations

from tankpit_bot.bot.ai.pathfinding import (
    find_path,
    find_path_segment_target,
    is_direct_path_clear,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap

# =============================================================================
# find_path
# =============================================================================


class TestFindPath:
    """Tests for A* pathfinding."""

    def test_same_start_and_goal(self) -> None:
        """Path from a point to itself is a single step."""
        terrain = InMemoryTerrainMap()
        path = find_path(terrain, 100, 100, 100, 100)
        assert len(path) == 1
        assert path[0]["x"] == 100
        assert path[0]["y"] == 100

    def test_straight_horizontal(self) -> None:
        """Horizontal path on open ground."""
        terrain = InMemoryTerrainMap()
        path = find_path(terrain, 10, 10, 15, 10)
        assert len(path) == 6  # 10,11,12,13,14,15
        assert path[0]["x"] == 10
        assert path[0]["y"] == 10
        assert path[-1]["x"] == 15
        assert path[-1]["y"] == 10
        # All y coords should be 10 for straight horizontal
        for step in path:
            assert step["y"] == 10

    def test_straight_vertical(self) -> None:
        """Vertical path on open ground."""
        terrain = InMemoryTerrainMap()
        path = find_path(terrain, 10, 10, 10, 15)
        assert len(path) == 6
        assert path[0]["x"] == 10
        assert path[-1]["y"] == 15
        for step in path:
            assert step["x"] == 10

    def test_diagonal_manhattan(self) -> None:
        """Diagonal path uses Manhattan distance steps."""
        terrain = InMemoryTerrainMap()
        path = find_path(terrain, 10, 10, 13, 13)
        # Manhattan distance is 6, path has 7 steps (including start)
        assert len(path) == 7
        assert path[0]["x"] == 10
        assert path[0]["y"] == 10
        assert path[-1]["x"] == 13
        assert path[-1]["y"] == 13

    def test_avoids_water(self) -> None:
        """Path routes around water tiles."""
        # Water wall at x=12 from y=9 to y=11
        water_tiles = {
            (12, 9): "W",
            (12, 10): "W",
            (12, 11): "W",
        }
        terrain = InMemoryTerrainMap(water_tiles)
        path = find_path(terrain, 10, 10, 14, 10)
        # Must route around the water wall
        assert len(path) > 5  # Direct would be 5 steps
        assert path[0]["x"] == 10
        assert path[-1]["x"] == 14
        assert path[-1]["y"] == 10
        # Verify no step is on water
        for step in path:
            assert (step["x"], step["y"]) not in water_tiles

    def test_avoids_rocks(self) -> None:
        """Path routes around rock tiles."""
        rock_tiles = {(12, 10): "#"}
        terrain = InMemoryTerrainMap(rock_tiles)
        path = find_path(terrain, 10, 10, 14, 10)
        assert len(path) > 5
        for step in path:
            assert (step["x"], step["y"]) not in rock_tiles

    def test_avoids_composed_hostile_mine_tiles(self) -> None:
        """Path routes around mines composed into the terrain view.

        Dynamic blockers are not a parameter anymore (2026-07-20) --
        hostile mines arrive through ``is_passable`` like every other
        impassable tile.
        """
        from tankpit_bot.bot.ai.ferry import FerryAwareTerrain

        terrain = FerryAwareTerrain(
            InMemoryTerrainMap(),
            {},
            riding=False,
            hostile_mine_keys=frozenset({"12,10"}),
            occupied_tank_keys=frozenset(),
        )
        path = find_path(terrain, 10, 10, 14, 10)
        assert len(path) > 5
        for step in path:
            assert not (step["x"] == 12 and step["y"] == 10)

    def test_no_path_returns_empty(self) -> None:
        """Returns empty list when completely blocked."""
        # Surround goal with water
        blocked = {}
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                blocked[(50 + dx, 50 + dy)] = "W"
        # Also block the 4-connected neighbors explicitly
        blocked[(49, 50)] = "W"
        blocked[(51, 50)] = "W"
        blocked[(50, 49)] = "W"
        blocked[(50, 51)] = "W"
        terrain = InMemoryTerrainMap(blocked)
        path = find_path(terrain, 10, 10, 50, 50)
        assert path == []

    def test_respects_map_bounds(self) -> None:
        """Path stays within 0-255 bounds."""
        terrain = InMemoryTerrainMap()
        path = find_path(terrain, 0, 0, 5, 0)
        assert len(path) == 6
        for step in path:
            assert 0 <= step["x"] <= 255
            assert 0 <= step["y"] <= 255

    def test_path_at_map_edge(self) -> None:
        """Path works at map boundaries."""
        terrain = InMemoryTerrainMap()
        path = find_path(terrain, 253, 0, 255, 0)
        assert len(path) == 3
        assert path[-1]["x"] == 255

    def test_path_includes_start_and_goal(self) -> None:
        """Path always includes both endpoints."""
        terrain = InMemoryTerrainMap()
        path = find_path(terrain, 50, 50, 55, 50)
        assert path[0]["x"] == 50
        assert path[0]["y"] == 50
        assert path[-1]["x"] == 55
        assert path[-1]["y"] == 50

    def test_consecutive_steps_adjacent(self) -> None:
        """Each step in path is adjacent to the previous (4-connected)."""
        terrain = InMemoryTerrainMap()
        path = find_path(terrain, 10, 10, 20, 15)
        for i in range(1, len(path)):
            dx = abs(path[i]["x"] - path[i - 1]["x"])
            dy = abs(path[i]["y"] - path[i - 1]["y"])
            assert dx + dy == 1, f"Steps {i - 1} and {i} are not adjacent"


# =============================================================================
# find_path step counts
# =============================================================================


class TestFindPathStepCounts:
    """Tests for the number of steps find_path returns.

    These asserted through ``path_length``, a zero-production-caller alias
    for ``len()`` deleted 2026-08-08. The assertions are about ``find_path``
    and are kept; only the wrapper is gone.
    """

    def test_same_tile_path_is_one_step(self) -> None:
        """A path to the tile you already stand on is just that tile."""
        terrain = InMemoryTerrainMap()
        path = find_path(terrain, 10, 10, 10, 10)
        assert len(path) == 1

    def test_straight_run_includes_both_endpoints(self) -> None:
        """Five tiles east is six steps: the origin plus each move."""
        terrain = InMemoryTerrainMap()
        path = find_path(terrain, 0, 0, 5, 0)
        assert len(path) == 6


class TestDirectPathHelpers:
    """Tests for direct-path and waypoint helpers."""

    def test_is_direct_path_clear_on_open_ground(self) -> None:
        """Direct path is clear when no terrain blocks the line."""
        terrain = InMemoryTerrainMap()
        assert is_direct_path_clear(terrain, 10, 10, 15, 15) is True

    def test_is_direct_path_clear_horizontal_line(self) -> None:
        """Direct path handles horizontal lines without diagonal stepping."""
        terrain = InMemoryTerrainMap()
        assert is_direct_path_clear(terrain, 10, 10, 15, 10) is True

    def test_is_direct_path_clear_vertical_line(self) -> None:
        """Direct path handles vertical lines without horizontal stepping."""
        terrain = InMemoryTerrainMap()
        assert is_direct_path_clear(terrain, 10, 10, 10, 15) is True

    def test_is_direct_path_clear_detects_blocked_line(self) -> None:
        """Direct path is blocked when any line tile is impassable."""
        terrain = InMemoryTerrainMap({(12, 12): "#"})
        assert is_direct_path_clear(terrain, 10, 10, 15, 15) is False

    def test_is_direct_path_clear_detects_mine_blocker(self) -> None:
        """Direct path is blocked when a composed hostile mine sits on the line."""
        from tankpit_bot.bot.ai.ferry import FerryAwareTerrain

        terrain = FerryAwareTerrain(
            InMemoryTerrainMap(),
            {},
            riding=False,
            hostile_mine_keys=frozenset({"12,10"}),
            occupied_tank_keys=frozenset(),
        )
        assert is_direct_path_clear(terrain, 10, 10, 15, 10) is False

    def test_find_path_segment_target_returns_straight_goal(self) -> None:
        """Waypoint helper returns the goal when the first segment stays straight."""
        terrain = InMemoryTerrainMap()
        assert find_path_segment_target(terrain, 10, 10, 15, 10) == (15, 10)

    def test_find_path_segment_target_returns_first_turn_waypoint(self) -> None:
        """Waypoint helper returns the farthest directly-reachable tile on the A* detour."""
        terrain = InMemoryTerrainMap({(12, 10): "#"})
        assert find_path_segment_target(terrain, 10, 10, 14, 10) == (13, 11)

    def test_find_path_segment_target_avoids_mined_detour(self) -> None:
        """Waypoint helper excludes composed-mine tiles from the segment."""
        from tankpit_bot.bot.ai.ferry import FerryAwareTerrain

        terrain = FerryAwareTerrain(
            InMemoryTerrainMap({(12, 10): "#"}),
            {},
            riding=False,
            hostile_mine_keys=frozenset({"13,11"}),
            occupied_tank_keys=frozenset(),
        )
        assert find_path_segment_target(terrain, 10, 10, 14, 10) == (13, 9)

    def test_find_path_segment_target_returns_none_when_no_path(self) -> None:
        """Waypoint helper returns None when no path exists."""
        blocked = {
            (11, 10): "W",
            (9, 10): "W",
            (10, 11): "W",
            (10, 9): "W",
        }
        terrain = InMemoryTerrainMap(blocked)
        assert find_path_segment_target(terrain, 10, 10, 15, 10) is None

    def test_find_path_segment_target_respects_action_bounds(self) -> None:
        """Waypoint helper returns None when bounds exclude the goal tile."""
        terrain = InMemoryTerrainMap({(12, 10): "#"})
        result = find_path_segment_target(
            terrain,
            10,
            10,
            14,
            10,
            min_x=10,
            min_y=10,
            max_x=12,
            max_y=12,
        )
        assert result is None

    def test_find_path_segment_target_breaks_at_first_non_direct_candidate(self) -> None:
        """Waypoint helper keeps the last directly reachable step before a bend."""
        terrain = InMemoryTerrainMap({(11, 10): "#", (11, 9): "#"})
        result = find_path_segment_target(terrain, 10, 10, 14, 10)
        assert result == (11, 11)

    def test_find_path_segment_target_returns_none_when_min_x_excludes_progress(self) -> None:
        """Waypoint helper rejects candidates left of the supplied min_x bound."""
        terrain = InMemoryTerrainMap()
        result = find_path_segment_target(
            terrain,
            10,
            10,
            12,
            10,
            min_x=11,
            max_x=10,
        )
        assert result is None

    def test_find_path_segment_target_returns_none_when_min_y_excludes_progress(self) -> None:
        """Waypoint helper rejects candidates above the supplied min_y bound."""
        terrain = InMemoryTerrainMap()
        result = find_path_segment_target(
            terrain,
            10,
            10,
            10,
            12,
            min_y=11,
            max_y=10,
        )
        assert result is None
