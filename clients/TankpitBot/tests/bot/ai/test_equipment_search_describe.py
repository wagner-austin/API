"""Tests for the container-search description strings."""

from __future__ import annotations

from tankpit_bot.bot.ai.equipment_search import (
    describe_container_search,
    find_nearest_equipment,
)
from tankpit_bot.state.types import (
    make_container_state,
)
from tests.bot.ai._equipment_fixtures import _world_and_self
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestDescribeContainerSearch:
    """Tests for container search diagnostics."""

    def test_reports_actionable_adjacent_equipment(self) -> None:
        """Summary reports nearby actionable equipment when it exists."""
        world, state = _world_and_self()
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=False,
            volume=0,
        )

        result = describe_container_search(
            world,
            state,
            None,
            want_fuel=False,
        )

        assert result == (
            "equipment: total=1 nearby=1 actionable=1 blocked=0 "
            "low_volume=0 nearest=(101,100) actionable"
        )

    def test_reports_water_locked_equipment_as_blocked(self) -> None:
        """Water-locked equipment is reported as blocked (no walk path)."""
        world, state = _world_and_self(x=10, y=10)
        world["containers"]["15,10"] = make_container_state(
            x=15,
            y=10,
            is_fuel=False,
            volume=0,
        )
        terrain_data: dict[tuple[int, int], str] = {
            (15, 10): "W",
            (16, 10): "W",
            (14, 10): "W",
            (15, 11): "#",
            (15, 9): "#",
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)

        result = describe_container_search(
            world,
            state,
            terrain,
            want_fuel=False,
        )

        assert result == (
            "equipment: total=1 nearby=1 actionable=0 blocked=1 "
            "low_volume=0 nearest=(15,10) blocked_walk"
        )

    def test_reports_low_volume_fuel_as_non_actionable(self) -> None:
        """Summary explains when a nearby fuel container is too small to matter."""
        world, state = _world_and_self()
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=True,
            volume=50,
        )

        result = describe_container_search(
            world,
            state,
            None,
            want_fuel=True,
            minimum_volume=100,
        )

        assert result == (
            "fuel: total=1 nearby=1 actionable=0 blocked=0 "
            "low_volume=1 nearest=(101,100) low_volume"
        )

    def test_ignores_other_container_type_and_out_of_viewport_target(self) -> None:
        """Summary filters out mismatched types and off-viewport candidates."""
        world, state = _world_and_self()
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=True,
            volume=500,
        )
        world["containers"]["109,100"] = make_container_state(
            x=109,
            y=100,
            is_fuel=False,
            volume=0,
        )

        result = describe_container_search(
            world,
            state,
            None,
            want_fuel=False,
        )

        assert result == (
            "equipment: total=1 nearby=0 actionable=0 blocked=0 low_volume=0 nearest=none"
        )

    def test_reports_blocked_walk_when_no_walk_path(self) -> None:
        """Summary marks blocked targets as non-actionable under the walk-only contract."""
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = InMemoryTerrainMap(terrain_data=wall)
        world["containers"]["15,10"] = make_container_state(
            x=15,
            y=10,
            is_fuel=False,
            volume=0,
        )

        result = describe_container_search(
            world,
            state,
            terrain,
            want_fuel=False,
        )

        assert result == (
            "equipment: total=1 nearby=1 actionable=0 blocked=1 "
            "low_volume=0 nearest=(15,10) blocked_walk"
        )

    def test_keeps_nearest_description_when_later_candidate_is_farther(self) -> None:
        """Summary preserves the nearest candidate when later ones are farther away."""
        world, state = _world_and_self()
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=False,
            volume=0,
        )
        world["containers"]["103,100"] = make_container_state(
            x=103,
            y=100,
            is_fuel=False,
            volume=0,
        )

        result = describe_container_search(
            world,
            state,
            None,
            want_fuel=False,
        )

        assert result == (
            "equipment: total=2 nearby=2 actionable=2 blocked=0 "
            "low_volume=0 nearest=(101,100) actionable"
        )

    def test_reports_failed_pickup_container_as_non_actionable(self) -> None:
        """Summary matches selector behavior for failed pickup targets."""
        world, state = _world_and_self()
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=False,
            volume=0,
            failed_pickups=1,
        )

        result = describe_container_search(
            world,
            state,
            None,
            want_fuel=False,
        )

        assert result == (
            "equipment: total=1 nearby=1 actionable=0 blocked=0 "
            "low_volume=0 nearest=(101,100) failed_pickup"
        )

    def test_reports_actionable_visible_target_without_viewport_scan_flag(self) -> None:
        """Visible targets are diagnosed by reachability, not viewport-origin scan state."""
        world, state = _world_and_self()
        world["scanned_tiles"] = {}
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=False,
            volume=0,
        )

        result = describe_container_search(
            world,
            state,
            None,
            want_fuel=False,
        )

        assert result == (
            "equipment: total=1 nearby=1 actionable=1 blocked=0 "
            "low_volume=0 nearest=(101,100) actionable"
        )

    def test_with_terrain_skips_unreachable(self) -> None:
        """Skips equipment container that is unreachable due to terrain."""
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = InMemoryTerrainMap(terrain_data=wall)
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=False, volume=0)
        assert find_nearest_equipment(world, state, terrain) is None

    def test_with_terrain_accepts_reachable(self) -> None:
        """Accepts equipment container that is reachable through terrain."""
        world, state = _world_and_self(x=10, y=10)
        terrain = InMemoryTerrainMap()
        expected = make_container_state(x=15, y=10, is_fuel=False, volume=0)
        world["containers"]["15,10"] = expected
        assert find_nearest_equipment(world, state, terrain) == expected

    def test_with_terrain_skips_wall_blocked_equipment(self) -> None:
        """Wall-blocked equipment is skipped under the walk-only contract."""
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = InMemoryTerrainMap(terrain_data=wall)
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=False, volume=0)
        assert find_nearest_equipment(world, state, terrain) is None
