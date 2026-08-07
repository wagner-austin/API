"""Tests for reachability, landing tiles, and equipment search.

``test_equipment.py`` was 905 lines; fuel search and search
descriptions are now siblings.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.equipment_search import (
    find_adjacent_container,
    find_nearest_deposit,
    find_nearest_equipment,
    find_teleport_landing_tile,
    is_reachable,
)
from tankpit_bot.state.types import (
    make_container_state,
)
from tests.bot.ai._equipment_fixtures import _world_and_self
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestIsReachable:
    """Tests for is_reachable helper."""

    def test_reachable_open_terrain(self) -> None:
        """Returns True when path exists on open ground."""
        terrain = InMemoryTerrainMap()
        assert is_reachable(terrain, 10, 10, 15, 10) is True

    def test_same_position(self) -> None:
        """Returns True when start equals goal."""
        terrain = InMemoryTerrainMap()
        assert is_reachable(terrain, 10, 10, 10, 10) is True

    def test_blocked_by_wall(self) -> None:
        """Returns False when terrain blocks all paths."""
        # Create a wall of rocks from y=0 to y=255 at x=12
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = InMemoryTerrainMap(terrain_data=wall)
        assert is_reachable(terrain, 10, 10, 15, 10) is False

    def test_blocked_by_mines(self) -> None:
        """Returns False when composed hostile mines block the only route.

        Mines are not a separate parameter anymore -- they are composed
        into the terrain view (2026-07-20), so every passability
        consumer inherits them through ``is_passable``.
        """
        from tankpit_bot.bot.ai.ferry import FerryAwareTerrain

        terrain = FerryAwareTerrain(
            InMemoryTerrainMap(),
            {},
            riding=False,
            hostile_mine_keys=frozenset(f"12,{y}" for y in range(256)),
            occupied_tank_keys=frozenset(),
        )
        assert is_reachable(terrain, 10, 10, 15, 10) is False


class TestFindTeleportLandingTile:
    """Tests for blocked-target teleport landing selection."""

    def test_returns_target_when_passable(self) -> None:
        """Returns target coordinates when the tile is passable ground."""
        terrain = InMemoryTerrainMap()

        result = find_teleport_landing_tile(terrain, 128, 126)

        assert result == (128, 126)

    def test_returns_adjacent_passable_when_target_is_water(self) -> None:
        """Returns nearest passable cardinal neighbor when target is water."""
        terrain_data: dict[tuple[int, int], str] = {
            (128, 126): "W",
            (129, 126): "W",
            (127, 126): "W",
            (128, 127): "#",
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)

        result = find_teleport_landing_tile(terrain, 128, 126)

        assert result == (128, 125)

    def test_returns_none_when_all_adjacent_impassable(self) -> None:
        """Returns None when target and all cardinal neighbors are impassable."""
        terrain_data: dict[tuple[int, int], str] = {
            (128, 126): "W",
            (129, 126): "W",
            (127, 126): "W",
            (128, 127): "W",
            (128, 125): "W",
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)

        result = find_teleport_landing_tile(terrain, 128, 126)

        assert result is None

    def test_returns_none_for_out_of_bounds(self) -> None:
        """Returns None for out-of-bounds coordinates."""
        terrain = InMemoryTerrainMap()

        assert find_teleport_landing_tile(terrain, 300, 300) is None


class TestFindNearestEquipment:
    """Tests for find_nearest_equipment."""

    def test_no_containers(self) -> None:
        """Returns None when no containers exist."""
        world, state = _world_and_self()
        assert find_nearest_equipment(world, state) is None

    def test_no_equipment_containers(self) -> None:
        """Returns None when only fuel containers exist."""
        world, state = _world_and_self()
        world["containers"]["105,100"] = make_container_state(
            x=105,
            y=100,
            is_fuel=True,
            volume=50,
        )
        assert find_nearest_equipment(world, state) is None

    def test_single_equipment_container(self) -> None:
        """Returns the only equipment container."""
        world, state = _world_and_self()
        expected = make_container_state(x=106, y=100, is_fuel=False, volume=0)
        world["containers"]["106,100"] = expected
        assert find_nearest_equipment(world, state) == expected

    def test_nearest_of_multiple(self) -> None:
        """Returns the closest equipment container, skipping farther ones."""
        world, state = _world_and_self()
        # Insert close one first so far one exercises the dist >= best_dist branch
        closest = make_container_state(x=103, y=100, is_fuel=False, volume=0)
        world["containers"]["103,100"] = closest
        world["containers"]["140,100"] = make_container_state(
            x=140,
            y=100,
            is_fuel=False,
            volume=0,
        )
        assert find_nearest_equipment(world, state) == closest

    def test_farther_visible_equipment_does_not_replace_closer_target(self) -> None:
        """Keeps the closest visible equipment when a farther visible one is checked later."""
        world, state = _world_and_self()
        closest = make_container_state(x=103, y=100, is_fuel=False, volume=0)
        world["containers"]["103,100"] = closest
        world["containers"]["107,100"] = make_container_state(
            x=107,
            y=100,
            is_fuel=False,
            volume=0,
        )
        assert find_nearest_equipment(world, state) == closest

    def test_ignores_fuel(self) -> None:
        """Equipment search skips fuel containers even if closer."""
        world, state = _world_and_self()
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=True,
            volume=80,
        )
        expected = make_container_state(x=107, y=100, is_fuel=False, volume=0)
        world["containers"]["107,100"] = expected
        assert find_nearest_equipment(world, state) == expected

    def test_skips_failed_pickup_equipment(self) -> None:
        """find_nearest_equipment skips containers with failed_pickups > 0."""
        world, state = _world_and_self()
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=False,
            volume=0,
            failed_pickups=2,
        )
        farther = make_container_state(
            x=105,
            y=100,
            is_fuel=False,
            volume=0,
        )
        world["containers"]["105,100"] = farther
        assert find_nearest_equipment(world, state) == farther

    def test_age_does_not_filter_equipment(self) -> None:
        """find_nearest_equipment ignores container age (TTL removed 2026-07-06)."""
        world, state = _world_and_self()
        old = make_container_state(
            x=101,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=10000,
        )
        world["containers"]["101,100"] = old
        world["containers"]["103,100"] = make_container_state(
            x=103,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=90000,
        )
        world["timestamp_ms"] = 100000
        assert find_nearest_equipment(world, state) == old


class TestFindNearestDeposit:
    """Tests for find_nearest_deposit."""

    def test_no_containers(self) -> None:
        """Returns None when no containers exist."""
        world, state = _world_and_self()
        assert find_nearest_deposit(world, state) is None

    def test_finds_fuel_container(self) -> None:
        """Returns nearest fuel container for depositing."""
        world, state = _world_and_self()
        expected = make_container_state(x=105, y=100, is_fuel=True, volume=50)
        world["containers"]["105,100"] = expected
        assert find_nearest_deposit(world, state) == expected

    def test_with_terrain(self) -> None:
        """Deposit search respects terrain reachability."""
        world, state = _world_and_self(x=10, y=10)
        terrain = InMemoryTerrainMap()
        expected = make_container_state(x=15, y=10, is_fuel=True, volume=50)
        world["containers"]["15,10"] = expected
        assert find_nearest_deposit(world, state, terrain) == expected


def test_find_adjacent_container_skips_diagonal_with_blocked_cardinals() -> None:
    """A diagonal fuel container with both cardinal stepping-stones blocked is skipped.

    ``find_adjacent_container`` is used by the combat secondary-pickup
    path. Diagonal pickups require stepping through one of the two
    cardinal intermediates first. If both intermediates and the target
    tile are blocked, the pickup is not collectable from the current
    tile and the candidate is dropped.
    """
    world, self_state = _world_and_self(x=100, y=100)
    world["containers"]["101,101"] = make_container_state(
        x=101,
        y=101,
        is_fuel=True,
        volume=300,
        timestamp_ms=100000,
        failed_pickups=0,
    )
    terrain = InMemoryTerrainMap(
        terrain_data={
            (101, 100): "W",
            (100, 101): "W",
            (101, 101): "W",
            (102, 101): "W",
            (101, 102): "W",
        },
    )

    assert find_adjacent_container(world, self_state, terrain, want_fuel=True) is None
