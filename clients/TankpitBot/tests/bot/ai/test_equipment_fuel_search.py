"""Tests for fuel container search and selection."""

from __future__ import annotations

from tankpit_bot.bot.ai.equipment_search import (
    find_best_fuel,
    find_nearest_fuel,
)
from tankpit_bot.state.types import (
    make_container_state,
    make_viewport_state,
)
from tests.bot.ai._equipment_fixtures import _world_and_self
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestFindNearestFuel:
    """Tests for find_nearest_fuel."""

    def test_no_containers(self) -> None:
        """Returns None when no containers exist."""
        world, state = _world_and_self()
        assert find_nearest_fuel(world, state) is None

    def test_no_fuel_containers(self) -> None:
        """Returns None when only equipment containers exist."""
        world, state = _world_and_self()
        world["containers"]["105,100"] = make_container_state(
            x=105,
            y=100,
            is_fuel=False,
            volume=0,
        )
        assert find_nearest_fuel(world, state) is None

    def test_single_fuel_container(self) -> None:
        """Returns the only fuel container."""
        world, state = _world_and_self()
        expected = make_container_state(x=105, y=100, is_fuel=True, volume=50)
        world["containers"]["105,100"] = expected
        assert find_nearest_fuel(world, state) == expected

    def test_nearest_of_multiple(self) -> None:
        """Returns the closest fuel container."""
        world, state = _world_and_self()
        world["containers"]["108,100"] = make_container_state(
            x=108,
            y=100,
            is_fuel=True,
            volume=50,
        )
        closest = make_container_state(x=105, y=100, is_fuel=True, volume=30)
        world["containers"]["105,100"] = closest
        world["containers"]["130,100"] = make_container_state(
            x=130,
            y=100,
            is_fuel=True,
            volume=80,
        )
        assert find_nearest_fuel(world, state) == closest

    def test_farther_visible_fuel_does_not_replace_closer_target(self) -> None:
        """Keeps the closest visible fuel when a farther visible one is checked later."""
        world, state = _world_and_self()
        closest = make_container_state(x=103, y=100, is_fuel=True, volume=30)
        world["containers"]["103,100"] = closest
        world["containers"]["107,100"] = make_container_state(
            x=107,
            y=100,
            is_fuel=True,
            volume=80,
        )
        assert find_nearest_fuel(world, state) == closest

    def test_ignores_equipment(self) -> None:
        """Fuel search skips equipment containers even if closer."""
        world, state = _world_and_self()
        world["containers"]["102,100"] = make_container_state(
            x=102,
            y=100,
            is_fuel=False,
            volume=0,
        )
        expected = make_container_state(x=107, y=100, is_fuel=True, volume=50)
        world["containers"]["107,100"] = expected
        assert find_nearest_fuel(world, state) == expected

    def test_with_terrain_skips_unreachable(self) -> None:
        """Skips fuel container that is unreachable due to terrain."""
        world, state = _world_and_self(x=10, y=10)
        # Container blocked behind wall
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = InMemoryTerrainMap(terrain_data=wall)
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=True, volume=50)
        assert find_nearest_fuel(world, state, terrain) is None

    def test_with_terrain_accepts_reachable(self) -> None:
        """Accepts fuel container that is reachable through terrain."""
        world, state = _world_and_self(x=10, y=10)
        terrain = InMemoryTerrainMap()
        expected = make_container_state(x=15, y=10, is_fuel=True, volume=50)
        world["containers"]["15,10"] = expected
        assert find_nearest_fuel(world, state, terrain) == expected

    def test_with_terrain_skips_wall_blocked_fuel(self) -> None:
        """Wall-blocked fuel is skipped -- the bot can't walk there.

        Per the 2026-06-26 user contract, containers are picked only when
        a walk path exists in the current viewport. Teleport-to-container
        was removed: a blocked container is not actionable, full stop.
        """
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = InMemoryTerrainMap(terrain_data=wall)
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=True, volume=50)
        assert find_nearest_fuel(world, state, terrain) is None

    def test_with_terrain_skips_water_locked_fuel(self) -> None:
        """Water-locked fuel with no land-neighbor approach is skipped."""
        world, state = _world_and_self(x=10, y=10)
        terrain_data: dict[tuple[int, int], str] = {
            (15, 10): "W",
            (16, 10): "W",
            (14, 10): "W",
            (15, 11): "#",
            (15, 9): "#",
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=True, volume=50)
        assert find_nearest_fuel(world, state, terrain) is None

    def test_with_terrain_skips_blocked_picks_reachable(self) -> None:
        """Skips closer blocked container, picks farther reachable one."""
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = InMemoryTerrainMap(terrain_data=wall)
        # Closer but blocked
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=True, volume=50)
        # Farther but reachable (same side of wall)
        expected = make_container_state(x=8, y=10, is_fuel=True, volume=30)
        world["containers"]["8,10"] = expected
        assert find_nearest_fuel(world, state, terrain) == expected

    def test_skips_failed_pickup_containers(self) -> None:
        """find_nearest_fuel skips containers with failed_pickups > 0."""
        world, state = _world_and_self()
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=True,
            volume=500,
            failed_pickups=1,
        )
        farther = make_container_state(
            x=103,
            y=100,
            is_fuel=True,
            volume=300,
        )
        world["containers"]["103,100"] = farther
        assert find_nearest_fuel(world, state) == farther


class TestFindNearestFuelExtras:
    """Edge cases for find_nearest_fuel: viewport bounds, freshness, scan state."""

    def test_uses_viewport_bounds_not_distance_from_self(self) -> None:
        """Visible containers at the far viewport edge are still eligible."""
        world, state = _world_and_self(x=91, y=100)
        world["viewport"] = make_viewport_state(left=90, top=91, width=18, height=18)
        expected = make_container_state(x=107, y=100, is_fuel=True, volume=300)
        world["containers"]["107,100"] = expected

        assert find_nearest_fuel(world, state) == expected

    def test_age_does_not_filter_containers(self) -> None:
        """find_nearest_fuel ignores container age.

        The 30 s freshness TTL was removed 2026-07-06: in-viewport
        containers are wire-truthful under the truth layer, so the
        nearer container wins regardless of timestamp.
        """
        world, state = _world_and_self()
        old = make_container_state(
            x=101,
            y=100,
            is_fuel=True,
            volume=500,
            timestamp_ms=10000,
        )
        world["containers"]["101,100"] = old
        world["containers"]["103,100"] = make_container_state(
            x=103,
            y=100,
            is_fuel=True,
            volume=300,
            timestamp_ms=90000,
        )
        world["timestamp_ms"] = 100000
        assert find_nearest_fuel(world, state) == old

    def test_unscanned_viewport_does_not_change_raw_fuel_selection(self) -> None:
        """find_nearest_fuel stays a pure viewport selector without radar policy."""
        world, state = _world_and_self()
        world["scanned_tiles"] = {}
        expected = make_container_state(
            x=101,
            y=100,
            is_fuel=True,
            volume=500,
        )
        world["containers"]["101,100"] = expected

        assert find_nearest_fuel(world, state) == expected


class TestFindBestFuel:
    """Tests for find_best_fuel (volume-prioritized fuel selection)."""

    def test_no_containers(self) -> None:
        """Returns None when no containers exist."""
        world, state = _world_and_self()
        assert find_best_fuel(world, state) is None

    def test_single_fuel_container(self) -> None:
        """Returns the only fuel container."""
        world, state = _world_and_self()
        expected = make_container_state(x=105, y=100, is_fuel=True, volume=500)
        world["containers"]["105,100"] = expected
        assert find_best_fuel(world, state) == expected

    def test_prefers_higher_volume_over_proximity(self) -> None:
        """Picks farther high-volume container over closer low-volume."""
        world, state = _world_and_self()
        # Far but high volume inserted first (dist=50, vol=1000, score=950)
        expected = make_container_state(x=108, y=100, is_fuel=True, volume=1000)
        world["containers"]["108,100"] = expected
        # Close but lower volume inserted second (dist=5, vol=600, score=-5)
        # Exercises the score <= best_score branch
        world["containers"]["105,100"] = make_container_state(
            x=105, y=100, is_fuel=True, volume=600
        )
        assert find_best_fuel(world, state) == expected

    def test_ignores_equipment(self) -> None:
        """Skips equipment containers."""
        world, state = _world_and_self()
        world["containers"]["102,100"] = make_container_state(x=102, y=100, is_fuel=False, volume=0)
        expected = make_container_state(x=107, y=100, is_fuel=True, volume=700)
        world["containers"]["107,100"] = expected
        assert find_best_fuel(world, state) == expected

    def test_with_terrain_skips_unreachable(self) -> None:
        """Skips high-volume container that is unreachable."""
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = InMemoryTerrainMap(terrain_data=wall)
        # High volume but blocked
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=True, volume=1000)
        # Lower volume but reachable (still >= 500 minimum)
        expected = make_container_state(x=8, y=10, is_fuel=True, volume=600)
        world["containers"]["8,10"] = expected
        assert find_best_fuel(world, state, terrain) == expected

    def test_skips_near_empty_fuel_container(self) -> None:
        """Ignores fuel containers below the minimum useful volume."""
        world, state = _world_and_self()
        world["containers"]["103,100"] = make_container_state(
            x=103,
            y=100,
            is_fuel=True,
            volume=50,
        )
        expected = make_container_state(x=104, y=100, is_fuel=True, volume=400)
        world["containers"]["104,100"] = expected
        assert find_best_fuel(world, state) == expected

    def test_skips_out_of_viewport_fuel_container(self) -> None:
        """Ignores fuel containers outside the walkable viewport window."""
        world, state = _world_and_self()
        world["containers"]["109,100"] = make_container_state(
            x=109,
            y=100,
            is_fuel=True,
            volume=1000,
        )
        expected = make_container_state(x=108, y=100, is_fuel=True, volume=600)
        world["containers"]["108,100"] = expected
        assert find_best_fuel(world, state) == expected

    def test_with_terrain_skips_blocked_picks_reachable_lower_value(self) -> None:
        """Wall-blocked high-value fuel is skipped for the reachable lower-value one.

        Walk-only contract (2026-06-26): a blocked container is not
        actionable regardless of its volume. The lower-volume reachable
        target wins.
        """
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = InMemoryTerrainMap(terrain_data=wall)
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=True, volume=1000)
        expected = make_container_state(x=8, y=10, is_fuel=True, volume=600)
        world["containers"]["8,10"] = expected
        assert find_best_fuel(world, state, terrain) == expected

    def test_with_terrain_skips_water_locked_high_value_fuel(self) -> None:
        """Water-locked fuel with no land-neighbor is skipped regardless of volume."""
        world, state = _world_and_self(x=10, y=10)
        terrain_data: dict[tuple[int, int], str] = {
            (15, 10): "W",
            (16, 10): "W",
            (14, 10): "W",
            (15, 11): "#",
            (15, 9): "#",
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=True, volume=1000)
        assert find_best_fuel(world, state, terrain) is None

    def test_skips_failed_pickup_fuel(self) -> None:
        """find_best_fuel skips containers with failed_pickups > 0."""
        world, state = _world_and_self()
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=True,
            volume=900,
            failed_pickups=1,
        )
        farther = make_container_state(
            x=105,
            y=100,
            is_fuel=True,
            volume=500,
        )
        world["containers"]["105,100"] = farther
        assert find_best_fuel(world, state) == farther

    def test_age_does_not_filter_best_fuel(self) -> None:
        """find_best_fuel ignores container age (TTL removed 2026-07-06)."""
        world, state = _world_and_self()
        old = make_container_state(
            x=101,
            y=100,
            is_fuel=True,
            volume=900,
            timestamp_ms=10000,
        )
        world["containers"]["101,100"] = old
        world["containers"]["105,100"] = make_container_state(
            x=105,
            y=100,
            is_fuel=True,
            volume=500,
            timestamp_ms=90000,
        )
        world["timestamp_ms"] = 100000
        assert find_best_fuel(world, state) == old
