"""Tests for AI equipment and container targeting."""

from __future__ import annotations

from tankpit_bot.bot.ai.equipment import (
    find_best_fuel,
    find_nearest_deposit,
    find_nearest_equipment,
    find_nearest_fuel,
    is_reachable,
)
from tankpit_bot.state.types import (
    SelfStateDict,
    ViewportStateDict,
    WorldStateDict,
    make_container_state,
    make_self_state,
)
from tests.fakes import FakeTerrainMap


def _world_and_self(x: int = 100, y: int = 100) -> tuple[WorldStateDict, SelfStateDict]:
    """Create empty world and self state at given position.

    Args:
        x: Self X coordinate.
        y: Self Y coordinate.

    Returns:
        Tuple of (empty WorldStateDict, SelfStateDict).
    """
    world = WorldStateDict(
        self_state=None,
        tanks={},
        containers={},
        mines={},
        terrain={},
        viewport=ViewportStateDict(left=0, top=0, width=18, height=18),
        timestamp_ms=0,
    )
    state = make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=0,
        rank=4,
        fuel=800,
        leaderboard_position=1,
    )
    return world, state


# =============================================================================
# is_reachable
# =============================================================================


class TestIsReachable:
    """Tests for is_reachable helper."""

    def test_reachable_open_terrain(self) -> None:
        """Returns True when path exists on open ground."""
        terrain = FakeTerrainMap()
        assert is_reachable(terrain, 10, 10, 15, 10) is True

    def test_same_position(self) -> None:
        """Returns True when start equals goal."""
        terrain = FakeTerrainMap()
        assert is_reachable(terrain, 10, 10, 10, 10) is True

    def test_blocked_by_wall(self) -> None:
        """Returns False when terrain blocks all paths."""
        # Create a wall of rocks from y=0 to y=255 at x=12
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = FakeTerrainMap(terrain_data=wall)
        assert is_reachable(terrain, 10, 10, 15, 10) is False


# =============================================================================
# find_nearest_fuel
# =============================================================================


class TestFindNearestFuel:
    """Tests for find_nearest_fuel."""

    def test_no_containers(self) -> None:
        """Returns None when no containers exist."""
        world, state = _world_and_self()
        assert find_nearest_fuel(world, state) is None

    def test_no_fuel_containers(self) -> None:
        """Returns None when only equipment containers exist."""
        world, state = _world_and_self()
        world["containers"]["110,100"] = make_container_state(
            x=110,
            y=100,
            is_fuel=False,
            volume=0,
        )
        assert find_nearest_fuel(world, state) is None

    def test_single_fuel_container(self) -> None:
        """Returns the only fuel container."""
        world, state = _world_and_self()
        expected = make_container_state(x=110, y=100, is_fuel=True, volume=50)
        world["containers"]["110,100"] = expected
        assert find_nearest_fuel(world, state) == expected

    def test_nearest_of_multiple(self) -> None:
        """Returns the closest fuel container."""
        world, state = _world_and_self()
        world["containers"]["150,100"] = make_container_state(
            x=150,
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

    def test_ignores_equipment(self) -> None:
        """Fuel search skips equipment containers even if closer."""
        world, state = _world_and_self()
        world["containers"]["102,100"] = make_container_state(
            x=102,
            y=100,
            is_fuel=False,
            volume=0,
        )
        expected = make_container_state(x=120, y=100, is_fuel=True, volume=50)
        world["containers"]["120,100"] = expected
        assert find_nearest_fuel(world, state) == expected

    def test_with_terrain_skips_unreachable(self) -> None:
        """Skips fuel container that is unreachable due to terrain."""
        world, state = _world_and_self(x=10, y=10)
        # Container blocked behind wall
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = FakeTerrainMap(terrain_data=wall)
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=True, volume=50)
        assert find_nearest_fuel(world, state, terrain) is None

    def test_with_terrain_accepts_reachable(self) -> None:
        """Accepts fuel container that is reachable through terrain."""
        world, state = _world_and_self(x=10, y=10)
        terrain = FakeTerrainMap()
        expected = make_container_state(x=15, y=10, is_fuel=True, volume=50)
        world["containers"]["15,10"] = expected
        assert find_nearest_fuel(world, state, terrain) == expected

    def test_with_terrain_skips_blocked_picks_reachable(self) -> None:
        """Skips closer blocked container, picks farther reachable one."""
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = FakeTerrainMap(terrain_data=wall)
        # Closer but blocked
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=True, volume=50)
        # Farther but reachable (same side of wall)
        expected = make_container_state(x=8, y=10, is_fuel=True, volume=30)
        world["containers"]["8,10"] = expected
        assert find_nearest_fuel(world, state, terrain) == expected


# =============================================================================
# find_nearest_equipment
# =============================================================================


class TestFindNearestEquipment:
    """Tests for find_nearest_equipment."""

    def test_no_containers(self) -> None:
        """Returns None when no containers exist."""
        world, state = _world_and_self()
        assert find_nearest_equipment(world, state) is None

    def test_no_equipment_containers(self) -> None:
        """Returns None when only fuel containers exist."""
        world, state = _world_and_self()
        world["containers"]["110,100"] = make_container_state(
            x=110,
            y=100,
            is_fuel=True,
            volume=50,
        )
        assert find_nearest_equipment(world, state) is None

    def test_single_equipment_container(self) -> None:
        """Returns the only equipment container."""
        world, state = _world_and_self()
        expected = make_container_state(x=115, y=100, is_fuel=False, volume=0)
        world["containers"]["115,100"] = expected
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

    def test_ignores_fuel(self) -> None:
        """Equipment search skips fuel containers even if closer."""
        world, state = _world_and_self()
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=True,
            volume=80,
        )
        expected = make_container_state(x=125, y=100, is_fuel=False, volume=0)
        world["containers"]["125,100"] = expected
        assert find_nearest_equipment(world, state) == expected

    def test_with_terrain_skips_unreachable(self) -> None:
        """Skips equipment container that is unreachable due to terrain."""
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = FakeTerrainMap(terrain_data=wall)
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=False, volume=0)
        assert find_nearest_equipment(world, state, terrain) is None

    def test_with_terrain_accepts_reachable(self) -> None:
        """Accepts equipment container that is reachable through terrain."""
        world, state = _world_and_self(x=10, y=10)
        terrain = FakeTerrainMap()
        expected = make_container_state(x=15, y=10, is_fuel=False, volume=0)
        world["containers"]["15,10"] = expected
        assert find_nearest_equipment(world, state, terrain) == expected


# =============================================================================
# find_nearest_deposit
# =============================================================================


class TestFindNearestDeposit:
    """Tests for find_nearest_deposit."""

    def test_no_containers(self) -> None:
        """Returns None when no containers exist."""
        world, state = _world_and_self()
        assert find_nearest_deposit(world, state) is None

    def test_finds_fuel_container(self) -> None:
        """Returns nearest fuel container for depositing."""
        world, state = _world_and_self()
        expected = make_container_state(x=110, y=100, is_fuel=True, volume=50)
        world["containers"]["110,100"] = expected
        assert find_nearest_deposit(world, state) == expected

    def test_with_terrain(self) -> None:
        """Deposit search respects terrain reachability."""
        world, state = _world_and_self(x=10, y=10)
        terrain = FakeTerrainMap()
        expected = make_container_state(x=15, y=10, is_fuel=True, volume=50)
        world["containers"]["15,10"] = expected
        assert find_nearest_deposit(world, state, terrain) == expected


# =============================================================================
# find_best_fuel
# =============================================================================


class TestFindBestFuel:
    """Tests for find_best_fuel (volume-prioritized fuel selection)."""

    def test_no_containers(self) -> None:
        """Returns None when no containers exist."""
        world, state = _world_and_self()
        assert find_best_fuel(world, state) is None

    def test_single_fuel_container(self) -> None:
        """Returns the only fuel container."""
        world, state = _world_and_self()
        expected = make_container_state(x=110, y=100, is_fuel=True, volume=500)
        world["containers"]["110,100"] = expected
        assert find_best_fuel(world, state) == expected

    def test_prefers_higher_volume_over_proximity(self) -> None:
        """Picks farther high-volume container over closer low-volume."""
        world, state = _world_and_self()
        # Far but high volume inserted first (dist=50, vol=1000, score=950)
        expected = make_container_state(x=150, y=100, is_fuel=True, volume=1000)
        world["containers"]["150,100"] = expected
        # Close but low volume inserted second (dist=5, vol=100, score=95)
        # Exercises the score <= best_score branch (95 < 950)
        world["containers"]["105,100"] = make_container_state(
            x=105, y=100, is_fuel=True, volume=100
        )
        assert find_best_fuel(world, state) == expected

    def test_ignores_equipment(self) -> None:
        """Skips equipment containers."""
        world, state = _world_and_self()
        world["containers"]["102,100"] = make_container_state(x=102, y=100, is_fuel=False, volume=0)
        expected = make_container_state(x=120, y=100, is_fuel=True, volume=300)
        world["containers"]["120,100"] = expected
        assert find_best_fuel(world, state) == expected

    def test_with_terrain_skips_unreachable(self) -> None:
        """Skips high-volume container that is unreachable."""
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = FakeTerrainMap(terrain_data=wall)
        # High volume but blocked
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=True, volume=1000)
        # Low volume but reachable
        expected = make_container_state(x=8, y=10, is_fuel=True, volume=200)
        world["containers"]["8,10"] = expected
        assert find_best_fuel(world, state, terrain) == expected
