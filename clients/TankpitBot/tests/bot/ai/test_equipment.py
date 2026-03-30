"""Tests for AI equipment and container targeting."""

from __future__ import annotations

from tankpit_bot.bot.ai.equipment import (
    describe_container_search,
    find_best_fuel,
    find_nearest_deposit,
    find_nearest_equipment,
    find_nearest_fuel,
    find_teleport_landing_tile,
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


class TestFindTeleportLandingTile:
    """Tests for blocked-target teleport landing selection."""

    def test_returns_nearest_passable_adjacent_tile(self) -> None:
        """Prefers the closest passable cardinal tile next to the target."""
        terrain_data: dict[tuple[int, int], str] = {
            (128, 126): "W",
            (127, 126): "W",
            (128, 127): "W",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)

        result = find_teleport_landing_tile(terrain, 130, 124, 128, 126)

        assert result == (129, 126)

    def test_returns_none_when_all_adjacent_tiles_blocked(self) -> None:
        """Returns None when no passable cardinal landing tile exists."""
        terrain_data: dict[tuple[int, int], str] = {
            (128, 126): "W",
            (129, 126): "W",
            (127, 126): "W",
            (128, 127): "#",
            (128, 125): "#",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)

        result = find_teleport_landing_tile(terrain, 130, 124, 128, 126)

        assert result is None

    def test_skips_out_of_bounds_adjacent_tiles(self) -> None:
        """Ignores adjacent coordinates that fall outside the map bounds."""
        terrain_data: dict[tuple[int, int], str] = {
            (0, 1): "W",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)

        result = find_teleport_landing_tile(terrain, 10, 10, 0, 0)

        assert result == (1, 0)


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

    def test_with_terrain_allow_unreachable_keeps_blocked_target(self) -> None:
        """allow_unreachable=True returns a blocked fuel container for teleport fallback."""
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = FakeTerrainMap(terrain_data=wall)
        expected = make_container_state(x=15, y=10, is_fuel=True, volume=50)
        world["containers"]["15,10"] = expected
        assert find_nearest_fuel(world, state, terrain, allow_unreachable=True) == expected

    def test_with_terrain_allow_unreachable_skips_target_without_landing_tile(self) -> None:
        """allow_unreachable=True still rejects a blocked fuel target with no landing tile."""
        world, state = _world_and_self(x=10, y=10)
        terrain_data: dict[tuple[int, int], str] = {
            (15, 10): "W",
            (16, 10): "W",
            (14, 10): "W",
            (15, 11): "#",
            (15, 9): "#",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=True, volume=50)
        assert find_nearest_fuel(world, state, terrain, allow_unreachable=True) is None

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

    def test_skips_stale_containers(self) -> None:
        """find_nearest_fuel skips containers older than freshness TTL."""
        world, state = _world_and_self()
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=True,
            volume=500,
            timestamp_ms=10000,
        )
        fresh = make_container_state(
            x=103,
            y=100,
            is_fuel=True,
            volume=300,
            timestamp_ms=90000,
        )
        world["containers"]["103,100"] = fresh
        assert find_nearest_fuel(world, state, now_ms=100000) == fresh

    def test_freshness_disabled_when_now_ms_zero(self) -> None:
        """find_nearest_fuel skips freshness check when now_ms=0."""
        world, state = _world_and_self()
        old = make_container_state(
            x=101,
            y=100,
            is_fuel=True,
            volume=500,
            timestamp_ms=0,
        )
        world["containers"]["101,100"] = old
        assert find_nearest_fuel(world, state, now_ms=0) == old


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

    def test_skips_stale_equipment(self) -> None:
        """find_nearest_equipment skips containers older than freshness TTL."""
        world, state = _world_and_self()
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=10000,
        )
        fresh = make_container_state(
            x=103,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=90000,
        )
        world["containers"]["103,100"] = fresh
        assert find_nearest_equipment(world, state, now_ms=100000) == fresh


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
            allow_unreachable=True,
        )

        assert result == (
            "equipment: total=1 nearby=1 actionable=1 blocked=0 "
            "no_landing=0 low_volume=0 nearest=(101,100) actionable"
        )

    def test_reports_blocked_equipment_without_landing(self) -> None:
        """Summary explains when nearby equipment has no valid teleport landing tile."""
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
        terrain = FakeTerrainMap(terrain_data=terrain_data)

        result = describe_container_search(
            world,
            state,
            terrain,
            want_fuel=False,
            allow_unreachable=True,
        )

        assert result == (
            "equipment: total=1 nearby=1 actionable=0 blocked=1 "
            "no_landing=1 low_volume=0 nearest=(15,10) blocked_no_landing"
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
            allow_unreachable=True,
            minimum_volume=100,
        )

        assert result == (
            "fuel: total=1 nearby=1 actionable=0 blocked=0 "
            "no_landing=0 low_volume=1 nearest=(101,100) low_volume"
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
            allow_unreachable=True,
        )

        assert result == (
            "equipment: total=1 nearby=0 actionable=0 blocked=0 "
            "no_landing=0 low_volume=0 nearest=none"
        )

    def test_reports_blocked_walk_when_unreachable_targets_disallowed(self) -> None:
        """Summary marks blocked targets as non-actionable without teleport fallback."""
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = FakeTerrainMap(terrain_data=wall)
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
            allow_unreachable=False,
        )

        assert result == (
            "equipment: total=1 nearby=1 actionable=0 blocked=1 "
            "no_landing=0 low_volume=0 nearest=(15,10) blocked_walk"
        )

    def test_reports_blocked_target_as_actionable_when_landing_tile_exists(self) -> None:
        """Summary keeps blocked targets actionable when teleport landing is available."""
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain_data: dict[tuple[int, int], str] = dict(wall)
        terrain_data[(15, 10)] = "W"
        terrain = FakeTerrainMap(terrain_data=terrain_data)
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
            allow_unreachable=True,
        )

        assert result == (
            "equipment: total=1 nearby=1 actionable=1 blocked=1 "
            "no_landing=0 low_volume=0 nearest=(15,10) actionable"
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
            allow_unreachable=True,
        )

        assert result == (
            "equipment: total=2 nearby=2 actionable=2 blocked=0 "
            "no_landing=0 low_volume=0 nearest=(101,100) actionable"
        )

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

    def test_with_terrain_allow_unreachable_keeps_blocked_target(self) -> None:
        """allow_unreachable=True returns a blocked equipment container for teleport fallback."""
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = FakeTerrainMap(terrain_data=wall)
        expected = make_container_state(x=15, y=10, is_fuel=False, volume=0)
        world["containers"]["15,10"] = expected
        assert find_nearest_equipment(world, state, terrain, allow_unreachable=True) == expected


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
        expected = make_container_state(x=105, y=100, is_fuel=True, volume=50)
        world["containers"]["105,100"] = expected
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
        terrain = FakeTerrainMap(terrain_data=wall)
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

    def test_with_terrain_allow_unreachable_prefers_blocked_high_value_target(self) -> None:
        """allow_unreachable=True keeps a blocked high-value fuel target for teleport fallback."""
        world, state = _world_and_self(x=10, y=10)
        wall: dict[tuple[int, int], str] = {(12, y): "#" for y in range(256)}
        terrain = FakeTerrainMap(terrain_data=wall)
        expected = make_container_state(x=15, y=10, is_fuel=True, volume=1000)
        world["containers"]["15,10"] = expected
        world["containers"]["8,10"] = make_container_state(x=8, y=10, is_fuel=True, volume=600)
        assert find_best_fuel(world, state, terrain, allow_unreachable=True) == expected

    def test_with_terrain_allow_unreachable_skips_blocked_fuel_without_landing_tile(self) -> None:
        """allow_unreachable=True rejects blocked fuel when no landing tile exists."""
        world, state = _world_and_self(x=10, y=10)
        terrain_data: dict[tuple[int, int], str] = {
            (15, 10): "W",
            (16, 10): "W",
            (14, 10): "W",
            (15, 11): "#",
            (15, 9): "#",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)
        world["containers"]["15,10"] = make_container_state(x=15, y=10, is_fuel=True, volume=1000)
        assert find_best_fuel(world, state, terrain, allow_unreachable=True) is None

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

    def test_skips_stale_fuel(self) -> None:
        """find_best_fuel skips containers older than freshness TTL."""
        world, state = _world_and_self()
        world["containers"]["101,100"] = make_container_state(
            x=101,
            y=100,
            is_fuel=True,
            volume=900,
            timestamp_ms=10000,
        )
        fresh = make_container_state(
            x=105,
            y=100,
            is_fuel=True,
            volume=500,
            timestamp_ms=90000,
        )
        world["containers"]["105,100"] = fresh
        assert find_best_fuel(world, state, now_ms=100000) == fresh
