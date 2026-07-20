"""Tests for ferry-aware terrain composition and surface clamping."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.ferry import (
    FerryAwareTerrain,
    clamp_move_target_at_surface_transition,
    compose_decision_terrain,
    is_riding_ferry,
)
from tankpit_bot.state.types import (
    TERRAIN_FERRY,
    TERRAIN_GROUND,
    TerrainTileDict,
    make_terrain_tile,
)
from tests.bot.ai._support import make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _ferry_tile(x: int, y: int) -> dict[str, TerrainTileDict]:
    """Build the wire terrain entry for a ferry at a coordinate.

    Args:
        x: Tile X coordinate.
        y: Tile Y coordinate.

    Returns:
        Single-entry wire-terrain mapping fragment.
    """
    return {
        f"{x},{y}": make_terrain_tile(
            x=x,
            y=y,
            terrain_type=TERRAIN_FERRY,
        )
    }


class TestFerryAwareTerrain:
    """Tests for the composed terrain view."""

    def test_ferry_tile_overlays_static_water(self) -> None:
        """A live ferry tile renders and passes over static water."""
        base = InMemoryTerrainMap({(100, 100): "W"})
        wire = _ferry_tile(100, 100)
        terrain = FerryAwareTerrain(base, wire, riding=False)

        assert terrain.get_terrain(100, 100) == "~"
        assert terrain.is_passable(100, 100) is True

    def test_water_is_passable_only_while_riding(self) -> None:
        """Open water flips passable exactly with the riding flag."""
        base = InMemoryTerrainMap({(101, 100): "W"})

        assert FerryAwareTerrain(base, {}, riding=True).is_passable(101, 100) is True
        assert FerryAwareTerrain(base, {}, riding=False).is_passable(101, 100) is False

    def test_ground_always_passable_and_rock_never(self) -> None:
        """Static ground and rock keep their semantics in both modes."""
        base = InMemoryTerrainMap({(102, 100): "#"})

        for riding in (False, True):
            terrain = FerryAwareTerrain(base, {}, riding=riding)
            assert terrain.is_passable(100, 100) is True
            assert terrain.is_passable(102, 100) is False

    def test_render_viewport_includes_live_ferry(self) -> None:
        """The rendered grid carries the ferry overlay."""
        base = InMemoryTerrainMap({(100, 100): "W"})
        terrain = FerryAwareTerrain(base, _ferry_tile(100, 100), riding=True)

        grid = terrain.render_viewport(100, 100, width=3, height=3)

        assert grid[1][1] == "~"
        assert grid[0][0] == "."


class TestRidingAndComposition:
    """Tests for is_riding_ferry and compose_decision_terrain."""

    def test_riding_when_self_tile_is_ferry(self) -> None:
        """The wire tile under the tank decides the riding flag."""
        world, _self_state = make_world(self_x=100, self_y=100)
        world["terrain"].update(_ferry_tile(100, 100))

        assert is_riding_ferry(world) is True

    def test_not_riding_on_ground_or_unknown_tile(self) -> None:
        """A ground wire tile or no wire tile means not riding."""
        world, _self_state = make_world(self_x=100, self_y=100)
        assert is_riding_ferry(world) is False

        world["terrain"]["100,100"] = make_terrain_tile(
            x=100,
            y=100,
            terrain_type=TERRAIN_GROUND,
        )
        assert is_riding_ferry(world) is False

    def test_not_riding_without_self_state(self) -> None:
        """A world with no self tank cannot be riding."""
        world, _self_state = make_world(self_x=100, self_y=100)
        world["self_state"] = None

        assert is_riding_ferry(world) is False

    def test_compose_passes_through_missing_static_map(self) -> None:
        """Without a static map there is nothing to compose."""
        world, _self_state = make_world()

        assert compose_decision_terrain(world, None) is None

    def test_compose_builds_riding_view(self) -> None:
        """Composition carries the riding flag into water passability."""
        world, _self_state = make_world(self_x=100, self_y=100)
        world["terrain"].update(_ferry_tile(100, 100))
        base = InMemoryTerrainMap({(101, 100): "W"})

        composed = compose_decision_terrain(world, base)
        if composed is None:
            pytest.fail("expected composed terrain from ferry + base map")

        assert composed.is_passable(101, 100) is True


class TestSurfaceTransitionClamp:
    """Tests for clamp_move_target_at_surface_transition."""

    def test_boarding_clamps_at_ferry_tile(self) -> None:
        """A land walk onto a ferry stops at the ferry tile."""
        world, _self_state = make_world(self_x=100, self_y=100)
        base = InMemoryTerrainMap({(103, 100): "W", (104, 100): "W"})
        wire = _ferry_tile(102, 100)
        terrain = FerryAwareTerrain(base, wire, riding=False)

        clamped = clamp_move_target_at_surface_transition(
            world,
            terrain,
            100,
            100,
            102,
            100,
            {},
        )

        assert clamped == (102, 100)

    def test_disembark_clamps_at_first_land_tile(self) -> None:
        """A ride across water toward inland stops one step onto land."""
        world, _self_state = make_world(self_x=100, self_y=100)
        base = InMemoryTerrainMap(
            {(100, 100): "W", (101, 100): "W", (102, 100): "W"},
        )
        terrain = FerryAwareTerrain(base, _ferry_tile(100, 100), riding=True)

        clamped = clamp_move_target_at_surface_transition(
            world,
            terrain,
            100,
            100,
            105,
            100,
            {},
        )

        assert clamped == (103, 100)

    def test_pure_land_path_keeps_target(self) -> None:
        """A path that never changes surface class is not clamped."""
        world, _self_state = make_world(self_x=100, self_y=100)
        terrain = FerryAwareTerrain(InMemoryTerrainMap(), {}, riding=False)

        clamped = clamp_move_target_at_surface_transition(
            world,
            terrain,
            100,
            100,
            105,
            100,
            {},
        )

        assert clamped == (105, 100)

    def test_pure_water_ride_keeps_target(self) -> None:
        """Riding across open water to a water tile is one command."""
        world, _self_state = make_world(self_x=100, self_y=100)
        data = {(x, 100): "W" for x in range(100, 106)}
        terrain = FerryAwareTerrain(InMemoryTerrainMap(data), _ferry_tile(100, 100), riding=True)

        clamped = clamp_move_target_at_surface_transition(
            world,
            terrain,
            100,
            100,
            105,
            100,
            {},
        )

        assert clamped == (105, 100)


class TestGroundOnlyTerrain:
    """The single-action pickup-routing surface (user contract 2026-07-19)."""

    def test_only_plain_ground_is_traversable(self) -> None:
        """Ground passes; water, rock, and live ferry tiles all block.

        One server-routed command never chains surfaces: crossing onto
        a ferry is a queue-consuming boarding action, so a pickup route
        may not traverse it even though the riding view calls it
        passable.
        """
        from tankpit_bot.bot.ai.ferry import GroundOnlyTerrain

        base = InMemoryTerrainMap({(101, 100): "W", (102, 100): "#"})
        ferry_view = FerryAwareTerrain(base, _ferry_tile(103, 100), riding=True)
        ground_only = GroundOnlyTerrain(ferry_view)

        assert ground_only.is_passable(100, 100) is True  # plain ground
        assert ground_only.is_passable(101, 100) is False  # water (even riding)
        assert ground_only.is_passable(102, 100) is False  # rock
        assert ground_only.is_passable(103, 100) is False  # live ferry tile
        # Cell display is unchanged -- only passability differs.
        assert ground_only.get_terrain(103, 100) == "~"
        assert ground_only.get_terrain(101, 100) == "W"

    def test_render_viewport_delegates_to_wrapped_view(self) -> None:
        """The rendering is the wrapped view's rendering, verbatim."""
        from tankpit_bot.bot.ai.ferry import GroundOnlyTerrain

        base = InMemoryTerrainMap({(100, 100): "W"})
        view = FerryAwareTerrain(base, _ferry_tile(100, 100), riding=False)
        assert GroundOnlyTerrain(view).render_viewport(100, 100, 1, 1) == view.render_viewport(
            100, 100, 1, 1
        )
