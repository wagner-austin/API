"""Tests for ferry-aware terrain composition and surface clamping."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.ferry import (
    FerryAwareTerrain,
    clamp_move_target_at_surface_transition,
    compose_decision_terrain,
    is_riding_ferry,
)
from tankpit_bot.bot.ai.recover_fuel_mode import decide_recover_fuel_mode
from tankpit_bot.state.types import (
    TERRAIN_FERRY,
    TERRAIN_GROUND,
    TerrainTileDict,
    make_terrain_tile,
)
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
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
            cache_value=0,
            overlay_value=0,
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
            cache_value=0,
            overlay_value=0,
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


def test_tank_on_ferry_walks_across_water_to_fuel_dot() -> None:
    """The stranded-on-ferry tank sails to the dot across the lake.

    Regression guard for live run 20260612-131003: the tank stood on a
    ferry at 7 fuel with a fuel dot two tiles away across water and
    idled for 28 minutes because the walkability model treated all
    water as impassable.
    """
    world, self_state = make_world(self_x=100, self_y=100, fuel=7, scanned=False)
    world["terrain"].update(_ferry_tile(100, 100))
    world["map_fuel_dots"] = {"104,100": 100000}
    lake = {(x, y): "W" for x in range(96, 108) for y in range(96, 105)}
    base = InMemoryTerrainMap(lake)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        compose_decision_terrain(world, base),
        "",
    )

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["reason"] == "fuel_dot_walk"
    assert decision["command"]["cmd_type"] == "move"
    assert decision["command"]["target_x"] == 104
    assert decision["command"]["target_y"] == 100
