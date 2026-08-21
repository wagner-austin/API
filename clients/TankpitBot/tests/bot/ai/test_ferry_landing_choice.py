"""Tests for ferry landing selection."""

from __future__ import annotations

from tankpit_bot.bot.ai.ferry import (
    FerryAwareTerrain,
    clamp_move_target_at_surface_transition,
)
from tests.bot.ai._ferry_fixtures import _ferry_tile
from tests.bot.ai._support import make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestSurfaceTransitionClamp:
    """Tests for clamp_move_target_at_surface_transition."""

    def test_boarding_clamps_at_ferry_tile(self) -> None:
        """A land walk onto a ferry stops at the ferry tile."""
        world, _self_state = make_world(self_x=100, self_y=100)
        base = InMemoryTerrainMap({(103, 100): "W", (104, 100): "W"})
        wire = _ferry_tile(102, 100)
        terrain = FerryAwareTerrain(
            base,
            wire,
            riding=False,
            hostile_mine_keys=frozenset(),
            occupied_tank_keys=frozenset(),
            refused_landing_keys=frozenset(),
        )

        clamped = clamp_move_target_at_surface_transition(
            world,
            terrain,
            100,
            100,
            102,
            100,
        )

        assert clamped == (102, 100)

    def test_disembark_clamps_at_first_land_tile(self) -> None:
        """A ride across water toward inland stops one step onto land."""
        world, _self_state = make_world(self_x=100, self_y=100)
        base = InMemoryTerrainMap(
            {(100, 100): "W", (101, 100): "W", (102, 100): "W"},
        )
        terrain = FerryAwareTerrain(
            base,
            _ferry_tile(100, 100),
            riding=True,
            hostile_mine_keys=frozenset(),
            occupied_tank_keys=frozenset(),
            refused_landing_keys=frozenset(),
        )

        clamped = clamp_move_target_at_surface_transition(
            world,
            terrain,
            100,
            100,
            105,
            100,
        )

        assert clamped == (103, 100)

    def test_pure_land_path_keeps_target(self) -> None:
        """A path that never changes surface class is not clamped."""
        world, _self_state = make_world(self_x=100, self_y=100)
        terrain = FerryAwareTerrain(
            InMemoryTerrainMap(),
            {},
            riding=False,
            hostile_mine_keys=frozenset(),
            occupied_tank_keys=frozenset(),
            refused_landing_keys=frozenset(),
        )

        clamped = clamp_move_target_at_surface_transition(
            world,
            terrain,
            100,
            100,
            105,
            100,
        )

        assert clamped == (105, 100)

    def test_pure_water_ride_keeps_target(self) -> None:
        """Riding across open water to a water tile is one command."""
        world, _self_state = make_world(self_x=100, self_y=100)
        data = {(x, 100): "W" for x in range(100, 106)}
        terrain = FerryAwareTerrain(
            InMemoryTerrainMap(data),
            _ferry_tile(100, 100),
            riding=True,
            hostile_mine_keys=frozenset(),
            occupied_tank_keys=frozenset(),
            refused_landing_keys=frozenset(),
        )

        clamped = clamp_move_target_at_surface_transition(
            world,
            terrain,
            100,
            100,
            105,
            100,
        )

        assert clamped == (105, 100)


class TestSurfaceRouteTerrain:
    """The single-action pickup-routing surface (user contract 2026-07-19/20)."""

    def test_ground_surface_only_plain_ground_is_traversable(self) -> None:
        """On land: ground passes; water, rock, and ferry tiles all block.

        One server-routed command never chains surfaces: crossing onto
        a ferry is a queue-consuming boarding action, so a pickup route
        may not traverse it even though the riding view calls it
        passable.
        """
        from tankpit_bot.bot.ai.ferry import SurfaceRouteTerrain

        base = InMemoryTerrainMap({(101, 100): "W", (102, 100): "#"})
        ferry_view = FerryAwareTerrain(
            base,
            _ferry_tile(103, 100),
            riding=True,
            hostile_mine_keys=frozenset(),
            occupied_tank_keys=frozenset(),
            refused_landing_keys=frozenset(),
        )
        ground_surface = SurfaceRouteTerrain(ferry_view, water=False)

        assert ground_surface.is_passable(100, 100) is True  # plain ground
        assert ground_surface.is_passable(101, 100) is False  # water (even riding)
        assert ground_surface.is_passable(102, 100) is False  # rock
        assert ground_surface.is_passable(103, 100) is False  # live ferry tile
        # Cell display is unchanged -- only passability differs.
        assert ground_surface.get_terrain(103, 100) == "~"
        assert ground_surface.get_terrain(101, 100) == "W"

    def test_water_surface_water_and_ferry_pass_land_blocks(self) -> None:
        """Riding: water and ferry tiles pass; ground and rock block.

        A container floating on water picks up normally from the ferry
        (user 2026-07-20) -- the route stays on the water surface. Land
        is the OTHER surface: reaching it is a queue-consuming
        disembark, never part of a pickup route.
        """
        from tankpit_bot.bot.ai.ferry import SurfaceRouteTerrain

        base = InMemoryTerrainMap({(101, 100): "W", (102, 100): "#"})
        ferry_view = FerryAwareTerrain(
            base,
            _ferry_tile(103, 100),
            riding=True,
            hostile_mine_keys=frozenset(),
            occupied_tank_keys=frozenset(),
            refused_landing_keys=frozenset(),
        )
        water_surface = SurfaceRouteTerrain(ferry_view, water=True)

        assert water_surface.is_passable(101, 100) is True  # water
        assert water_surface.is_passable(103, 100) is True  # live ferry tile
        assert water_surface.is_passable(100, 100) is False  # plain ground
        assert water_surface.is_passable(102, 100) is False  # rock

    def test_mined_tile_blocks_both_surfaces(self) -> None:
        """A composed hostile mine is unroutable on either surface.

        The surface view intersects with the wrapped view's
        passability, so mine-blocking composed upstream propagates.
        """
        from tankpit_bot.bot.ai.ferry import SurfaceRouteTerrain

        base = InMemoryTerrainMap({(101, 100): "W"})
        view = FerryAwareTerrain(
            base,
            {},
            riding=True,
            hostile_mine_keys=frozenset({"100,100", "101,100"}),
            occupied_tank_keys=frozenset(),
            refused_landing_keys=frozenset(),
        )

        assert SurfaceRouteTerrain(view, water=False).is_passable(100, 100) is False
        assert SurfaceRouteTerrain(view, water=True).is_passable(101, 100) is False

    def test_landing_legality_intersects_surface_with_wrapped_view(self) -> None:
        """The surface view answers landing without walk-only blockers.

        Three outcomes, one per branch: terrain the wrapped view calls
        illegal is refused; a legal tile off the routing surface is
        refused; a legal tile on the surface is accepted even when a
        mine or body makes it unwalkable.
        """
        from tankpit_bot.bot.ai.ferry import SurfaceRouteTerrain

        base = InMemoryTerrainMap({(102, 100): "#", (101, 100): "W"})
        view = FerryAwareTerrain(
            base,
            {},
            riding=False,
            hostile_mine_keys=frozenset({"100,100"}),
            occupied_tank_keys=frozenset(),
            refused_landing_keys=frozenset(),
        )
        ground_surface = SurfaceRouteTerrain(view, water=False)

        assert ground_surface.is_landing_legal(102, 100) is False  # rock
        assert ground_surface.is_landing_legal(101, 100) is False  # off-surface water
        assert ground_surface.is_landing_legal(100, 100) is True  # mined but landable
        assert ground_surface.is_passable(100, 100) is False  # and still unwalkable

    def test_render_viewport_delegates_to_wrapped_view(self) -> None:
        """The rendering is the wrapped view's rendering, verbatim."""
        from tankpit_bot.bot.ai.ferry import SurfaceRouteTerrain

        base = InMemoryTerrainMap({(100, 100): "W"})
        view = FerryAwareTerrain(
            base,
            _ferry_tile(100, 100),
            riding=False,
            hostile_mine_keys=frozenset(),
            occupied_tank_keys=frozenset(),
            refused_landing_keys=frozenset(),
        )
        assert SurfaceRouteTerrain(view, water=False).render_viewport(
            100, 100, 1, 1
        ) == view.render_viewport(100, 100, 1, 1)


class TestSurfaceAttainability:
    """Surface-view landing attainability delegates through the wrap."""

    def test_hostile_mine_blocks_attainability_on_the_surface(self) -> None:
        """The wrapped view's team-scoped mine knowledge survives the wrap."""
        from tankpit_bot.bot.ai.ferry import SurfaceRouteTerrain

        base = InMemoryTerrainMap()
        ferry_view = FerryAwareTerrain(
            base,
            {},
            riding=False,
            hostile_mine_keys=frozenset({"101,100"}),
            occupied_tank_keys=frozenset(),
            refused_landing_keys=frozenset(),
        )
        surface = SurfaceRouteTerrain(ferry_view, water=False)

        assert surface.is_landing_attainable(101, 100) is False
        assert surface.is_landing_attainable(100, 100) is True

    def test_off_surface_tiles_are_unattainable(self) -> None:
        """A clean tile off the routing surface still cannot be landed on."""
        from tankpit_bot.bot.ai.ferry import SurfaceRouteTerrain

        base = InMemoryTerrainMap({(101, 100): "W"})
        ferry_view = FerryAwareTerrain(
            base,
            {},
            riding=True,
            hostile_mine_keys=frozenset(),
            occupied_tank_keys=frozenset(),
            refused_landing_keys=frozenset(),
        )
        ground_surface = SurfaceRouteTerrain(ferry_view, water=False)

        assert ground_surface.is_landing_attainable(101, 100) is False
