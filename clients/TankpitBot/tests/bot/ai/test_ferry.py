"""Tests for ferry detection and boarding.

``test_ferry.py`` was 602 lines; landing selection is now a sibling.
"""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.ferry import (
    FerryAwareTerrain,
    compose_decision_terrain,
    is_riding_ferry,
)
from tankpit_bot.state.types import (
    VIEWPORT_PRESENCE_TTL_MS,
    make_mine_state,
    make_tank_state,
    make_terrain_tile,
)
from tankpit_bot.types.constants import (
    TERRAIN_GROUND,
)
from tests.bot.ai._ferry_fixtures import (
    _NOW_MS,
    _ferry_tile,
)
from tests.bot.ai._support import make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestFerryAwareTerrain:
    """Tests for the composed terrain view."""

    def test_ferry_tile_overlays_static_water(self) -> None:
        """A live ferry tile renders and passes over static water."""
        base = InMemoryTerrainMap({(100, 100): "W"})
        wire = _ferry_tile(100, 100)
        terrain = FerryAwareTerrain(
            base, wire, riding=False, hostile_mine_keys=frozenset(), occupied_tank_keys=frozenset()
        )

        assert terrain.get_terrain(100, 100) == "~"
        assert terrain.is_passable(100, 100) is True

    def test_water_is_passable_only_while_riding(self) -> None:
        """Open water flips passable exactly with the riding flag."""
        base = InMemoryTerrainMap({(101, 100): "W"})

        riding = FerryAwareTerrain(
            base, {}, riding=True, hostile_mine_keys=frozenset(), occupied_tank_keys=frozenset()
        )
        parked = FerryAwareTerrain(
            base, {}, riding=False, hostile_mine_keys=frozenset(), occupied_tank_keys=frozenset()
        )
        assert riding.is_passable(101, 100) is True
        assert parked.is_passable(101, 100) is False

    def test_ground_always_passable_and_rock_never(self) -> None:
        """Static ground and rock keep their semantics in both modes."""
        base = InMemoryTerrainMap({(102, 100): "#"})

        for riding in (False, True):
            terrain = FerryAwareTerrain(
                base,
                {},
                riding=riding,
                hostile_mine_keys=frozenset(),
                occupied_tank_keys=frozenset(),
            )
            assert terrain.is_passable(100, 100) is True
            assert terrain.is_passable(102, 100) is False

    def test_render_viewport_includes_live_ferry(self) -> None:
        """The rendered grid carries the ferry overlay."""
        base = InMemoryTerrainMap({(100, 100): "W"})
        terrain = FerryAwareTerrain(
            base,
            _ferry_tile(100, 100),
            riding=True,
            hostile_mine_keys=frozenset(),
            occupied_tank_keys=frozenset(),
        )

        grid = terrain.render_viewport(100, 100, width=3, height=3)

        assert grid[1][1] == "~"
        assert grid[0][0] == "."

    def test_block_tiles_compose_by_walkability(self) -> None:
        """Bridges read as walkable ground; land/stacked blocks as rock.

        Wire terrain values 1/2/3 are movable concrete blocks
        ([[movable-blocks]], archive-verified 2026-07-20: 4,352 bridge
        tiles over static water, 2,396 land blocks over static ground,
        250 stacks over water — the wire value alone determines
        walkability). A bridge over static water is passable WITHOUT
        riding; a land or stacked block is impassable even though the
        static map calls its tile ground/water.
        """
        from tankpit_bot.types.constants import (
            TERRAIN_BLOCK_BRIDGE,
            TERRAIN_BLOCK_LAND,
            TERRAIN_BLOCK_STACKED,
        )

        base = InMemoryTerrainMap({(101, 100): "W", (103, 100): "W"})
        wire = {
            "101,100": make_terrain_tile(x=101, y=100, terrain_type=TERRAIN_BLOCK_BRIDGE),
            "102,100": make_terrain_tile(x=102, y=100, terrain_type=TERRAIN_BLOCK_LAND),
            "103,100": make_terrain_tile(x=103, y=100, terrain_type=TERRAIN_BLOCK_STACKED),
        }
        terrain = FerryAwareTerrain(
            base, wire, riding=False, hostile_mine_keys=frozenset(), occupied_tank_keys=frozenset()
        )

        assert terrain.get_terrain(101, 100) == "."  # bridge = ground class
        assert terrain.is_passable(101, 100) is True  # walkable, not riding
        assert terrain.get_terrain(102, 100) == "#"  # land block = rock class
        assert terrain.is_passable(102, 100) is False
        assert terrain.get_terrain(103, 100) == "#"  # stack = rock class
        assert terrain.is_passable(103, 100) is False

    def test_bridge_is_routable_for_ground_surface_pickups(self) -> None:
        """A bridge joins the ground routing surface for pickups.

        Walking onto a bridge is ordinary movement (no queue-consuming
        surface transition — user capture 2026-07-20 collected
        equipment reachable only across a built bridge in one
        dispatch), so the single-surface pickup gate must treat bridge
        tiles as ground.
        """
        from tankpit_bot.bot.ai.ferry import SurfaceRouteTerrain
        from tankpit_bot.types.constants import TERRAIN_BLOCK_BRIDGE

        base = InMemoryTerrainMap({(101, 100): "W"})
        wire = {"101,100": make_terrain_tile(x=101, y=100, terrain_type=TERRAIN_BLOCK_BRIDGE)}
        view = FerryAwareTerrain(
            base, wire, riding=False, hostile_mine_keys=frozenset(), occupied_tank_keys=frozenset()
        )

        assert SurfaceRouteTerrain(view, water=False).is_passable(101, 100) is True

    def test_hostile_mine_tile_is_impassable_on_any_surface(self) -> None:
        """A composed hostile mine blocks ground, ferry, and water tiles.

        Stepping on a hostile mine detonates it (45 fuel), so the tile
        is never walkable -- regardless of what terrain sits under the
        mine or whether the tank is riding. Display is unchanged: the
        mine is a passability fact, not a terrain character.
        """
        base = InMemoryTerrainMap({(101, 100): "W"})
        mines = frozenset({"100,100", "101,100", "102,100"})
        wire = _ferry_tile(102, 100)
        terrain = FerryAwareTerrain(
            base, wire, riding=True, hostile_mine_keys=mines, occupied_tank_keys=frozenset()
        )

        assert terrain.is_passable(100, 100) is False  # mined ground
        assert terrain.is_passable(101, 100) is False  # mined water (even riding)
        assert terrain.is_passable(102, 100) is False  # mined ferry tile
        assert terrain.get_terrain(100, 100) == "."
        assert terrain.get_terrain(102, 100) == "~"


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

        assert compose_decision_terrain(world, None, _NOW_MS) is None

    def test_compose_builds_riding_view(self) -> None:
        """Composition carries the riding flag into water passability."""
        world, _self_state = make_world(self_x=100, self_y=100)
        world["terrain"].update(_ferry_tile(100, 100))
        base = InMemoryTerrainMap({(101, 100): "W"})

        composed = compose_decision_terrain(world, base, _NOW_MS)
        if composed is None:
            pytest.fail("expected composed terrain from ferry + base map")

        assert composed.is_passable(101, 100) is True

    def test_compose_folds_hostile_mines_but_not_friendly(self) -> None:
        """Composition blocks enemy mines and keeps same-team mines passable.

        Regression guard for run 2026-07-20 17:16: the dot-hop selector
        consulted terrain without mines and looped 23 ticks proposing a
        mined fuel dot. Mines now arrive through the ONE composed view
        every selector already uses.
        """
        world, _self_state = make_world(self_x=100, self_y=100)
        world["mines"]["103,100"] = make_mine_state(103, 100, 0, -1, 3, source="radar")
        world["mines"]["104,100"] = make_mine_state(104, 100, 0, -1, 1, source="radar")

        composed = compose_decision_terrain(world, InMemoryTerrainMap(), _NOW_MS)
        if composed is None:
            pytest.fail("expected composed terrain from mines + base map")

        assert composed.is_passable(103, 100) is False  # hostile (team 3 vs self 1)
        assert composed.is_passable(104, 100) is True  # friendly (same team)

    def test_compose_folds_tank_bodies_of_every_team(self) -> None:
        """A tank body blocks its tile whether ally or enemy.

        Regression guard for [[flag-triage-20260729]] F6: the composed
        view answered "open" for tiles holding another tank, the bot
        dispatched walks through them, and the server walked the tank
        to the body, stopped, and returned ``error_code=1`` (ten stops
        in run bot-20260803-180918).
        """
        ally = make_tank_state(
            tank_id=2,
            x=103,
            y=100,
            team=1,
            rank=1,
            damage_state=3,
            name="purple-2",
            is_bot=True,
            is_self=False,
            last_viewport_observation_ms=_NOW_MS,
        )
        enemy = make_tank_state(
            tank_id=3,
            x=104,
            y=100,
            team=3,
            rank=1,
            damage_state=3,
            name="orange-3",
            is_bot=True,
            is_self=False,
            last_viewport_observation_ms=_NOW_MS,
        )
        world, _self_state = make_world(self_x=100, self_y=100, tanks={"2": ally, "3": enemy})

        composed = compose_decision_terrain(world, InMemoryTerrainMap(), _NOW_MS)
        if composed is None:
            pytest.fail("expected composed terrain from tanks + base map")

        assert composed.is_passable(103, 100) is False
        assert composed.is_passable(104, 100) is False
        assert composed.is_passable(105, 100) is True

    def test_landing_legality_ignores_mines_and_bodies(self) -> None:
        """A teleport may be aimed where a walk may not go.

        The server displaces a landing off mines and off occupied
        tiles rather than refusing it, so both blockers belong to the
        walk question alone. Conflating them made every approach
        teleport at an enemy impossible -- an enemy always occupies
        its own tile.
        """
        terrain = FerryAwareTerrain(
            InMemoryTerrainMap(),
            {},
            riding=False,
            hostile_mine_keys=frozenset({"100,100"}),
            occupied_tank_keys=frozenset({"101,100"}),
        )

        assert terrain.is_passable(100, 100) is False
        assert terrain.is_landing_legal(100, 100) is True
        assert terrain.is_passable(101, 100) is False
        assert terrain.is_landing_legal(101, 100) is True

    def test_landing_legality_still_refuses_illegal_terrain(self) -> None:
        """Terrain legality is the one thing a landing does respect."""
        base = InMemoryTerrainMap({(102, 100): "#", (103, 100): "W"})
        terrain = FerryAwareTerrain(
            base,
            {},
            riding=False,
            hostile_mine_keys=frozenset(),
            occupied_tank_keys=frozenset(),
        )

        assert terrain.is_landing_legal(102, 100) is False
        assert terrain.is_landing_legal(103, 100) is False

    def test_compose_ages_stale_tank_bodies_out_of_the_view(self) -> None:
        """A tank last seen past the presence TTL stops blocking.

        Without the age-out, the registry's global roster (refreshed
        for every tank on the map by 0x4C MapData) would wall off tiles
        the bot cannot even see.
        """
        stale = make_tank_state(
            tank_id=4,
            x=103,
            y=100,
            team=3,
            rank=1,
            damage_state=3,
            name="orange-4",
            is_bot=True,
            is_self=False,
            last_viewport_observation_ms=_NOW_MS - VIEWPORT_PRESENCE_TTL_MS - 1,
        )
        world, _self_state = make_world(self_x=100, self_y=100, tanks={"4": stale})

        composed = compose_decision_terrain(world, InMemoryTerrainMap(), _NOW_MS)
        if composed is None:
            pytest.fail("expected composed terrain from tanks + base map")

        assert composed.is_passable(103, 100) is True
