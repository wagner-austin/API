"""Tests for viewport exploration target selection."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.movement import (
    walk_or_teleport,
)
from tankpit_bot.sniffer.world_state import (
    reset_world_state,
)
from tests.bot.ai._movement_fixtures import _NOW_MS
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestPickupSurfaceRouting:
    """Pickups route on ground only (user ferry contract 2026-07-19).

    One server-routed command never chains surfaces: boarding a ferry
    is its own click, disembarking auto-stops on the first land tile,
    and a cross-surface click draws "You can't go there!". Regression
    guards for run 2026-07-19 18:20:33, where the riding rule ("water
    is passable while on a ferry") leaked into a pickup dispatch and
    the server refused it after the disembark stop.
    """

    def setup_method(self) -> None:
        """Reset world-state test globals before each case."""
        reset_world_state()

    def _channel_terrain(self, columns: tuple[int, ...]) -> InMemoryTerrainMap:
        """Build terrain with full-height water columns sealing a channel.

        Args:
            columns: X coordinates to fill with water across the whole
                viewport (92..107), so no land detour exists.

        Returns:
            Terrain map with the sealed channel.
        """
        data: dict[tuple[int, int], str] = {}
        for x in columns:
            for y in range(92, 108):
                data[(x, y)] = "W"
        return InMemoryTerrainMap(data)

    def test_pickup_while_riding_disembarks_first(self) -> None:
        """Riding + inland container across water plans the disembark move.

        The ferry-aware view says the container is reachable by
        sailing, but a pickup is one click and the server will not
        pilot the ferry for it. With the container beyond adjacency
        service from the water (a shore-adjacent one dispatches
        directly), the planner must issue the piloted move instead --
        bounded at the first land tile -- and let the next tick
        dispatch the pickup from solid ground.
        """
        from tankpit_bot.bot.ai.ferry import compose_decision_terrain
        from tankpit_bot.state.types import make_terrain_tile
        from tankpit_bot.types.constants import TERRAIN_FERRY

        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        static = self._channel_terrain((100, 101, 102, 103))
        world["terrain"]["100,100"] = make_terrain_tile(
            x=100,
            y=100,
            terrain_type=TERRAIN_FERRY,
        )
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = compose_decision_terrain(world, static, _NOW_MS)
        if terrain is None:
            raise AssertionError("composed terrain unexpectedly None")
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 105, 100, pickup_kind="equipment")

        if result is None:
            raise AssertionError("expected a disembark move, got None")
        assert result["cmd_type"] == "move"
        assert result["target_x"] == 104
        assert result["target_y"] == 100

    def test_pickup_of_water_container_while_riding_dispatches(self) -> None:
        """Riding + container floating on water dispatches the pickup.

        The route stays on the water surface, so it is one command --
        "cant you just pick it up essentially like we were on land?"
        (user 2026-07-20). Regression guard for run 2026-07-20 00:57,
        where the ground-only gate refused water containers outright
        and the disembark branch sailed the bot in circles instead.
        """
        from tankpit_bot.bot.ai.ferry import compose_decision_terrain
        from tankpit_bot.state.types import make_terrain_tile
        from tankpit_bot.types.constants import TERRAIN_FERRY

        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        static = self._channel_terrain((100, 101, 102, 103))
        world["terrain"]["100,100"] = make_terrain_tile(
            x=100,
            y=100,
            terrain_type=TERRAIN_FERRY,
        )
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = compose_decision_terrain(world, static, _NOW_MS)
        if terrain is None:
            raise AssertionError("composed terrain unexpectedly None")
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 103, 100, pickup_kind="equipment")

        if result is None:
            raise AssertionError("expected a pickup dispatch, got None")
        assert result["cmd_type"] == "pickup_equipment"
        assert result["target_x"] == 103
        assert result["target_y"] == 100

    def test_pickup_on_own_water_tile_while_riding_dispatches(self) -> None:
        """Riding directly over the container's tile dispatches the pickup.

        The exact run 2026-07-20 00:57 loop: the bot sat ON the water
        container at (226,196) for 78 ticks re-issuing a refused move
        to its own tile because a water tile is never ground-reachable.
        A zero-length route on the current surface is trivially valid.
        """
        from tankpit_bot.bot.ai.ferry import compose_decision_terrain
        from tankpit_bot.state.types import make_terrain_tile
        from tankpit_bot.types.constants import TERRAIN_FERRY

        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        static = self._channel_terrain((100, 101, 102, 103))
        world["terrain"]["100,100"] = make_terrain_tile(
            x=100,
            y=100,
            terrain_type=TERRAIN_FERRY,
        )
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = compose_decision_terrain(world, static, _NOW_MS)
        if terrain is None:
            raise AssertionError("composed terrain unexpectedly None")
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 100, 100, pickup_kind="equipment")

        if result is None:
            raise AssertionError("expected a pickup dispatch, got None")
        assert result["cmd_type"] == "pickup_equipment"
        assert result["target_x"] == 100
        assert result["target_y"] == 100

    def test_pickup_across_water_on_land_is_skipped(self) -> None:
        """On land, a container across a sealed channel is not dispatchable.

        Even with a ferry bridging the channel, the server does not
        auto-route onto it for a click past the water -- the pickup is
        skipped and the cascade relocates instead.
        """
        from tankpit_bot.bot.ai.ferry import compose_decision_terrain
        from tankpit_bot.state.types import make_terrain_tile
        from tankpit_bot.types.constants import TERRAIN_FERRY

        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        static = self._channel_terrain((102, 103))
        world["terrain"]["102,100"] = make_terrain_tile(
            x=102,
            y=100,
            terrain_type=TERRAIN_FERRY,
        )
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = compose_decision_terrain(world, static, _NOW_MS)
        if terrain is None:
            raise AssertionError("composed terrain unexpectedly None")
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 104, 100, pickup_kind="equipment")

        assert result is None
