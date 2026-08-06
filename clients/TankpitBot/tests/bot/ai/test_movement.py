"""Focused tests for viewport-aware movement behavior."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.movement import (
    _direct_move_command,
    select_exploration_command,
    viewport_exploration_candidates,
    walk_or_teleport,
)
from tankpit_bot.sniffer.world_state import mark_move_target_failed, reset_world_state
from tankpit_bot.state.types import (
    TankStateDict,
    make_mine_state,
    make_tank_state,
    make_viewport_state,
)
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap

_NOW_MS = 1_000_000


class TestWalkOrTeleport:
    """Tests for public movement planning behavior."""

    def setup_method(self) -> None:
        """Reset world-state test globals before each case."""
        reset_world_state()

    def test_approach_sidesteps_blocked_clamp_tile(self) -> None:
        """A rock on the projected edge tile no longer kills the approach.

        Regression guard for live run 20260610-000x: known off-viewport
        containers were rejected for whole sessions because the single
        geometric clamp tile was rock/water. The approach must slide
        along the facing edge to the nearest walkable tile instead.
        """
        from tankpit_bot.bot.ai.context import local_actionable_bounds

        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        probe_ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
        _left, _top, right, _bottom = local_actionable_bounds(probe_ctx)
        # Target due east, beyond the right edge; its clamp tile is rock.
        target_x, target_y = right + 10, 100
        terrain = InMemoryTerrainMap({(right, 100): "#"})
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, target_x, target_y, pickup_kind=None)

        if result is None:
            raise AssertionError("expected edge-sidestep approach move")
        assert result["cmd_type"] == "move"
        # Nearest walkable edge tile to the rock clamp is one row off it.
        assert result["target_x"] == right
        assert result["target_y"] in (99, 101)

    def test_approach_teleports_to_target_when_facing_edge_is_blocked(self) -> None:
        """A fully blocked facing edge falls back to teleporting at the target.

        The bot knows the target's exact coordinates; when no facing-edge
        tile is walkable it must go to the target directly rather than
        abandoning it for blind search hops.
        """
        from tankpit_bot.bot.ai.context import local_actionable_bounds

        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        probe_ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
        _left, top, right, bottom = local_actionable_bounds(probe_ctx)
        target_x, target_y = right + 10, 100
        terrain_data: dict[tuple[int, int], str] = {(right, y): "#" for y in range(top, bottom + 1)}
        terrain = InMemoryTerrainMap(terrain_data)
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, target_x, target_y, pickup_kind=None)

        if result is None:
            raise AssertionError("expected direct teleport to the known target")
        assert result["cmd_type"] == "teleport"
        assert result["target_x"] == target_x
        assert result["target_y"] == target_y

    def test_approach_scans_horizontal_edge_for_southern_target(self) -> None:
        """A target beyond the bottom edge scans the facing row for ground."""
        from tankpit_bot.bot.ai.context import local_actionable_bounds

        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        probe_ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
        _left, _top, _right, bottom = local_actionable_bounds(probe_ctx)
        target_x, target_y = 100, bottom + 10
        terrain = InMemoryTerrainMap({(100, bottom): "#"})
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, target_x, target_y, pickup_kind=None)

        if result is None:
            raise AssertionError("expected edge-sidestep approach move on the bottom row")
        assert result["cmd_type"] == "move"
        assert result["target_y"] == bottom
        assert result["target_x"] in (99, 101)

    def test_approach_skips_own_tile_when_already_on_the_edge(self) -> None:
        """Standing on the clamp tile selects the next nearest edge tile."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        # Re-frame the viewport so self sits exactly on its right edge.
        world["viewport"] = make_viewport_state(left=85, top=92, width=16, height=16)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 110, 100, pickup_kind=None)

        if result is None:
            raise AssertionError("expected sidestep move off the own-tile clamp")
        assert result["cmd_type"] == "move"
        assert result["target_x"] == 100
        assert result["target_y"] in (99, 101)

    def test_approach_without_terrain_returns_none_when_clamp_is_occupied(self) -> None:
        """Without terrain, an enemy on the clamp tile leaves no approach."""
        from tankpit_bot.bot.ai.context import local_actionable_bounds

        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        probe_ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
        _left, _top, right, _bottom = local_actionable_bounds(probe_ctx)
        world["tanks"] = {
            "50": make_tank_state(
                tank_id=50,
                x=right,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                # A blocking body must be viewport-fresh under the
                # occupancy law -- a stale entry no longer vetoes.
                last_viewport_observation_ms=100000,
            ),
        }
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        result = walk_or_teleport(ctx, right + 10, 100, pickup_kind="equipment")

        assert result is None

    def test_approach_teleports_when_facing_edge_is_passable_but_unreachable(self) -> None:
        """Passable edge tiles with no walk path still fall back to teleport.

        A water wall one column inside the facing edge leaves every
        edge candidate passable but walk-unreachable; the approach must
        skip them all and teleport to the real target.
        """
        from tankpit_bot.bot.ai.context import local_actionable_bounds

        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        probe_ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
        _left, top, right, bottom = local_actionable_bounds(probe_ctx)
        target_x, target_y = right + 10, 100
        wall = {(right - 1, y): "W" for y in range(top, bottom + 1)}
        terrain = InMemoryTerrainMap(wall)
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, target_x, target_y, pickup_kind=None)

        if result is None:
            raise AssertionError("expected direct teleport past the unreachable edge")
        assert result["cmd_type"] == "teleport"
        assert result["target_x"] == target_x
        assert result["target_y"] == target_y

    def test_approach_teleports_when_selected_edge_tile_is_enemy_occupied(self) -> None:
        """An enemy parked on the chosen approach tile forces the teleport fallback.

        The edge selector checks terrain and walk paths but not tank
        occupancy; the recursive walk then refuses the enemy-occupied
        tile, and the approach must fall through to teleporting at the
        real target instead of returning nothing.
        """
        from tankpit_bot.bot.ai.context import local_actionable_bounds

        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        probe_ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
        _left, _top, right, _bottom = local_actionable_bounds(probe_ctx)
        target_x, target_y = right + 10, 100
        world["tanks"] = {
            "50": make_tank_state(
                tank_id=50,
                x=right,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                # A blocking body must be viewport-fresh under the
                # occupancy law -- a stale entry no longer vetoes.
                last_viewport_observation_ms=100000,
            ),
        }
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, target_x, target_y, pickup_kind=None)

        if result is None:
            raise AssertionError("expected teleport fallback past the occupied edge tile")
        assert result["cmd_type"] == "teleport"
        assert result["target_x"] == target_x
        assert result["target_y"] == target_y

    def test_approach_move_without_terrain_returns_none_when_clamp_is_occupied(self) -> None:
        """Without terrain, a plain move's occupied clamp tile yields no command.

        The pickup variant is covered above; the plain-move variant must
        also return ``None`` because without a terrain map there is no
        teleport-fallback landing computation.
        """
        from tankpit_bot.bot.ai.context import local_actionable_bounds

        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        probe_ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
        _left, _top, right, _bottom = local_actionable_bounds(probe_ctx)
        world["tanks"] = {
            "50": make_tank_state(
                tank_id=50,
                x=right,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                # A blocking body must be viewport-fresh under the
                # occupancy law -- a stale entry no longer vetoes.
                last_viewport_observation_ms=100000,
            ),
        }
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        result = walk_or_teleport(ctx, right + 10, 100, pickup_kind=None)

        assert result is None

    def test_uses_final_move_target_when_viewport_path_exists(self) -> None:
        """In-viewport detours still issue the final move target."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap({(102, 100): "#"})
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 104, 100, pickup_kind=None)

        if result is None:
            raise AssertionError("expected direct move to final viewport target")
        assert result["cmd_type"] == "move"
        assert result["target_x"] == 104
        assert result["target_y"] == 100

    def test_uses_final_pickup_target_when_viewport_collection_path_exists(self) -> None:
        """In-viewport detours still issue the final pickup target."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 103, 100, pickup_kind="equipment")

        if result is None:
            raise AssertionError("expected direct pickup for final viewport target")
        assert result["cmd_type"] == "pickup_equipment"
        assert result["target_x"] == 103
        assert result["target_y"] == 100

    def test_ignores_enemy_on_old_waypoint_tile(self) -> None:
        """Enemy occupancy off the final target does not block the move."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=103,
                y=99,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(self_x=100, self_y=100, fuel=150, tanks=tanks)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap({(102, 100): "#"})
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 104, 100, pickup_kind=None)

        if result is None:
            raise AssertionError("expected move despite non-target enemy occupancy")
        assert result["cmd_type"] == "move"
        assert result["target_x"] == 104
        assert result["target_y"] == 100

    def test_ignores_mine_on_old_waypoint_tile(self) -> None:
        """Mine occupancy off the final target does not block the move."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        world["mines"] = {"103,99": make_mine_state(x=103, y=99, mine_type=0, tank_id=-1, team=0)}
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap({(102, 100): "#"})
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 104, 100, pickup_kind=None)

        if result is None:
            raise AssertionError("expected move despite non-target mine occupancy")
        assert result["cmd_type"] == "move"
        assert result["target_x"] == 104
        assert result["target_y"] == 100

    def test_rejects_recently_failed_move_target(self) -> None:
        """Recently failed move targets are skipped before planning."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        mark_move_target_failed(107, 100, 90000)
        result = walk_or_teleport(ctx, 107, 100, pickup_kind=None)

        assert result is None

    def test_rejects_enemy_occupied_direct_move(self) -> None:
        """Enemy occupancy on the final move target blocks the move."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=107,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                # A blocking body must be viewport-fresh under the
                # occupancy law -- a stale entry no longer vetoes.
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(self_x=100, self_y=100, fuel=150, tanks=tanks)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 107, 100, pickup_kind=None)

        assert result is None

    def test_direct_move_command_rejects_off_viewport_target(self) -> None:
        """Direct move helper refuses targets outside the visible viewport."""
        world, self_state = make_world(self_x=64, self_y=64, fuel=150)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = _direct_move_command(ctx, 72, 63)

        assert result is None

    def test_direct_move_command_rejects_mined_target(self) -> None:
        """Direct move helper rejects final targets occupied by known mines."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        world["mines"] = {"107,100": make_mine_state(x=107, y=100, mine_type=0, tank_id=-1, team=0)}
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = _direct_move_command(ctx, 107, 100)

        assert result is None

    def test_rejects_enemy_occupied_move_without_terrain(self) -> None:
        """Without terrain, enemy occupancy on the target still blocks the move."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=107,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                # A blocking body must be viewport-fresh under the
                # occupancy law -- a stale entry no longer vetoes.
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(self_x=100, self_y=100, fuel=150, tanks=tanks)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        result = walk_or_teleport(ctx, 107, 100, pickup_kind=None)

        assert result is None

    def test_rejects_mined_move_without_terrain(self) -> None:
        """Without terrain, mine occupancy on the target blocks the move."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        world["mines"] = {"107,100": make_mine_state(x=107, y=100, mine_type=0, tank_id=-1, team=0)}
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        result = walk_or_teleport(ctx, 107, 100, pickup_kind=None)

        assert result is None

    def test_moves_to_final_target_when_mine_blocks_straight_line(self) -> None:
        """Known mines on the straight line still allow an in-viewport detour."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        world["mines"] = {"103,100": make_mine_state(x=103, y=100, mine_type=0, tank_id=-1, team=0)}
        ai_state = make_scanned_ai_state()
        inventory = make_inventory(default_count=5)
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 107, 100, pickup_kind=None)

        if result is None:
            raise AssertionError("expected safe move around mined straight line")
        assert result["cmd_type"] == "move"
        assert result["target_x"] == 107
        assert result["target_y"] == 100

    def test_rejects_terrain_blocked_teleport_when_exact_fuel_is_too_low(self) -> None:
        """Terrain fallback rejects teleports whose exact jump cost exceeds fuel."""
        terrain_data = {(103, y): "#" for y in range(92, 108)}
        world, self_state = make_world(self_x=100, self_y=100, fuel=30)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap(terrain_data)
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 107, 100, pickup_kind=None)

        assert result is None

    def test_viewport_exploration_candidates_rotate_with_offset(self) -> None:
        """Exploration candidate order can rotate to avoid repeating one edge."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        base_candidates = viewport_exploration_candidates(ctx)
        rotated_candidates = viewport_exploration_candidates(ctx, candidate_offset=1)

        assert rotated_candidates[0] == base_candidates[1]
        assert rotated_candidates[-1] == base_candidates[0]

    def test_viewport_exploration_candidates_return_empty_for_single_tile_viewport(self) -> None:
        """Degenerate one-tile viewports yield no exploration candidates."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        world["viewport"] = make_viewport_state(left=100, top=100, width=1, height=1)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        candidates = viewport_exploration_candidates(ctx)

        assert candidates == []

    def test_select_exploration_prefers_unscanned_neighboring_viewport(self) -> None:
        """Exploration prefers edges that expose fresh unscanned space."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        current = world["viewport"]
        scanned_viewport_origins = [
            (current["left"], current["top"]),
            (92, 99),
            (99, 92),
            (99, 99),
        ]
        covered: dict[str, int] = {}
        for left, top in scanned_viewport_origins:
            for y in range(top, top + current["height"]):
                for x in range(left, left + current["width"]):
                    covered[f"{x},{y}"] = 100000
        world["scanned_tiles"] = covered
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        result = select_exploration_command(ctx)

        if result is None:
            raise AssertionError("expected exploration command")
        candidate_x, candidate_y, command = result
        assert command["cmd_type"] == "move"
        next_left = max(0, min(240, candidate_x - 8))
        next_top = max(0, min(240, candidate_y - 8))
        assert (next_left, next_top) not in scanned_viewport_origins

    def test_surface_transition_clamps_move_target(self) -> None:
        """A ground-to-ferry transition clamps the move at the ferry tile."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap({(103, 100): "~"})
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 105, 100, pickup_kind=None)

        if result is None:
            raise AssertionError("expected clamped move at surface transition")
        assert result["cmd_type"] == "move"
        assert result["target_x"] == 103
        assert result["target_y"] == 100

    def test_select_exploration_skips_recently_failed_candidate(self) -> None:
        """Exploration steps past a candidate that walk_or_teleport rejects.

        ``walk_or_teleport`` short-circuits to ``None`` for tiles on the
        recent-failure list. The exploration loop must skip such
        candidates and try the next one rather than giving up after the
        first None.
        """
        from tankpit_bot.sniffer.world_state import (
            mark_move_target_failed,
            reset_world_state,
        )

        reset_world_state()
        try:
            world, self_state = make_world(self_x=100, self_y=100, fuel=800)
            ai_state = make_scanned_ai_state()
            inventory = make_inventory()
            ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
            candidates = viewport_exploration_candidates(ctx)
            if not candidates:
                raise AssertionError("expected at least one exploration candidate")
            first_x, first_y = candidates[0]
            mark_move_target_failed(first_x, first_y, 99000)

            result = select_exploration_command(ctx)

            if result is None:
                raise AssertionError("expected fallback to a non-failed candidate")
            picked_x, picked_y, _command = result
            assert (picked_x, picked_y) != (first_x, first_y)
        finally:
            reset_world_state()


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
        from tankpit_bot.state.types import TERRAIN_FERRY, make_terrain_tile

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
        from tankpit_bot.state.types import TERRAIN_FERRY, make_terrain_tile

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
        from tankpit_bot.state.types import TERRAIN_FERRY, make_terrain_tile

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
        from tankpit_bot.state.types import TERRAIN_FERRY, make_terrain_tile

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
