"""Tests for approach-tile selection when the target is off-viewport."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.movement import (
    walk_or_teleport,
)
from tankpit_bot.state.types import (
    make_tank_state,
    make_viewport_state,
)
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestApproachTarget:
    """Tests for approach-tile selection when the target is off-viewport."""

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
