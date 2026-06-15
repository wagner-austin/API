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
    ViewportStateDict,
    make_mine_state,
    make_tank_state,
    viewport_scan_key,
)
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


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

        result = walk_or_teleport(ctx, target_x, target_y, pickup_kind="equipment")

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

        result = walk_or_teleport(ctx, target_x, target_y, pickup_kind="equipment")

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

        result = walk_or_teleport(ctx, target_x, target_y, pickup_kind="equipment")

        if result is None:
            raise AssertionError("expected edge-sidestep approach move on the bottom row")
        assert result["cmd_type"] == "move"
        assert result["target_y"] == bottom
        assert result["target_x"] in (99, 101)

    def test_approach_skips_own_tile_when_already_on_the_edge(self) -> None:
        """Standing on the clamp tile selects the next nearest edge tile."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        # Re-frame the viewport so self sits exactly on its right edge.
        world["viewport"] = ViewportStateDict(left=85, top=92, width=16, height=16)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 110, 100, pickup_kind="equipment")

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
            ),
        }
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        result = walk_or_teleport(ctx, right + 10, 100, pickup_kind="equipment")

        assert result is None

    def test_uses_final_move_target_when_viewport_path_exists(self) -> None:
        """In-viewport detours still issue the final move target."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
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
        world, self_state = make_world(self_x=100, self_y=100, fuel=300, tanks=tanks)
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
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        world["mines"] = {"103,99": make_mine_state(x=103, y=99, mine_type=0, tank_id=-1, team=1)}
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
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
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
            ),
        }
        world, self_state = make_world(self_x=100, self_y=100, fuel=300, tanks=tanks)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = walk_or_teleport(ctx, 107, 100, pickup_kind=None)

        assert result is None

    def test_direct_move_command_rejects_off_viewport_target(self) -> None:
        """Direct move helper refuses targets outside the visible viewport."""
        world, self_state = make_world(self_x=64, self_y=64, fuel=300)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = _direct_move_command(ctx, 72, 63, pickup_kind=None)

        assert result is None

    def test_direct_move_command_rejects_mined_target(self) -> None:
        """Direct move helper rejects final targets occupied by known mines."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        world["mines"] = {"107,100": make_mine_state(x=107, y=100, mine_type=0, tank_id=-1, team=1)}
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = _direct_move_command(ctx, 107, 100, pickup_kind=None)

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
            ),
        }
        world, self_state = make_world(self_x=100, self_y=100, fuel=300, tanks=tanks)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        result = walk_or_teleport(ctx, 107, 100, pickup_kind=None)

        assert result is None

    def test_rejects_mined_move_without_terrain(self) -> None:
        """Without terrain, mine occupancy on the target blocks the move."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        world["mines"] = {"107,100": make_mine_state(x=107, y=100, mine_type=0, tank_id=-1, team=1)}
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        result = walk_or_teleport(ctx, 107, 100, pickup_kind=None)

        assert result is None

    def test_moves_to_final_target_when_mine_blocks_straight_line(self) -> None:
        """Known mines on the straight line still allow an in-viewport detour."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        world["mines"] = {"103,100": make_mine_state(x=103, y=100, mine_type=0, tank_id=-1, team=1)}
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
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        base_candidates = viewport_exploration_candidates(ctx)
        rotated_candidates = viewport_exploration_candidates(ctx, candidate_offset=1)

        assert rotated_candidates[0] == base_candidates[1]
        assert rotated_candidates[-1] == base_candidates[0]

    def test_viewport_exploration_candidates_return_empty_for_single_tile_viewport(self) -> None:
        """Degenerate one-tile viewports yield no exploration candidates."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        world["viewport"] = ViewportStateDict(left=100, top=100, width=1, height=1)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        candidates = viewport_exploration_candidates(ctx)

        assert candidates == []

    def test_select_exploration_prefers_unscanned_neighboring_viewport(self) -> None:
        """Exploration prefers edges that expose fresh unscanned space."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        current = world["viewport"]
        world["scanned_viewports"] = {
            viewport_scan_key(current["left"], current["top"]): 100000,
            viewport_scan_key(92, 99): 100000,
            viewport_scan_key(99, 92): 100000,
            viewport_scan_key(99, 99): 100000,
        }
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
        assert viewport_scan_key(next_left, next_top) not in world["scanned_viewports"]

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
