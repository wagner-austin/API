"""Tests for direct moves, waypoints, and rejection guards.

``test_movement.py`` was 761 lines; the approach-tile suite and the
exploration suite are now siblings.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.movement import (
    _direct_move_command,
    walk_or_teleport,
)
from tankpit_bot.bot.ai.movement_exploration import (
    select_exploration_command,
    viewport_exploration_candidates,
)
from tankpit_bot.sniffer.world_state import (
    mark_move_target_failed,
)
from tankpit_bot.state.types import (
    TankStateDict,
    make_mine_state,
    make_tank_state,
    make_viewport_state,
)
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestWalkOrTeleport:
    """Tests for public movement planning behavior."""

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
        )

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
