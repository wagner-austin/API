"""Tests for COLLECT-mode recovery helpers.

Equipment computation, kill expiry, and command requirements.
``test_collect_mode_helpers.py`` was 629 lines; the move-selection
tests are now a sibling.
"""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.collect_pickups import (
    select_equipment_target as _select_equipment_target_command,
)
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    compute_equipment,
    expire_kills,
    filter_killed_tanks,
    require_command,
)
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.types import make_move_command
from tankpit_bot.state.types import ContainerStateDict, TankStateDict, make_mine_state
from tests.bot.ai._collect_helper_fixtures import _enemy
from tests.bot.ai._support import (
    make_container,
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestRecoveryHelpers:
    """Tests for helper-level recovery planning behavior."""

    def test_compute_equipment_collect_fuel(self) -> None:
        """Stocked combat equipment enables the standard equipment list."""
        assert compute_equipment(800, make_inventory()) == [2, 4, 5]

    def test_compute_equipment_hunt(self) -> None:
        """Combat mode uses the same stocked equipment selection."""
        assert compute_equipment(800, make_inventory()) == [2, 4, 5]

    def test_compute_equipment_no_shields(self) -> None:
        """Shields are never part of the active combat equipment selection."""
        assert 1 not in compute_equipment(800, make_inventory())

    def test_compute_equipment_dual_depleted(self) -> None:
        """Dual depletion removes dual from the active equipment list."""
        assert compute_equipment(800, make_inventory(dual_count=0)) == [4, 5]

    def test_compute_equipment_homing_depleted(self) -> None:
        """Homing depletion removes homing from the active equipment list."""
        inventory = make_inventory()
        inventory["homing_shots"]["count"] = 0
        assert compute_equipment(800, inventory) == [2, 5]

    def test_expire_kills_removes_expired(self) -> None:
        """Expired kill cooldown entries are removed."""
        assert expire_kills({"50": 1000, "60": 5000}, 22000, 20000) == {"60": 5000}

    def test_expire_kills_keeps_recent(self) -> None:
        """Recent kill cooldown entries are preserved."""
        assert expire_kills({"50": 10000, "60": 15000}, 20000, 20000) == {
            "50": 10000,
            "60": 15000,
        }

    def test_expire_kills_empty_input(self) -> None:
        """Empty kill maps remain empty."""
        assert expire_kills({}, 20000, 20000) == {}

    def test_filter_killed_tanks_removes_killed(self) -> None:
        """Killed tanks are removed from the filtered world snapshot."""
        tanks: dict[str, TankStateDict] = {
            "50": _enemy(x=105, y=105),
            "60": _enemy(x=110, y=110),
        }
        world, _ = make_world(tanks=tanks)

        filtered = filter_killed_tanks(world, {"50": 100000})

        assert "50" not in filtered["tanks"]
        assert "60" in filtered["tanks"]

    def test_filter_killed_tanks_empty_killed(self) -> None:
        """No killed tanks returns the original world object."""
        world, _ = make_world(tanks={"50": _enemy(x=105, y=105)})

        filtered = filter_killed_tanks(world, {})

        assert filtered is world

    def test_require_command_returns_command(self) -> None:
        """Concrete commands are returned unchanged."""
        command = make_move_command(10, 20)

        assert require_command(command, 10, 20, "fuel") == command

    def test_require_command_raises_on_missing_command(self) -> None:
        """Missing commands raise instead of soft-failing."""
        with pytest.raises(ValueError, match="No executable command for fuel target"):
            require_command(None, 10, 20, "fuel")

    def test_select_equipment_target_command_picks_up_mined_tile(self) -> None:
        """Equipment target selection allows pickup on mined tiles."""
        containers: dict[str, ContainerStateDict] = {
            "103,100": make_container(103, 100, 0, False),
        }
        world, self_state = make_world(self_x=100, self_y=100, fuel=800, containers=containers)
        world["mines"] = {
            "103,100": make_mine_state(x=103, y=100, mine_type=0, tank_id=-1, team=0),
        }
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(default_count=30),
            100000,
            InMemoryTerrainMap(),
            "",
        )

        result = _select_equipment_target_command(ctx)

        if result is None:
            raise AssertionError("expected equipment pickup command")
        container, command = result
        assert container["x"] == 103
        assert command["cmd_type"] == "pickup_equipment"
        assert command["target_x"] == 103
        assert command["target_y"] == 100

    def test_walk_or_teleport_skips_water_locked_target(self) -> None:
        """Water-locked targets return None (unreachable by teleport)."""
        containers: dict[str, ContainerStateDict] = {
            "107,107": make_container(107, 107, 0, False),
        }
        world, self_state = make_world(self_x=100, self_y=100, fuel=150, containers=containers)
        terrain = InMemoryTerrainMap(
            terrain_data={
                (107, 107): "W",
                (108, 107): "W",
                (106, 107): "W",
                (107, 108): "#",
                (107, 106): "#",
            }
        )
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(default_count=5),
            100000,
            terrain,
            "",
        )

        result = walk_or_teleport(ctx, 107, 107, pickup_kind=None)

        assert result is None

    def test_walk_or_teleport_direct_move_when_pickup_disabled(self) -> None:
        """Open-ground scouting uses a direct move when pickup mode is disabled."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(default_count=5),
            100000,
            InMemoryTerrainMap(),
            "",
        )

        result = walk_or_teleport(ctx, 107, 100, pickup_kind=None)

        if result is None:
            raise AssertionError("expected direct move command")
        assert result["cmd_type"] == "move"
        assert result["target_x"] == 107
        assert result["target_y"] == 100

    def test_walk_or_teleport_direct_pickup_on_open_ground(self) -> None:
        """Open-ground collection keeps pickup mode when the route is clear."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(default_count=5),
            100000,
            InMemoryTerrainMap(),
            "",
        )

        result = walk_or_teleport(ctx, 103, 100, pickup_kind="equipment")

        if result is None:
            raise AssertionError("expected direct pickup command")
        assert result["cmd_type"] == "pickup_equipment"
        assert result["target_x"] == 103
        assert result["target_y"] == 100

    def test_walk_or_teleport_picks_up_mined_tile_with_terrain(self) -> None:
        """Terrain routing still produces a legal command for mined pickup tiles."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        world["mines"] = {"107,100": make_mine_state(x=107, y=100, mine_type=0, tank_id=-1, team=0)}
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(default_count=5),
            100000,
            InMemoryTerrainMap(),
            "",
        )

        result = walk_or_teleport(ctx, 107, 100, pickup_kind="equipment")

        if result is None:
            raise AssertionError("expected command for mined pickup tile")
        assert result["cmd_type"] in ("pickup_equipment", "teleport")

    def test_walk_or_teleport_picks_up_mined_tile_without_terrain(self) -> None:
        """Occupancy-only routing still allows pickup on mined tiles."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        world["mines"] = {"107,100": make_mine_state(x=107, y=100, mine_type=0, tank_id=-1, team=0)}
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(default_count=5),
            100000,
            None,
            "",
        )

        result = walk_or_teleport(ctx, 107, 100, pickup_kind="equipment")

        if result is None:
            raise AssertionError("expected pickup command for mined tile")
        assert result["cmd_type"] == "pickup_equipment"

    def test_walk_or_teleport_picks_up_visible_edge_target(self) -> None:
        """Visible edge pickup targets are actionable without an approach step."""
        containers: dict[str, ContainerStateDict] = {
            "66,63": make_container(66, 63, 0, False),
        }
        world, self_state = make_world(self_x=64, self_y=64, fuel=800, containers=containers)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(default_count=5),
            100000,
            InMemoryTerrainMap(),
            "",
        )

        result = walk_or_teleport(ctx, 66, 63, pickup_kind="equipment")

        if result is None:
            raise AssertionError("expected pickup command")
        assert result["cmd_type"] == "pickup_equipment"
        assert result["target_x"] == 66
        assert result["target_y"] == 63

    def test_walk_or_teleport_moves_to_visible_edge_target(self) -> None:
        """Visible edge movement targets are actionable without an approach step."""
        world, self_state = make_world(self_x=64, self_y=64, fuel=150)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(default_count=5),
            100000,
            InMemoryTerrainMap(),
            "",
        )

        result = walk_or_teleport(ctx, 71, 63, pickup_kind=None)

        if result is None:
            raise AssertionError("expected direct edge move command")
        assert result["cmd_type"] == "move"
        assert result["target_x"] == 71
        assert result["target_y"] == 63

    def test_walk_or_teleport_without_terrain_moves_to_visible_edge_target(self) -> None:
        """Visible edge movement works even without a terrain map."""
        world, self_state = make_world(self_x=64, self_y=64, fuel=150)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(default_count=5),
            100000,
            None,
            "",
        )

        result = walk_or_teleport(ctx, 71, 63, pickup_kind=None)

        if result is None:
            raise AssertionError("expected direct edge move command")
        assert result["cmd_type"] == "move"
        assert result["target_x"] == 71
        assert result["target_y"] == 63

    def test_walk_or_teleport_without_terrain_approaches_off_viewport_target(self) -> None:
        """Off-viewport movement clamps to the visible edge without terrain."""
        world, self_state = make_world(self_x=64, self_y=64, fuel=150)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(default_count=5),
            100000,
            None,
            "",
        )

        result = walk_or_teleport(ctx, 72, 63, pickup_kind=None)

        if result is None:
            raise AssertionError("expected off-viewport move to clamp to edge")
        assert result["cmd_type"] == "move"
        assert result["target_x"] == 71
        assert result["target_y"] == 63
