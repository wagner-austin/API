"""Helper-focused tests for recovery planning utilities."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    compute_equipment,
    expire_kills,
    filter_killed_tanks,
    require_command,
)
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.recover_equipment_mode import (
    select_equipment_target as _select_equipment_target_command,
)
from tankpit_bot.bot.ai.recover_equipment_mode import (
    try_search_critical_equipment,
)
from tankpit_bot.bot.ai.resource_search import (
    local_resource_search_hop,
    make_resource_search_hop,
    select_fuel_dot_hop,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.bot.types import make_move_command
from tankpit_bot.sniffer.world_state import mark_move_target_failed, reset_world_state
from tankpit_bot.state.types import ContainerStateDict, TankStateDict, make_mine_state
from tests.bot.ai._support import (
    make_container,
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _enemy(*, x: int, y: int, timestamp_ms: int = 100000) -> TankStateDict:
    """Create a visible enemy tank for helper tests.

    Args:
        x: Enemy x coordinate.
        y: Enemy y coordinate.
        timestamp_ms: Observation timestamp.

    Returns:
        Enemy tank state.
    """
    from tankpit_bot.state.types import make_tank_state

    return make_tank_state(
        tank_id=50,
        x=x,
        y=y,
        team=2,
        rank=1,
        name="Enemy",
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=timestamp_ms,
        last_position_update_ms=timestamp_ms,
    )


class TestRecoveryHelpers:
    """Tests for helper-level recovery planning behavior."""

    def setup_method(self) -> None:
        """Reset world-state test globals before each case."""
        reset_world_state()

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

    def test_local_equipment_search_hop_rotates_cardinal(self) -> None:
        """Local equipment search rotates through cardinal directions."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(dual_count=5, default_count=30),
            100000,
            None,
            "",
        )

        hop_x, hop_y, next_index = local_resource_search_hop(ctx)

        assert hop_x == 130
        assert hop_y == 100
        assert next_index == 1

    def test_local_equipment_search_hop_skips_clamped_edge_hops(self) -> None:
        """Edge-clamped hops onto covered ground are skipped, not taken.

        At (250,250) the east hop clamps 280->254 (4 tiles onto the
        already-scanned viewport) and south clamps the same way; both
        are degenerate re-visits. The search advances to west ring 1,
        the first hop into unscanned ground, consuming three cycle
        positions. The old behavior returned the clamped corner hop and
        produced the live 20260610 corner-trap loop.
        """
        world, self_state = make_world(self_x=250, self_y=250, fuel=800)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(dual_count=5, default_count=30),
            100000,
            None,
            "",
        )

        hop_x, hop_y, next_index = local_resource_search_hop(ctx)

        assert hop_x == 220
        assert hop_y == 250
        assert next_index == 3

    def test_local_search_hop_falls_back_when_everything_is_covered(self) -> None:
        """With every cycle position covered, the raw indexed hop is returned.

        Coverage expires within the scan TTL, so the fallback hop
        self-heals instead of leaving the owner with no target.
        """
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        hop_targets = [
            (130, 100),
            (100, 130),
            (70, 100),
            (100, 70),
            (160, 100),
            (100, 160),
            (40, 100),
            (100, 40),
            (190, 100),
            (100, 190),
            (10, 100),
            (100, 10),
        ]
        for target_x, target_y in hop_targets:
            world["scanned_viewports"][f"{target_x - 8},{target_y - 8}"] = 100000
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(dual_count=5, default_count=30),
            100000,
            None,
            "",
        )

        hop_x, hop_y, next_index = local_resource_search_hop(ctx)

        assert (hop_x, hop_y) == (130, 100)
        assert next_index == 1

    def test_local_search_hop_escapes_map_corner(self) -> None:
        """A corner position never re-targets the same clamped corner tile.

        Live run 20260610: at (1,254) the west/south hops clamped back
        onto the corner, and the bot teleported to (1,254)/(3,254)/
        (5,254) repeatedly, re-radaring the same ground. With the
        patrol index pointing west, the search must skip the
        zero-displacement clamp and hop north into fresh ground.
        """
        world, self_state = make_world(self_x=1, self_y=254, fuel=800)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "patrol_waypoint_index": 2,
            }
        )
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(dual_count=5, default_count=30),
            100000,
            None,
            "",
        )

        hop_x, hop_y, next_index = local_resource_search_hop(ctx)

        assert (hop_x, hop_y) != (1, 254)
        assert hop_y == 224
        assert next_index == 4

    def test_local_equipment_search_hop_ring_wraps_at_cap(self) -> None:
        """The hop ring wraps instead of growing with the session-long index.

        Regression guard for live run 20260610-000x: an ever-growing
        patrol index produced 90+ tile hops costing more fuel than the
        bot held, leaving the recovery owner with no affordable action.
        Index 100 wraps to cycle 4 (east, ring 2) -> 60 tiles, not 780.
        """
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "patrol_waypoint_index": 100,
            }
        )
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(dual_count=5, default_count=30),
            100000,
            None,
            "",
        )

        hop_x, hop_y, next_index = local_resource_search_hop(ctx)

        assert hop_x == 160
        assert hop_y == 100
        assert next_index == 101

    def test_select_fuel_dot_hop_returns_none_with_empty_atlas(self) -> None:
        """No fuel dots means no dot-guided target."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        assert select_fuel_dot_hop(ctx) is None

    def test_select_fuel_dot_hop_picks_nearest_dot(self) -> None:
        """The nearest worthwhile dot wins over farther ones."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        world["map_fuel_dots"] = {"140,100": 100000, "120,100": 100000}
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        assert select_fuel_dot_hop(ctx) == (120, 100)

    def test_select_fuel_dot_hop_skips_dot_in_scanned_ground(self) -> None:
        """A dot whose tile sits inside fresh scan coverage is refuted, not a lead."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        world["map_fuel_dots"] = {"120,100": 100000, "140,100": 100000}
        world["scanned_viewports"]["112,92"] = 100000
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        assert select_fuel_dot_hop(ctx) == (140, 100)

    def test_select_fuel_dot_hop_skips_degenerate_close_dot(self) -> None:
        """A dot within the degenerate-hop displacement is never a teleport target."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        world["map_fuel_dots"] = {"102,100": 100000, "130,100": 100000}
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        assert select_fuel_dot_hop(ctx) == (130, 100)

    def test_select_fuel_dot_hop_returns_none_when_unaffordable(self) -> None:
        """An unaffordable nearest dot ends the scan -- farther dots cost more."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        world["map_fuel_dots"] = {"160,100": 100000, "200,100": 100000}
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        assert select_fuel_dot_hop(ctx) is None

    def test_select_fuel_dot_hop_returns_none_when_all_dots_covered(self) -> None:
        """Every dot inside scanned ground leaves no dot-guided target."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        world["map_fuel_dots"] = {"120,100": 100000}
        world["scanned_viewports"]["112,92"] = 100000
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        assert select_fuel_dot_hop(ctx) is None

    def test_select_fuel_dot_hop_revives_dot_after_scan_coverage_expires(self) -> None:
        """An old scan no longer refutes a dot -- containers respawn."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        world["map_fuel_dots"] = {"120,100": 100000}
        world["scanned_viewports"]["112,92"] = 100000 - 46000
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        assert select_fuel_dot_hop(ctx) == (120, 100)

    def test_select_fuel_dot_hop_dot_outside_scan_row_is_a_lead(self) -> None:
        """A scan covering the dot's column but not its row does not refute it."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        world["map_fuel_dots"] = {"120,140": 100000}
        world["scanned_viewports"]["112,92"] = 100000
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        assert select_fuel_dot_hop(ctx) == (120, 140)

    def test_dot_guided_hop_teleports_to_dot_without_consuming_patrol(self) -> None:
        """A dot-guided hop targets the dot and leaves the ring patrol untouched."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        world["map_fuel_dots"] = {"120,110": 100000}
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        decision = make_resource_search_hop(
            ctx,
            mode="COLLECT_FUEL",
            score=900,
            reason="search_fuel_local",
            fuel_dot_guided=True,
        )

        if decision is None:
            raise AssertionError("expected dot-guided teleport decision")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 120
        assert decision["command"]["target_y"] == 110
        assert decision["updated_ai_state"]["patrol_waypoint_index"] == 0

    def test_dot_guided_hop_falls_back_to_ring_when_atlas_empty(self) -> None:
        """Without dots the dot-guided hop degrades to the ring patrol."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        decision = make_resource_search_hop(
            ctx,
            mode="COLLECT_FUEL",
            score=900,
            reason="search_fuel_local",
            fuel_dot_guided=True,
        )

        if decision is None:
            raise AssertionError("expected ring-patrol teleport decision")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 130
        assert decision["command"]["target_y"] == 100
        assert decision["updated_ai_state"]["patrol_waypoint_index"] == 1

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

    def test_try_search_critical_equipment_returns_none_when_fuel_low(self) -> None:
        """Critical equipment search stops once fuel is already low."""
        world, self_state = make_world(fuel=250)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(dual_count=0, dual_enabled=False, default_count=0),
            100000,
            None,
            "",
        )

        assert try_search_critical_equipment(ctx) is None

    def test_select_equipment_target_command_picks_up_mined_tile(self) -> None:
        """Equipment target selection allows pickup on mined tiles."""
        containers: dict[str, ContainerStateDict] = {
            "103,100": make_container(103, 100, 0, False),
        }
        world, self_state = make_world(self_x=100, self_y=100, fuel=800, containers=containers)
        world["mines"] = {
            "103,100": make_mine_state(x=103, y=100, mine_type=0, tank_id=-1, team=1),
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

        result = _select_equipment_target_command(ctx, allow_unreachable=True)

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
        world, self_state = make_world(self_x=100, self_y=100, fuel=300, containers=containers)
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
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
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
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        world["mines"] = {"107,100": make_mine_state(x=107, y=100, mine_type=0, tank_id=-1, team=1)}
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
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        world["mines"] = {"107,100": make_mine_state(x=107, y=100, mine_type=0, tank_id=-1, team=1)}
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
        world, self_state = make_world(self_x=64, self_y=64, fuel=300)
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
        world, self_state = make_world(self_x=64, self_y=64, fuel=300)
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
        world, self_state = make_world(self_x=64, self_y=64, fuel=300)
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

    def test_decide_picks_up_visible_edge_equipment(self) -> None:
        """Visible edge equipment is actionable without an approach step."""
        containers: dict[str, ContainerStateDict] = {
            "66,63": make_container(66, 63, 0, False),
        }
        world, self_state = make_world(self_x=64, self_y=64, fuel=800, containers=containers)
        inventory = make_inventory(default_count=30)
        inventory["dual_shots"]["count"] = 0
        inventory["dual_shots"]["enabled"] = False

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            inventory,
            100000,
            InMemoryTerrainMap(),
        )

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["command"]["cmd_type"] == "pickup_equipment"
        assert decision["command"]["target_x"] == 66
        assert decision["command"]["target_y"] == 63

    def test_non_emergency_equipment_low_does_not_preempt_hunt(self) -> None:
        """Non-emergency equipment depletion no longer overrides HUNT ownership."""
        containers: dict[str, ContainerStateDict] = {
            "106,106": make_container(106, 106, 30, False),
        }
        world, self_state = make_world(fuel=800, containers=containers)
        inventory = make_inventory(default_count=30)
        inventory["missile_shots"]["count"] = 5
        inventory["dual_shots"]["count"] = 20
        inventory["extra_radars"]["count"] = 20

        decision = decide(world, self_state, make_scanned_ai_state(), inventory, 100000, None)

        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["command"]["cmd_type"] == "map_open"

    def test_non_emergency_equipment_low_does_not_enter_recovery_search(self) -> None:
        """Non-emergency equipment depletion leaves HUNT in charge of the tick."""
        containers: dict[str, ContainerStateDict] = {
            "107,107": make_container(107, 107, 0, False),
        }
        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=800,
            containers=containers,
            tanks={"50": _enemy(x=120, y=100)},
        )
        inventory = make_inventory(default_count=30)
        inventory["missile_shots"]["count"] = 5
        terrain = InMemoryTerrainMap(
            terrain_data={
                (107, 107): "W",
                (108, 107): "W",
                (106, 107): "W",
                (107, 108): "#",
                (107, 106): "#",
            }
        )

        decision = decide(world, self_state, make_scanned_ai_state(), inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "HUNT"

    def test_non_emergency_equipment_low_keeps_hunt_even_with_visible_equipment(self) -> None:
        """Visible equipment does not override HUNT when reserves are not at break levels."""
        containers: dict[str, ContainerStateDict] = {
            "103,100": make_container(103, 100, 0, False),
            "106,100": make_container(106, 100, 0, False),
        }
        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=800,
            containers=containers,
            tanks={"50": _enemy(x=103, y=100, timestamp_ms=100000)},
        )
        inventory = make_inventory(default_count=30)
        inventory["missile_shots"]["count"] = 5

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            inventory,
            100000,
            InMemoryTerrainMap(),
        )

        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["command"]["cmd_type"] == "map_open"

    def test_non_emergency_equipment_low_does_not_force_outer_ring_search(self) -> None:
        """Blocked outer-ring equipment does not start recovery outside break thresholds."""
        containers: dict[str, ContainerStateDict] = {
            "129,184": make_container(129, 184, 0, False),
        }
        world, self_state = make_world(
            self_x=138,
            self_y=192,
            fuel=800,
            containers=containers,
            tanks={"50": _enemy(x=120, y=100)},
        )
        inventory = make_inventory(default_count=30)
        inventory["missile_shots"]["count"] = 5
        terrain = InMemoryTerrainMap(
            terrain_data={
                (130, 184): "W",
                (131, 184): "W",
                (129, 184): "W",
                (130, 185): "#",
                (130, 183): "#",
            }
        )

        decision = decide(world, self_state, make_scanned_ai_state(), inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "HUNT"

    def test_waypoint_clamped_to_viewport_bounds(self) -> None:
        """A* waypoints never produce moves outside the visible viewport."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        terrain = InMemoryTerrainMap(
            terrain_data={(row, col): "#" for row in range(92, 100) for col in range(92, 100)}
        )
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            terrain,
            "",
        )

        result = walk_or_teleport(ctx, 91, 91, pickup_kind=None)

        if result is not None and result["cmd_type"] == "move":
            viewport = ctx.world["viewport"]
            left = viewport["left"]
            top = viewport["top"]
            right = left + viewport["width"] - 1
            bottom = top + viewport["height"] - 1
            assert left <= result["target_x"] <= right
            assert top <= result["target_y"] <= bottom

    def test_walk_or_teleport_rejects_failed_move_target(self) -> None:
        """Recently failed move targets are skipped."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            InMemoryTerrainMap(),
            "",
        )
        mark_move_target_failed(107, 100, 90000)

        assert walk_or_teleport(ctx, 107, 100, pickup_kind=None) is None

    def test_walk_or_teleport_rejects_enemy_occupied_direct_move(self) -> None:
        """Direct moves to occupied enemy tiles are rejected."""
        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=300,
            tanks={"50": _enemy(x=107, y=100, timestamp_ms=100000)},
        )
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            InMemoryTerrainMap(),
            "",
        )

        assert walk_or_teleport(ctx, 107, 100, pickup_kind=None) is None

    def test_walk_or_teleport_rejects_occupied_move_without_terrain(self) -> None:
        """Enemy occupancy still blocks direct moves without terrain."""
        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=300,
            tanks={"50": _enemy(x=107, y=100, timestamp_ms=100000)},
        )
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        assert walk_or_teleport(ctx, 107, 100, pickup_kind=None) is None

    def test_walk_or_teleport_returns_none_for_out_of_bounds_target(self) -> None:
        """Out-of-bounds target returns None via teleport fallback (landing=None)."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        terrain = InMemoryTerrainMap(
            terrain_data={
                (100, 100): InMemoryTerrainMap.GROUND,
                (101, 100): InMemoryTerrainMap.ROCK,
                (100, 101): InMemoryTerrainMap.ROCK,
                (99, 100): InMemoryTerrainMap.ROCK,
                (100, 99): InMemoryTerrainMap.ROCK,
            }
        )
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            terrain,
            "",
        )

        assert walk_or_teleport(ctx, 300, 300, pickup_kind="fuel") is None

    def test_walk_or_teleport_rejects_mined_move_without_terrain(self) -> None:
        """Mine occupancy blocks direct moves without terrain."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        world["mines"] = {"107,100": make_mine_state(x=107, y=100, mine_type=0, tank_id=-1, team=1)}
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        assert walk_or_teleport(ctx, 107, 100, pickup_kind=None) is None
