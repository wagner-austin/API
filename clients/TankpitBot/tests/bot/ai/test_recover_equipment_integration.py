"""Integration tests for equipment and fuel recovery arbitration."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.movement import select_exploration_command, viewport_exploration_candidates
from tankpit_bot.bot.ai.types import AIConfigDict, AIStateDict, make_default_ai_config
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.sniffer.world_state import mark_scan_viewport_failed, reset_world_state
from tankpit_bot.state.types import (
    ContainerStateDict,
    TankStateDict,
    ViewportStateDict,
    make_tank_state,
)
from tests.bot.ai._support import (
    make_container,
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _enemy(
    *,
    tank_id: int = 50,
    x: int = 103,
    y: int = 103,
    name: str = "Enemy",
    timestamp_ms: int = 100000,
) -> TankStateDict:
    """Create a visible enemy tank for recovery arbitration tests.

    Args:
        tank_id: Enemy tank id.
        x: Enemy x coordinate.
        y: Enemy y coordinate.
        name: Enemy display name.
        timestamp_ms: Observation timestamp in milliseconds.

    Returns:
        Enemy tank state.
    """
    return make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=2,
        rank=1,
        name=name,
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=timestamp_ms,
    )


class TestRecoverEquipmentPriority:
    """Tests for top-level recovery priority ordering."""

    def setup_method(self) -> None:
        """Reset world-state test globals before each case."""
        reset_world_state()

    def test_equipment_before_combat(self) -> None:
        """Critical equipment recovery preempts combat when supplies are depleted."""
        containers: dict[str, ContainerStateDict] = {
            "106,106": make_container(106, 106, 30, False),
        }
        tanks = {"50": _enemy()}
        world, self_state = make_world(fuel=800, containers=containers, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 103,
                "combat_target_y": 103,
            }
        )
        inventory = make_inventory(dual_count=0, dual_enabled=False)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["command"]["cmd_type"] == "pickup_equipment"

    def test_no_equipment_when_none_visible(self) -> None:
        """Critical equipment depletion searches when no actionable target is visible."""
        world, self_state = make_world(fuel=800, tanks={"50": _enemy()})
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_shoot_ms": 99500,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 103,
                "combat_target_y": 103,
            }
        )
        inventory = make_inventory(dual_count=0, dual_enabled=False)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["reason"] == "search_equipment_local"
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 130
        assert decision["command"]["target_y"] == 100
        assert decision["updated_ai_state"]["patrol_waypoint_index"] == 1
        assert decision["updated_ai_state"]["equipment_search_failures"] == 1

    def test_equipment_at_break_threshold_relocates_in_scanned_viewport(self) -> None:
        """Break-threshold equipment depletion still enters recovery."""
        world, self_state = make_world(fuel=800, tanks={"50": _enemy(x=120, y=100)})
        inventory = make_inventory(dual_count=3, dual_enabled=True, default_count=30)
        inventory["extra_radars"]["count"] = 30

        decision = decide(world, self_state, make_scanned_ai_state(), inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["command"]["cmd_type"] == "teleport"

    def test_critical_homing_shots_interrupts_for_equipment(self) -> None:
        """Critical homing depletion uses the same recovery path as dual/radar."""
        containers: dict[str, ContainerStateDict] = {
            "106,106": make_container(106, 106, 30, False),
        }
        world, self_state = make_world(
            fuel=800,
            containers=containers,
            tanks={"50": _enemy(x=120, y=100)},
        )
        inventory = make_inventory(default_count=30)
        inventory["homing_shots"]["count"] = 3

        decision = decide(world, self_state, make_scanned_ai_state(), inventory, 100000, None)

        assert decision["behavior"]["reason"] == "equipment_critical"
        assert decision["command"]["cmd_type"] == "pickup_equipment"

    def test_homing_at_break_threshold_triggers_equipment_recovery(self) -> None:
        """Break-threshold homing count still enters equipment recovery."""
        world, self_state = make_world(fuel=800, tanks={"50": _enemy(x=120, y=100)})
        inventory = make_inventory(default_count=30)
        inventory["homing_shots"]["count"] = 3

        decision = decide(world, self_state, make_scanned_ai_state(), inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"

    def test_active_combat_interrupted_by_critical_equipment(self) -> None:
        """Locked combat yields to critical equipment recovery."""
        containers: dict[str, ContainerStateDict] = {
            "106,106": make_container(106, 106, 30, False),
        }
        world, self_state = make_world(fuel=800, containers=containers, tanks={"50": _enemy()})
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 103,
                "combat_target_y": 103,
            }
        )
        inventory = make_inventory(default_count=3)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["reason"] == "equipment_critical"
        assert decision["command"]["cmd_type"] == "pickup_equipment"

    def test_active_combat_with_subcritical_fuel_interrupts_for_collection(self) -> None:
        """Fuel recovery outranks combat once fuel drops below the threshold."""
        containers: dict[str, ContainerStateDict] = {
            "104,104": make_container(104, 104, 900, True),
        }
        world, self_state = make_world(fuel=250, containers=containers, tanks={"50": _enemy()})
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 103,
                "combat_target_y": 103,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"
        assert decision["command"]["cmd_type"] == "pickup_fuel"
        assert decision["updated_ai_state"]["combat_target_id"] == -1

    def test_blocked_equipment_uses_final_pickup_command(self) -> None:
        """Blocked equipment in view still uses the final pickup target."""
        containers: dict[str, ContainerStateDict] = {
            "128,126": make_container(128, 126, 0, False),
        }
        world, self_state = make_world(self_x=129, self_y=125, fuel=800, containers=containers)
        inventory = make_inventory(default_count=3)
        terrain = InMemoryTerrainMap(
            terrain_data={
                (128, 126): "W",
                (127, 126): "W",
                (128, 127): "W",
            }
        )

        decision = decide(world, self_state, make_scanned_ai_state(), inventory, 100000, terrain)

        assert decision["command"]["cmd_type"] == "pickup_equipment"
        assert decision["command"]["target_x"] == 128
        assert decision["command"]["target_y"] == 126

    def test_blocked_equipment_without_adjacent_landing_is_skipped(self) -> None:
        """Blocked equipment without a landing tile forces equipment sensing."""
        containers: dict[str, ContainerStateDict] = {
            "128,126": make_container(128, 126, 0, False),
        }
        world, self_state = make_world(
            self_x=130,
            self_y=124,
            fuel=800,
            containers=containers,
            scanned=False,
        )
        inventory = make_inventory(default_count=5)
        terrain = InMemoryTerrainMap(
            terrain_data={
                (128, 126): "W",
                (129, 126): "W",
                (127, 126): "W",
                (128, 127): "#",
                (128, 125): "#",
            }
        )

        decision = decide(world, self_state, make_scanned_ai_state(), inventory, 100000, terrain)

        assert decision["behavior"]["reason"] == "radar_for_equipment"
        assert decision["command"]["cmd_type"] == "radar"

    def test_adjacent_blocked_equipment_uses_pickup_move(self) -> None:
        """Adjacent blocked equipment still produces a pickup command."""
        containers: dict[str, ContainerStateDict] = {
            "128,126": make_container(128, 126, 0, False),
        }
        world, self_state = make_world(self_x=129, self_y=126, fuel=800, containers=containers)
        inventory = make_inventory(default_count=5)
        terrain = InMemoryTerrainMap(terrain_data={(128, 126): "W"})

        decision = decide(world, self_state, make_scanned_ai_state(), inventory, 100000, terrain)

        assert decision["command"]["cmd_type"] == "pickup_equipment"

    def test_locked_combat_with_critical_fuel_collects_fuel(self) -> None:
        """Critical fuel interrupts a locked combat phase."""
        containers: dict[str, ContainerStateDict] = {
            "101,100": make_container(101, 100, 700, True),
        }
        world, self_state = make_world(fuel=250, containers=containers, tanks={"50": _enemy()})
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 103,
                "combat_target_y": 103,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"
        assert decision["command"]["cmd_type"] == "pickup_fuel"

    def test_critical_fuel_beats_critical_equipment(self) -> None:
        """Fuel recovery outranks even critical equipment depletion."""
        containers: dict[str, ContainerStateDict] = {
            "101,100": make_container(101, 100, 700, True),
            "102,100": make_container(102, 100, 0, False),
        }
        world, self_state = make_world(fuel=250, containers=containers)
        inventory = make_inventory(dual_count=3, default_count=30)
        inventory["extra_radars"]["count"] = 4

        decision = decide(world, self_state, make_scanned_ai_state(), inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"
        assert decision["behavior"]["reason"] == "fuel=700"
        assert decision["command"]["cmd_type"] == "pickup_fuel"


class TestRecoverEquipmentSearch:
    """Tests for equipment search and related recovery transitions."""

    def setup_method(self) -> None:
        """Reset world-state test globals before each case."""
        reset_world_state()

    def test_critical_equipment_search_uses_radar_when_ready(self) -> None:
        """Critical equipment depletion scans before relocating when radar is ready."""
        world, self_state = make_world(fuel=800, scanned=False)
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=30)

        decision = decide(world, self_state, make_scanned_ai_state(), inventory, 100000, None)

        assert decision["behavior"]["reason"] == "radar_for_equipment"
        assert decision["command"]["cmd_type"] == "radar"

    def test_critical_equipment_new_unscanned_viewport_ignores_scan_cooldown(self) -> None:
        """A new unscanned viewport bypasses the global radar cooldown."""
        world, self_state = make_world(fuel=800, scanned=False)
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_scan_ms": 99999})
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=30)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["reason"] == "radar_for_equipment"
        assert decision["command"]["cmd_type"] == "radar"

    def test_critical_equipment_search_relocates_when_radar_on_cooldown(self) -> None:
        """Critical equipment depletion relocates when radar is cooling down."""
        world, self_state = make_world(fuel=800)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 94000,
                "patrol_waypoint_index": 2,
            }
        )
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=30)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["reason"] == "search_equipment_local"
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 70
        assert decision["command"]["target_y"] == 100
        assert decision["updated_ai_state"]["patrol_waypoint_index"] == 3
        assert decision["updated_ai_state"]["equipment_search_failures"] == 1

    def test_equipment_search_bails_out_after_max_failures(self) -> None:
        """Critical equipment search stays in recovery after hitting the failure cap."""
        world, self_state = make_world(fuel=800)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 94000,
                "equipment_search_failures": 3,
            }
        )
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=30)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"

    def test_equipment_search_edge_walks_when_teleport_unaffordable(self) -> None:
        """Unaffordable critical equipment search degrades to an edge walk.

        Regression guard for live run 20260610-000x: this path used to
        raise and kill the bot process mid-game.
        """
        world, self_state = make_world(fuel=550)
        config = AIConfigDict(
            **{
                **make_default_ai_config(),
                "equip_search_hop_distance": 90,
            }
        )
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "config": config,
                "last_scan_ms": 99500,
                "last_map_open_ms": 99500,
            }
        )
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=0)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["behavior"]["reason"] == "forage_radar"
        assert decision["command"]["cmd_type"] == "radar"

    def test_equipment_search_skips_when_fuel_too_low(self) -> None:
        """Equipment search defers to fuel recovery when fuel is already low."""
        world, self_state = make_world(fuel=250)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 99500,
            }
        )
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=0)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"

    def test_reachable_container_behind_wall_uses_final_pickup_target(self) -> None:
        """In-viewport terrain detours preserve the final pickup target."""
        containers: dict[str, ContainerStateDict] = {
            "103,100": make_container(103, 100, 700, True),
        }
        world, self_state = make_world(fuel=250, containers=containers)
        terrain = InMemoryTerrainMap({(102, 100): "#"})

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            terrain,
        )

        assert decision["command"]["cmd_type"] == "pickup_fuel"
        assert decision["command"]["target_x"] == 103
        assert decision["command"]["target_y"] == 100

    def test_low_fuel_without_targets_uses_radar(self) -> None:
        """Low fuel scans when no actionable fuel target exists."""
        world, self_state = make_world(fuel=300, scanned=False)

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
        )

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"
        assert decision["behavior"]["reason"] == "radar_for_fuel"
        assert decision["command"]["cmd_type"] == "radar"

    def test_low_fuel_cache_only_target_in_unscanned_viewport_collects(self) -> None:
        """Visible critical fuel still collects directly in an unscanned viewport."""
        containers: dict[str, ContainerStateDict] = {
            "104,100": make_container(104, 100, 700, True),
        }
        world, self_state = make_world(fuel=300, containers=containers)
        world["scanned_viewports"] = {}

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
        )

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"
        assert decision["command"]["cmd_type"] == "pickup_fuel"

    def test_low_fuel_new_unscanned_viewport_ignores_global_scan_cooldown(self) -> None:
        """A newly entered unconfirmed viewport radars immediately."""
        world, self_state = make_world(fuel=300, scanned=False)
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_scan_ms": 99999})

        decision = decide(world, self_state, ai_state, make_inventory(), 100000, None)

        assert decision["behavior"]["reason"] == "radar_for_fuel"
        assert decision["command"]["cmd_type"] == "radar"

    def test_low_fuel_recent_failed_scan_walks_instead_of_repeating_radar(self) -> None:
        """Recent scan failure suppresses immediate radar retry."""
        world, self_state = make_world(fuel=300, scanned=False)
        viewport = world["viewport"]
        mark_scan_viewport_failed(viewport["left"], viewport["top"], 100000)

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
        )

        assert decision["behavior"]["reason"] == "search_fuel_local"
        assert decision["command"]["cmd_type"] == "teleport"

    def test_low_fuel_blocked_search_with_visible_threats_falls_back_to_map(self) -> None:
        """Blocked low-fuel exploration does not break recovery ownership."""
        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=300,
            tanks={"50": _enemy(x=120, y=100)},
        )
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_scan_ms": 99999})
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
        terrain_data: dict[tuple[int, int], str] = dict.fromkeys(
            viewport_exploration_candidates(ctx),
            "W",
        )
        for candidate_x, candidate_y in viewport_exploration_candidates(ctx):
            terrain_data[(candidate_x - 1, candidate_y)] = "#"
            terrain_data[(candidate_x + 1, candidate_y)] = "#"
            terrain_data[(candidate_x, candidate_y - 1)] = "#"
            terrain_data[(candidate_x, candidate_y + 1)] = "#"
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"

    def test_exploration_candidates_omit_self_and_duplicates(self) -> None:
        """Exploration candidates omit the current tile and duplicate entries."""
        world, self_state = make_world(self_x=107, self_y=100, fuel=800)
        world["viewport"] = ViewportStateDict(left=92, top=92, width=16, height=16)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        candidates = viewport_exploration_candidates(ctx)

        assert (107, 100) not in candidates
        assert len(candidates) == len(set(candidates))

    def test_exploration_skips_blocked_target_and_uses_next_candidate(self) -> None:
        """Exploration skips blocked edges and falls through to the next candidate."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=550)
        terrain = InMemoryTerrainMap(
            terrain_data={
                (107, 107): "W",
                (106, 107): "W",
                (107, 106): "W",
                (108, 107): "#",
                (107, 108): "#",
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

        exploration = select_exploration_command(ctx)

        if exploration is None:
            raise AssertionError("expected exploration command")
        candidate_x, candidate_y, command = exploration
        assert (candidate_x, candidate_y) != (107, 107)
        assert command["cmd_type"] in ("move", "teleport")

    def test_locked_combat_with_zero_dual_releases_to_equipment(self) -> None:
        """Combat lock releases once dual shots are critically depleted."""
        world, self_state = make_world(fuel=800, tanks={"50": _enemy()})
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 103,
                "combat_target_y": 103,
            }
        )
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=30)
        inventory["extra_radars"]["count"] = 30

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"

    def test_killed_target_releases_combat_lock_for_recovery(self) -> None:
        """Killed locked targets release combat so recovery can proceed."""
        containers: dict[str, ContainerStateDict] = {
            "101,100": make_container(101, 100, 700, True),
        }
        tanks: dict[str, TankStateDict] = {
            "50": _enemy(timestamp_ms=100000),
            "60": _enemy(tank_id=60, x=105, y=105, name="Enemy2", timestamp_ms=100000),
        }
        world, self_state = make_world(fuel=250, tanks=tanks, containers=containers)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 103,
                "combat_target_y": 103,
                "killed_tank_ids": {"50": 99000},
            }
        )

        decision = decide(world, self_state, ai_state, make_inventory(), 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"

    def test_new_target_selection_skips_recently_killed_enemy(self) -> None:
        """Threat acquisition skips enemies still on the kill cooldown."""
        tanks: dict[str, TankStateDict] = {
            "50": _enemy(name="DeadEnemy", timestamp_ms=100000),
            "60": _enemy(tank_id=60, x=104, y=103, name="LiveEnemy", timestamp_ms=100000),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "killed_tank_ids": {"50": 99900},
            }
        )

        decision = decide(world, self_state, ai_state, make_inventory(), 100000, None)

        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["behavior"]["reason"] == "find LiveEnemy"
