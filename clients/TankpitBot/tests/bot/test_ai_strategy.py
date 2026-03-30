"""Tests for AI strategy decide() function."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import (
    AIStateDict,
    EnemyThreatDict,
    make_default_ai_config,
    make_initial_ai_state,
)
from tankpit_bot.bot.ai_strategy import (
    _combat_landing_tile,
    _compute_equipment,
    _DecideCtx,
    _expire_kills,
    _filter_killed_tanks,
    _is_occupied_by_enemy,
    _local_equipment_search_hop,
    _require_command,
    _walk_or_teleport,
    _waypoint_move_command,
    decide,
)
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.sniffer.world_state import (
    mark_move_target_failed,
    reset_world_state,
)
from tankpit_bot.state.types import (
    ContainerStateDict,
    SelfStateDict,
    TankStateDict,
    ViewportStateDict,
    WorldStateDict,
    make_container_state,
)
from tests.fakes import FakeTerrainMap


def _c(
    x: int,
    y: int,
    volume: int,
    is_fuel: bool,
    timestamp_ms: int = 100000,
) -> ContainerStateDict:
    """Shorthand for creating ContainerStateDict in tests.

    Default timestamp_ms matches the standard test "now" of 100000,
    keeping containers within the freshness window.
    """
    return make_container_state(
        x=x,
        y=y,
        volume=volume,
        is_fuel=is_fuel,
        timestamp_ms=timestamp_ms,
    )


def _scanned_ai_state() -> AIStateDict:
    """Create AI state with radar already done (last_scan_ms=1).

    Most tests don't want radar to fire first; only the radar-specific
    tests should use make_initial_ai_state() which has last_scan_ms=0.
    """
    return AIStateDict(**{**make_initial_ai_state(), "last_scan_ms": 1})


def _make_world(
    *,
    self_x: int = 100,
    self_y: int = 100,
    fuel: int = 800,
    containers: dict[str, ContainerStateDict] | None = None,
    tanks: dict[str, TankStateDict] | None = None,
) -> tuple[WorldStateDict, SelfStateDict]:
    """Create world and self state for testing.

    Returns:
        Tuple of (world_state, self_state).
    """
    self_state = SelfStateDict(
        tank_id=1,
        x=self_x,
        y=self_y,
        team=1,
        rank=2,
        fuel=fuel,
        leaderboard_position=5,
    )
    viewport = ViewportStateDict(left=self_x - 9, top=self_y - 9, width=18, height=18)
    world = WorldStateDict(
        self_state=self_state,
        tanks=tanks or {},
        containers=containers or {},
        mines={},
        terrain={},
        viewport=viewport,
        timestamp_ms=0,
    )
    return world, self_state


def _make_inventory(
    *,
    dual_count: int = 30,
    dual_enabled: bool = True,
    default_count: int = 30,
) -> InventoryState:
    """Create a basic inventory state.

    Args:
        dual_count: Count for dual shots item.
        dual_enabled: Whether dual shots are enabled.
        default_count: Default count for all other items.

    Returns:
        InventoryState with the specified values.
    """
    return InventoryState(
        armor_shields=InventoryItem(count=default_count, enabled=True),
        dual_shots=InventoryItem(count=dual_count, enabled=dual_enabled),
        missile_shots=InventoryItem(count=default_count, enabled=True),
        homing_shots=InventoryItem(count=default_count, enabled=True),
        extra_radars=InventoryItem(count=default_count, enabled=True),
    )


class TestDecideProactiveRadar:
    """Tests for proactive radar tactical override."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_no_emergency_fuel_when_fuel_high(self) -> None:
        """decide() skips emergency fuel when fuel is above threshold."""
        world, self_state = _make_world(fuel=1000)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert "emergency_fuel" not in decision["behavior"]["reason"]

    def test_collects_fuel_when_low_and_containers_visible(self) -> None:
        """decide() collects fuel when below threshold and containers visible."""
        containers: dict[str, ContainerStateDict] = {
            "95,95": _c(95, 95, 700, True),
        }
        world, self_state = _make_world(fuel=400, containers=containers)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "pickup_move"
        assert decision["behavior"]["mode"] == "COLLECT_FUEL"


class TestDecideTeleportFuelGuard:
    """Tests for teleport fuel cost guard."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_teleport_search_blocked_when_fuel_too_low(self) -> None:
        """decide() skips teleport search when fuel can't cover cost + critical.

        With low fuel, no visible containers, and radar on cooldown, the planner
        must avoid turning edge scouting into a costly teleport.
        """
        world, self_state = _make_world(fuel=250)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=-1,
            combat_target_x=0,
            combat_target_y=0,
            combat_phase="none",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()

        # fuel=250, teleport_cost=100, critical=200 → 250 < 300 → teleport blocked
        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # Teleport was blocked, so should NOT be teleport
        assert decision["command"]["cmd_type"] != "teleport"


class TestDecideEquipmentDepletion:
    """Tests for dual shots depletion override."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_equipment_before_combat(self) -> None:
        """decide() prioritizes equipment when any item < 10, even with enemies.

        Equipment comes before combat — the bot needs dual shots and
        radars to fight effectively.
        """
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=103,
                y=103,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        containers: dict[str, ContainerStateDict] = {
            "106,106": _c(106, 106, 30, False),
        }
        world, self_state = _make_world(fuel=800, containers=containers, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=103,
            combat_target_y=103,
            combat_phase="none",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        # dual_count=0 → equipment_low triggers
        inventory = _make_inventory(dual_count=0, dual_enabled=False)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # Equipment collection fires before combat
        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["command"]["cmd_type"] == "pickup_move"

    def test_no_equipment_when_none_visible(self) -> None:
        """Critical equipment depletion triggers search when none is visible.

        With dual shots depleted and no actionable equipment in view, the bot
        should search for equipment instead of falling through to combat hunt.
        """
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=103,
                y=103,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=99500,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=103,
            combat_target_y=103,
            combat_phase="none",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory(dual_count=0, dual_enabled=False)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["behavior"]["reason"] == "search_equipment_local"
        assert decision["command"]["cmd_type"] == "teleport"
        # Local hop from (100,100) direction 0=(+1,0) dist=15 → (115,100)
        assert decision["command"]["target_x"] == 115
        assert decision["command"]["target_y"] == 100
        assert decision["updated_ai_state"]["patrol_waypoint_index"] == 1
        assert decision["updated_ai_state"]["equipment_search_failures"] == 1

    def test_noncritical_equipment_logs_and_falls_through_to_hunt(self) -> None:
        """Noncritical equipment depletion does not enter emergency search mode."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(dual_count=12, dual_enabled=True, default_count=30)
        inventory["extra_radars"]["count"] = 30

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["command"]["cmd_type"] == "map_open"

    def test_active_combat_interrupted_by_critical_equipment(self) -> None:
        """Critical equipment recovery preempts a locked combat engagement."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=103,
                y=103,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        containers: dict[str, ContainerStateDict] = {
            "106,106": _c(106, 106, 30, False),
        }
        world, self_state = _make_world(fuel=800, containers=containers, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=103,
            combat_target_y=103,
            combat_phase="engaging",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory(default_count=5)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["behavior"]["reason"] == "equipment_critical"
        assert decision["command"]["cmd_type"] == "pickup_move"

    def test_active_combat_not_interrupted_by_noncritical_fuel(self) -> None:
        """Locked combat does not switch to ordinary low-fuel collection mid-fight."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=103,
                y=103,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        containers: dict[str, ContainerStateDict] = {
            "104,104": _c(104, 104, 900, True),
        }
        world, self_state = _make_world(fuel=476, containers=containers, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=103,
            combat_target_y=103,
            combat_phase="engaging",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["command"]["cmd_type"] == "shoot"

    def test_blocked_equipment_teleports_to_adjacent_passable_tile(self) -> None:
        """Blocked equipment targets use a safe adjacent landing tile for teleport."""
        containers: dict[str, ContainerStateDict] = {
            "128,126": _c(128, 126, 0, False),
        }
        world, self_state = _make_world(self_x=130, self_y=124, fuel=800, containers=containers)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(default_count=5)
        terrain_data: dict[tuple[int, int], str] = {
            (128, 126): "W",
            (127, 126): "W",
            (128, 127): "W",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 129
        assert decision["command"]["target_y"] == 126

    def test_blocked_equipment_without_adjacent_landing_is_skipped(self) -> None:
        """Blocked critical equipment triggers search when no legal landing tile exists."""
        containers: dict[str, ContainerStateDict] = {
            "128,126": _c(128, 126, 0, False),
        }
        world, self_state = _make_world(self_x=130, self_y=124, fuel=800, containers=containers)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(default_count=5)
        terrain_data: dict[tuple[int, int], str] = {
            (128, 126): "W",
            (129, 126): "W",
            (127, 126): "W",
            (128, 127): "#",
            (128, 125): "#",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["behavior"]["reason"] == "radar_for_equipment"
        assert decision["command"]["cmd_type"] == "radar"

    def test_adjacent_blocked_equipment_uses_pickup_move(self) -> None:
        """When already beside a blocked container, pickup_move is still used."""
        containers: dict[str, ContainerStateDict] = {
            "128,126": _c(128, 126, 0, False),
        }
        world, self_state = _make_world(self_x=129, self_y=126, fuel=800, containers=containers)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(default_count=5)
        terrain_data: dict[tuple[int, int], str] = {
            (128, 126): "W",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["command"]["cmd_type"] == "pickup_move"

    def test_locked_combat_with_low_fuel_stays_in_combat(self) -> None:
        """Noncritical low fuel does not interrupt a locked combat phase."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=103,
                y=103,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        containers: dict[str, ContainerStateDict] = {
            "101,100": _c(101, 100, 700, True),
        }
        world, self_state = _make_world(fuel=400, containers=containers, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=103,
            combat_target_y=103,
            combat_phase="engaging",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["command"]["cmd_type"] == "shoot"

    def test_critical_equipment_beats_noncritical_low_fuel(self) -> None:
        """Critical dual/radar depletion outranks ordinary low-fuel collection."""
        containers: dict[str, ContainerStateDict] = {
            "101,100": _c(101, 100, 700, True),
            "102,100": _c(102, 100, 0, False),
        }
        world, self_state = _make_world(fuel=450, containers=containers)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(dual_count=3, default_count=30)
        inventory["extra_radars"]["count"] = 4

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["behavior"]["reason"] == "equipment_critical"
        assert decision["command"]["cmd_type"] == "pickup_move"

    def test_critical_equipment_search_uses_radar_when_ready(self) -> None:
        """Critical equipment depletion scans before relocating when radar is ready."""
        world, self_state = _make_world(fuel=800)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(dual_count=0, dual_enabled=False, default_count=30)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["behavior"]["reason"] == "radar_for_equipment"
        assert decision["command"]["cmd_type"] == "radar"

    def test_critical_equipment_search_relocates_when_radar_on_cooldown(self) -> None:
        """Critical equipment depletion relocates to the next sector when radar is cooling down."""
        world, self_state = _make_world(fuel=800)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=2,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=-1,
            combat_target_x=0,
            combat_target_y=0,
            combat_phase="none",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory(dual_count=0, dual_enabled=False, default_count=30)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["behavior"]["reason"] == "search_equipment_local"
        assert decision["command"]["cmd_type"] == "teleport"
        # Local hop from (100,100) direction 2=(-1,0) dist=15 → (85,100)
        assert decision["command"]["target_x"] == 85
        assert decision["command"]["target_y"] == 100
        assert decision["updated_ai_state"]["patrol_waypoint_index"] == 3
        assert decision["updated_ai_state"]["equipment_search_failures"] == 1

    def test_equipment_search_bails_out_after_max_failures(self) -> None:
        """Equipment search stops after equip_search_max_failures consecutive failures."""
        world, self_state = _make_world(fuel=800)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=-1,
            combat_target_x=0,
            combat_target_y=0,
            combat_phase="none",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=3,
        )
        inventory = _make_inventory(dual_count=0, dual_enabled=False, default_count=30)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # After 3 failures (== max), search bails out → falls through to hunt
        assert decision["behavior"]["mode"] == "HUNT"

    def test_equipment_search_skips_when_teleport_unaffordable(self) -> None:
        """Equipment search returns None when fuel can't cover a teleport hop."""
        from tankpit_bot.bot.ai.types import AIConfigDict

        world, self_state = _make_world(fuel=550)
        base_config = make_default_ai_config()
        # Set teleport_fuel_cost high so 550 < 400 + 400 = 800
        config = AIConfigDict(**{**base_config, "teleport_fuel_cost": 400})
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=-1,
            combat_target_x=0,
            combat_target_y=0,
            combat_phase="none",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        # dual=0 triggers critical, radar=0 so no radar scan possible
        inventory = _make_inventory(dual_count=0, dual_enabled=False, default_count=0)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # fuel=550 >= fuel_low(500) so search runs, but can't afford teleport
        assert decision["behavior"]["mode"] != "COLLECT_EQUIPMENT"

    def test_equipment_search_skips_when_fuel_too_low(self) -> None:
        """Equipment search defers when fuel is below fuel_low_threshold."""
        # fuel=450: above critical (400) so critical_fuel doesn't fire,
        # but below low (500) so equipment search defers to fuel recovery.
        world, self_state = _make_world(fuel=450)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 99500,
            }
        )
        # dual=0 triggers emergency equipment, radar=0 so no scan
        inventory = _make_inventory(dual_count=0, dual_enabled=False, default_count=0)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # Equipment search skips because fuel too low → falls through to fuel
        assert decision["behavior"]["mode"] == "COLLECT_FUEL"

    def test_reachable_container_behind_wall_uses_move_waypoint(self) -> None:
        """Terrain detours walk via waypoint instead of pickup-move to final target."""
        containers: dict[str, ContainerStateDict] = {
            "104,100": _c(104, 100, 700, True),
        }
        world, self_state = _make_world(fuel=450, containers=containers)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()
        terrain = FakeTerrainMap({(102, 100): "#"})

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"
        assert decision["command"]["cmd_type"] == "move"
        assert decision["command"]["target_x"] == 103
        assert decision["command"]["target_y"] == 101

    def test_low_fuel_without_targets_uses_radar(self) -> None:
        """Low fuel uses radar when no actionable fuel target exists and cooldown elapsed."""
        world, self_state = _make_world(fuel=300)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"
        assert decision["behavior"]["reason"] == "radar_for_fuel"
        assert decision["command"]["cmd_type"] == "radar"

    def test_low_fuel_blocked_search_with_visible_threats_falls_back_to_map(self) -> None:
        """Visible threats reach the hunt fallback when fuel search cannot execute."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(self_x=100, self_y=100, fuel=300, tanks=tanks)
        ai_state = AIStateDict(**{**_scanned_ai_state(), "last_scan_ms": 99999})
        inventory = _make_inventory()
        # Block viewport edge target (108,108) and all its neighbors
        terrain_data: dict[tuple[int, int], str] = {
            (108, 108): "W",
            (109, 108): "W",
            (107, 108): "W",
            (108, 109): "#",
            (108, 107): "#",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["command"]["cmd_type"] == "map_open"

    def test_locked_combat_with_zero_dual_releases_to_equipment(self) -> None:
        """Dual depletion releases combat lock so equipment recovery starts."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=103,
                y=103,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=103,
            combat_target_y=103,
            combat_phase="engaging",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory(dual_count=0, dual_enabled=False, default_count=30)
        inventory["extra_radars"]["count"] = 30

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"

    def test_killed_target_releases_combat_lock_for_recovery(self) -> None:
        """After killing the locked target, combat lock releases for fuel/equipment."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=103,
                y=103,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
            ),
            "60": TankStateDict(
                tank_id=60,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="Enemy2",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        containers: dict[str, ContainerStateDict] = {
            "101,100": _c(101, 100, 700, True),
        }
        world, self_state = _make_world(
            fuel=400,
            tanks=tanks,
            containers=containers,
        )
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=103,
            combat_target_y=103,
            combat_phase="engaging",
            killed_tank_ids={"50": 99000},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # With target 50 killed, combat lock is released.
        # Fuel=400 < 500 threshold → should collect fuel, not reacquire.
        assert decision["behavior"]["mode"] == "COLLECT_FUEL"


class TestHelpers:
    """Tests for internal helper functions."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_compute_equipment_collect_fuel(self) -> None:
        """_compute_equipment returns [2, 5] for COLLECT_FUEL mode."""
        inventory = _make_inventory()
        result = _compute_equipment(800, inventory)
        assert result == [2, 5]

    def test_compute_equipment_hunt(self) -> None:
        """_compute_equipment returns [2, 5] for HUNT mode."""
        inventory = _make_inventory()
        result = _compute_equipment(800, inventory)
        assert result == [2, 5]

    def test_compute_equipment_no_shields(self) -> None:
        """_compute_equipment never includes shields."""
        inventory = _make_inventory()
        result = _compute_equipment(800, inventory)
        assert 1 not in result

    def test_compute_equipment_dual_depleted(self) -> None:
        """_compute_equipment drops dual when count is 0."""
        inventory = _make_inventory(dual_count=0)
        result = _compute_equipment(800, inventory)
        assert result == [5]

    def test_local_equipment_search_hop_rotates_cardinal(self) -> None:
        """Local search hop rotates through cardinal directions."""
        world, self_state = _make_world(self_x=100, self_y=100, fuel=800)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(dual_count=5, dual_enabled=True, default_count=30)
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        hop_x, hop_y, next_index = _local_equipment_search_hop(ctx)

        # Direction 0 is (1,0), hop_distance=15: x+15, y+0
        assert hop_x == 115
        assert hop_y == 100
        assert next_index == 1

    def test_local_equipment_search_hop_clamps_to_map_bounds(self) -> None:
        """Local search hop clamps to [1, 254] at map edges."""
        world, self_state = _make_world(self_x=250, self_y=250, fuel=800)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(dual_count=5, dual_enabled=True, default_count=30)
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        hop_x, hop_y, next_index = _local_equipment_search_hop(ctx)

        assert hop_x == 254
        assert hop_y == 250
        assert next_index == 1

    def test_expire_kills_removes_expired(self) -> None:
        """_expire_kills removes entries older than cooldown."""
        killed = {"50": 1000, "60": 5000}
        result = _expire_kills(killed, 22000, 20000)
        # 50: 22000-1000=21000 >= 20000 → expired
        # 60: 22000-5000=17000 < 20000 → kept
        assert result == {"60": 5000}

    def test_expire_kills_keeps_recent(self) -> None:
        """_expire_kills keeps entries within cooldown."""
        killed = {"50": 10000, "60": 15000}
        result = _expire_kills(killed, 20000, 20000)
        assert result == {"50": 10000, "60": 15000}

    def test_expire_kills_empty_input(self) -> None:
        """_expire_kills handles empty dict."""
        result = _expire_kills({}, 20000, 20000)
        assert result == {}

    def test_filter_killed_tanks_removes_killed(self) -> None:
        """_filter_killed_tanks removes tanks on kill cooldown."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="Enemy1",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
            "60": TankStateDict(
                tank_id=60,
                x=110,
                y=110,
                team=2,
                rank=1,
                name="Enemy2",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, _ = _make_world(tanks=tanks)
        killed = {"50": 10000}
        filtered = _filter_killed_tanks(world, killed)
        assert "50" not in filtered["tanks"]
        assert "60" in filtered["tanks"]

    def test_filter_killed_tanks_empty_killed(self) -> None:
        """_filter_killed_tanks returns same world when no killed tanks."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="Enemy1",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, _ = _make_world(tanks=tanks)
        filtered = _filter_killed_tanks(world, {})
        assert filtered is world  # No copy needed

    def test_require_command_returns_command(self) -> None:
        """_require_command returns a concrete command unchanged."""
        from tankpit_bot.bot.types import make_move_command

        command = make_move_command(10, 20)

        result = _require_command(command, 10, 20, "fuel")

        assert result == command

    def test_require_command_raises_on_missing_command(self) -> None:
        """_require_command raises when a selected target has no executable action."""
        import pytest

        with pytest.raises(ValueError, match="No executable command for fuel target"):
            _require_command(None, 10, 20, "fuel")

    def test_walk_or_teleport_returns_none_when_no_landing_exists(self) -> None:
        """_walk_or_teleport rejects blocked targets with no legal landing tile."""
        containers: dict[str, ContainerStateDict] = {
            "107,107": _c(107, 107, 0, False),
        }
        world, self_state = _make_world(self_x=100, self_y=100, fuel=300, containers=containers)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(default_count=5)
        terrain_data: dict[tuple[int, int], str] = {
            (107, 107): "W",
            (108, 107): "W",
            (106, 107): "W",
            (107, 108): "#",
            (107, 106): "#",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = _walk_or_teleport(ctx, 107, 107, pickup=False)

        assert result is None

    def test_walk_or_teleport_direct_move_when_pickup_disabled(self) -> None:
        """Open-ground scouting uses move when pickup mode is disabled."""
        world, self_state = _make_world(self_x=100, self_y=100, fuel=300)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(default_count=5)
        terrain = FakeTerrainMap()
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = _walk_or_teleport(ctx, 107, 100, pickup=False)

        if result is None:
            raise AssertionError("Expected direct move command")
        assert result["cmd_type"] == "move"
        assert result["target_x"] == 107
        assert result["target_y"] == 100

    def test_walk_or_teleport_direct_pickup_on_open_ground(self) -> None:
        """Open-ground collection keeps pickup_move when the direct route is clear."""
        world, self_state = _make_world(self_x=100, self_y=100, fuel=300)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(default_count=5)
        terrain = FakeTerrainMap()
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = _walk_or_teleport(ctx, 107, 100, pickup=True)

        if result is None:
            raise AssertionError("Expected direct pickup command")
        assert result["cmd_type"] == "pickup_move"
        assert result["target_x"] == 107
        assert result["target_y"] == 100

    def test_walk_or_teleport_approaches_outer_ring_pickup_before_collecting(self) -> None:
        """Outer-ring radar targets are approached before pickup_move is used."""
        containers: dict[str, ContainerStateDict] = {
            "72,63": _c(72, 63, 0, False),
        }
        world, self_state = _make_world(self_x=64, self_y=64, fuel=300, containers=containers)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(default_count=5)
        terrain = FakeTerrainMap()
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = _walk_or_teleport(ctx, 72, 63, pickup=True)

        if result is None:
            raise AssertionError("Expected approach move command")
        assert result["cmd_type"] == "move"
        assert result["target_x"] == 71
        assert result["target_y"] == 63

    def test_decide_approaches_outer_ring_equipment_before_pickup(self) -> None:
        """Visible outer-ring equipment becomes an approach move, not pickup_move."""
        containers: dict[str, ContainerStateDict] = {
            "72,63": _c(72, 63, 0, False),
        }
        world, self_state = _make_world(self_x=64, self_y=64, fuel=800, containers=containers)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(default_count=30)
        inventory["dual_shots"]["count"] = 0
        inventory["dual_shots"]["enabled"] = False
        terrain = FakeTerrainMap()

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["command"]["cmd_type"] == "move"
        assert decision["command"]["target_x"] == 71
        assert decision["command"]["target_y"] == 63

    def test_normal_equipment_low_uses_standard_equipment_branch(self) -> None:
        """Non-critical equipment depletion still collects equipment before combat."""
        containers: dict[str, ContainerStateDict] = {
            "106,106": _c(106, 106, 30, False),
        }
        world, self_state = _make_world(fuel=800, containers=containers)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(default_count=30)
        inventory["missile_shots"]["count"] = 5
        inventory["dual_shots"]["count"] = 20
        inventory["extra_radars"]["count"] = 20

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["behavior"]["reason"] == "equipment_low"
        assert decision["command"]["cmd_type"] == "pickup_move"

    def test_waypoint_clamped_to_viewport_bounds(self) -> None:
        """A* waypoint outside the viewport is clamped to viewport edge."""
        # Place a wall that forces A* to route around and produce an off-viewport waypoint.
        # Bot at (100,100), target at (91,91) (viewport corner).
        # viewport: left=91, top=91, right=108, bottom=108
        # Wall blocks the direct diagonal, A* detours through (89,95) — outside viewport.
        world, self_state = _make_world(self_x=100, self_y=100, fuel=300)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()
        # Create a wall that forces the A* path outside viewport
        terrain_data: dict[tuple[int, int], str] = {
            (row, col): "#" for row in range(92, 100) for col in range(92, 100)
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = _walk_or_teleport(ctx, 91, 91, pickup=False)

        # Should either clamp the waypoint to viewport or fall through to teleport,
        # but never issue a move to coordinates outside the viewport.
        if result is not None and result["cmd_type"] == "move":
            viewport = ctx.world["viewport"]
            left = viewport["left"]
            top = viewport["top"]
            right = left + viewport["width"] - 1
            bottom = top + viewport["height"] - 1
            assert left <= result["target_x"] <= right
            assert top <= result["target_y"] <= bottom

    def test_waypoint_clamped_to_self_position_rejected(self) -> None:
        """A waypoint that clamps to the bot's own position is rejected."""
        # Bot at (91,100), viewport left=82. Waypoint at (80,100) clamps to left=82.
        # But bot is at x=91, not 82, so this won't hit self. Use bot at viewport edge.
        world, self_state = _make_world(self_x=100, self_y=100, fuel=300)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()
        terrain = FakeTerrainMap()
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        # Waypoint at (100,100) = self position → should be rejected
        result = _waypoint_move_command(ctx, 105, 105, (100, 100))

        assert result is None

    def test_walk_or_teleport_rejects_failed_move_target(self) -> None:
        """_walk_or_teleport returns None for a recently failed move target."""
        world, self_state = _make_world(self_x=100, self_y=100, fuel=300)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()
        terrain = FakeTerrainMap()
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        mark_move_target_failed(107, 100, 90000)
        result = _walk_or_teleport(ctx, 107, 100, pickup=False)

        assert result is None

    def test_walk_or_teleport_rejects_enemy_occupied_direct_move(self) -> None:
        """_walk_or_teleport returns None when direct move target is occupied."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
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
        world, self_state = _make_world(
            self_x=100,
            self_y=100,
            fuel=300,
            tanks=tanks,
        )
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()
        terrain = FakeTerrainMap()
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = _walk_or_teleport(ctx, 107, 100, pickup=False)

        assert result is None

    def test_walk_or_teleport_rejects_enemy_occupied_waypoint(self) -> None:
        """_walk_or_teleport returns None when A* waypoint is occupied by enemy."""
        from tankpit_bot.bot.ai.pathfinding import find_path_segment_target

        # Set up terrain that blocks the direct path at x=101, forcing A*
        # to route through (100,99)->(101,99)->(102,99)->(102,100).
        # Block the entire row y=100 from x=101 to x=104.
        terrain_data: dict[tuple[int, int], str] = {
            (101, 100): "#",
            (102, 100): "#",
            (103, 100): "#",
            (104, 100): "#",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)

        # Verify what waypoint A* will produce so we can place enemy there
        waypoint = find_path_segment_target(terrain, 100, 100, 107, 100)
        if waypoint is None:
            raise AssertionError("expected A* waypoint for occupied-target test")
        wx, wy = waypoint

        # Place enemy at the A* waypoint
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=wx,
                y=wy,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = _make_world(
            self_x=100,
            self_y=100,
            fuel=300,
            tanks=tanks,
        )
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = _walk_or_teleport(ctx, 107, 100, pickup=False)

        assert result is None

    def test_walk_or_teleport_rejects_occupied_move_without_terrain(self) -> None:
        """_walk_or_teleport rejects occupied target even without terrain map."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
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
        world, self_state = _make_world(
            self_x=100,
            self_y=100,
            fuel=300,
            tanks=tanks,
        )
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        result = _walk_or_teleport(ctx, 107, 100, pickup=False)

        assert result is None

    def test_is_occupied_by_enemy_returns_true_for_enemy_tile(self) -> None:
        """_is_occupied_by_enemy detects a tank on the given tile."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=107,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        assert _is_occupied_by_enemy(ctx, 107, 100) is True
        assert _is_occupied_by_enemy(ctx, 108, 100) is False


class TestDecideCombatFeedback:
    """Tests for combat feedback handling in decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_hit_feedback_after_kill_sees_no_enemy(self) -> None:
        """After a kill, Deactivation sets victim to (0,0).

        The protocol handles kills: Deactivation message sets victim
        position to (0,0). The AI receives "hit" feedback (CombatHit
        arrives for kills too). With the tank at (0,0), threat analysis
        skips it, and map_open fires to find new enemies.
        """
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=0,
                y=0,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        ai_state = _scanned_ai_state()
        ai_state_with_shot = AIStateDict(
            **{
                **ai_state,
                "last_shot_target_id": 50,
                "last_shot_target_name": "Enemy",
            }
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state_with_shot,
            inventory,
            100000,
            None,
            "hit",
        )

        # Tank at (0,0) is skipped by threat analysis → no enemies → map open fires
        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "find_enemies"
        assert decision["updated_ai_state"]["last_shot_target_id"] == -1

    def test_miss_with_no_target_in_world_opens_map(self) -> None:
        """Miss with target gone from world falls through to find enemies."""
        world, self_state = _make_world(fuel=800)
        ai_state = _scanned_ai_state()
        ai_state_with_shot = AIStateDict(
            **{
                **ai_state,
                "last_shot_target_id": 50,
                "last_shot_target_name": "Enemy",
            }
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state_with_shot,
            inventory,
            100000,
            None,
            "miss",
        )

        # No target in world → falls through to find_enemies
        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "find_enemies"

    def test_hit_feedback_continues_normally(self) -> None:
        """Hit feedback falls back to the normal combat phase flow."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        ai_state = _scanned_ai_state()
        ai_state_with_shot = AIStateDict(
            **{
                **ai_state,
                "last_shot_target_id": 50,
                "last_shot_target_name": "Enemy",
                "last_map_open_ms": 99000,
            }
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state_with_shot,
            inventory,
            100000,
            None,
            "hit",
        )

        # With no locked combat phase, normal flow starts by reopening map.
        assert decision["behavior"]["reason"] != "kill_confirmed"
        assert decision["behavior"]["reason"] != "miss_relocate"
        assert decision["command"]["cmd_type"] == "map_open"

    def test_no_feedback_when_no_shot_pending(self) -> None:
        """Empty feedback when no shot was fired last tick."""
        world, self_state = _make_world(fuel=800)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            None,
            "",
        )

        # No shot pending → normal flow, map open for no enemies
        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "find_enemies"


class TestDecideMapOpen:
    """Tests for map open tactical override."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_map_open_when_no_enemies(self) -> None:
        """decide() triggers map open when no live enemies visible."""
        world, self_state = _make_world(fuel=800)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "find_enemies"

    def test_no_map_open_when_enemy_visible(self) -> None:
        """decide() skips map open when a live enemy is visible."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        ai_state = _scanned_ai_state()
        ai_state_recent_map = AIStateDict(
            **{
                **ai_state,
                "last_map_open_ms": 99000,
            }
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state_recent_map,
            inventory,
            100000,
            None,
        )

        assert decision["behavior"]["reason"] != "find_enemies"

    def test_fallback_uses_radar_when_map_on_cooldown(self) -> None:
        """Fallback uses radar instead of map_open when cooldown active."""
        world, self_state = _make_world(fuel=800)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=1,
            last_shoot_ms=0,
            last_map_open_ms=99000,  # recent map open
            combat_target_id=-1,
            combat_target_x=0,
            combat_target_y=0,
            combat_phase="none",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "radar"
        assert decision["behavior"]["reason"] == "radar_for_enemies"

    def test_fallback_walks_when_map_and_radar_on_cooldown(self) -> None:
        """Fallback walks to edge when both map and radar are on cooldown."""
        world, self_state = _make_world(fuel=800)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99000,  # recent radar
            last_shoot_ms=0,
            last_map_open_ms=99000,  # recent map open
            combat_target_id=-1,
            combat_target_x=0,
            combat_target_y=0,
            combat_phase="none",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "move"
        assert decision["behavior"]["reason"] == "edge_for_enemies"

    def test_fallback_opens_map_when_edge_walk_blocked(self) -> None:
        """Fallback opens map when map+radar on cooldown and edge blocked."""
        world, self_state = _make_world(fuel=800)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99000,
            last_shoot_ms=0,
            last_map_open_ms=99000,
            combat_target_id=-1,
            combat_target_x=0,
            combat_target_y=0,
            combat_phase="none",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()
        # Block edge target (108,108) and its neighbors so walk fails.
        # _viewport_edge_target picks the viewport right/bottom edge (108,108)
        # because the player at (100,100) is in the lower-left half of the map.
        terrain = FakeTerrainMap(
            terrain_data={
                (108, 108): "#",
                (107, 108): "#",
                (109, 108): "#",
                (108, 107): "#",
                (108, 109): "#",
            }
        )

        decision = decide(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            terrain,
        )

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "find_enemies"


class TestDecideShotTracking:
    """Tests for shot target tracking in decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_shoot_command_records_target(self) -> None:
        """Shoot command records target_id and target_name for feedback."""
        # Enemy within viewport range (dist=6 <= _MAX_SHOOT_RANGE=8)
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=103,
                y=103,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=103,
            combat_target_y=103,
            combat_phase="none",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        if decision["command"]["cmd_type"] == "shoot":
            assert decision["updated_ai_state"]["last_shot_target_id"] == 50
            assert decision["updated_ai_state"]["last_shot_target_name"] == "Enemy"


class TestDecideKillCooldown:
    """Tests for kill cooldown filtering in decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_killed_tanks_filtered_from_world(self) -> None:
        """Killed tanks are not visible to AI threat selection."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="KilledEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        ai_state = _scanned_ai_state()
        ai_state_with_kill = AIStateDict(
            **{
                **ai_state,
                "killed_tank_ids": {"50": 90000},
            }
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state_with_kill,
            inventory,
            100000,
            None,
        )

        # Tank 50 should be filtered — AI should not be targeting it
        # The behavior should not be HUNT targeting tank 50
        if decision["behavior"]["mode"] == "HUNT":
            bx = decision["behavior"]["target_x"]
            by = decision["behavior"]["target_y"]
            assert bx != 105 or by != 105

    def test_miss_reopens_map_when_target_far(self) -> None:
        """A miss during engaging reopens map when target is far away."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=110,
                y=110,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=110,
            combat_target_y=110,
            combat_phase="engaging",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=50,
            last_shot_target_name="Enemy",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            None,
            "miss",
        )

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["updated_ai_state"]["combat_target_id"] == 50
        assert decision["updated_ai_state"]["combat_phase"] == "closing"

    def test_miss_always_reacquires_even_when_close(self) -> None:
        """A miss always triggers reacquire, even for adjacent targets."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=101,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=101,
            combat_target_y=100,
            combat_phase="engaging",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=50,
            last_shot_target_name="Enemy",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            None,
            "miss",
        )

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["updated_ai_state"]["combat_phase"] == "closing"

    def test_expired_kills_removed(self) -> None:
        """Expired kill cooldowns are cleaned up."""
        world, self_state = _make_world(fuel=800)
        ai_state = _scanned_ai_state()
        # Kill at time 50000, now at 100000 → 50000ms > 20000ms cooldown → expired
        ai_state_with_old_kill = AIStateDict(
            **{
                **ai_state,
                "killed_tank_ids": {"50": 50000},
            }
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state_with_old_kill,
            inventory,
            100000,
            None,
        )

        # Expired kill should be removed from AI state
        assert "50" not in decision["updated_ai_state"]["killed_tank_ids"]


class TestDecideTeleportToFarTarget:
    """Tests for teleport-to-far-target override."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_far_target_starts_with_map_open(self) -> None:
        """decide() starts phase-based combat with map_open for a new far target."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="FarEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=-1,
            combat_target_x=0,
            combat_target_y=0,
            combat_phase="none",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["updated_ai_state"]["combat_target_id"] == 50
        assert decision["updated_ai_state"]["combat_phase"] == "closing"

    def test_locked_phase_one_target_teleports_to_existing_enemy(self) -> None:
        """A locked phase-1 combat target teleports to an adjacent landing tile."""
        tanks: dict[str, TankStateDict] = {
            "60": TankStateDict(
                tank_id=60,
                x=101,
                y=100,
                team=2,
                rank=1,
                name="CloserEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
            "50": TankStateDict(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="LockedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=120,
            combat_target_y=100,
            combat_phase="closing",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["behavior"]["target_x"] == 119
        assert decision["behavior"]["target_y"] == 100
        assert decision["updated_ai_state"]["combat_target_id"] == 50
        assert decision["updated_ai_state"]["combat_phase"] == "engaging"

    def test_locked_phase_one_target_uses_passable_adjacent_combat_landing(self) -> None:
        """Combat phase 1 picks a passable adjacent landing tile near the enemy."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=197,
                y=86,
                team=2,
                rank=1,
                name="LockedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(self_x=180, self_y=80, fuel=800, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=197,
            combat_target_y=86,
            combat_phase="closing",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()
        terrain_data: dict[tuple[int, int], str] = {
            (197, 86): "W",
            (198, 86): "W",
            (197, 87): "W",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 196
        assert decision["command"]["target_y"] == 86

    def test_locked_phase_one_target_without_landing_tile_resets_target(self) -> None:
        """Combat phase 1 clears the target when no adjacent landing tile exists."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=197,
                y=86,
                team=2,
                rank=1,
                name="LockedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(self_x=180, self_y=80, fuel=800, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=197,
            combat_target_y=86,
            combat_phase="closing",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()
        terrain_data: dict[tuple[int, int], str] = {
            (198, 86): "W",
            (196, 86): "W",
            (197, 87): "#",
            (197, 85): "#",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["updated_ai_state"]["combat_target_id"] == -1

    def test_combat_landing_tile_without_terrain_skips_out_of_bounds_candidates(self) -> None:
        """Combat fallback landing only considers in-bounds adjacent candidates."""
        world, self_state = _make_world(self_x=10, self_y=10, fuel=800)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()
        ctx = _DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
        target = EnemyThreatDict(
            tank_id=50,
            x=0,
            y=0,
            distance=20,
            damage_state=0,
            rank=1,
            team=2,
            name="EdgeEnemy",
            is_bot=False,
            timestamp_ms=0,
        )

        landing_x, landing_y = _combat_landing_tile(ctx, target)

        assert (landing_x, landing_y) == (1, 0)

    def test_missing_locked_target_reacquires_new_enemy(self) -> None:
        """When the locked target is gone, decide() reacquires from current threats."""
        tanks: dict[str, TankStateDict] = {
            "60": TankStateDict(
                tank_id=60,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="NewEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=110,
            combat_target_y=100,
            combat_phase="closing",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["updated_ai_state"]["combat_target_id"] == 60

    def test_no_teleport_when_fuel_too_low(self) -> None:
        """decide() skips teleport when fuel can't cover cost + critical."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="FarEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        # fuel=50, hunt_min_fuel=100 → 50 <= 100 → teleport blocked
        world, self_state = _make_world(fuel=50, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=-1,
            combat_target_x=0,
            combat_target_y=0,
            combat_phase="none",
            killed_tank_ids={},
            blocked_combat_targets={},
            last_shot_target_id=-1,
            last_shot_target_name="",
            equipment_search_failures=0,
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # Teleport blocked by fuel guard — falls through to move
        assert decision["command"]["cmd_type"] != "teleport"


class TestDecideBlockedCombatTargets:
    """Tests for blocked combat target memory."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_blocked_target_is_skipped_on_reacquire(self) -> None:
        """A target in blocked_combat_targets is not reacquired as a new target."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "blocked_combat_targets": {"50": 99000},
            }
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # Only threat is blocked → combat returns None → fallback
        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["behavior"]["reason"] == "find_enemies"

    def test_no_landing_tile_blocks_target_and_switches(self) -> None:
        """When landing fails, the target is blocked and the next viable threat is engaged."""
        tanks: dict[str, TankStateDict] = {
            # Enemy boxed in by water (all 4 adjacent tiles blocked)
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=100,
                team=2,
                rank=1,
                name="Boxed",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
            # Second enemy that is reachable
            "60": TankStateDict(
                tank_id=60,
                x=103,
                y=100,
                team=2,
                rank=1,
                name="Reachable",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        # Already locked on boxed target, phase=closing
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "combat_target_id": 50,
                "combat_target_x": 105,
                "combat_target_y": 100,
                "combat_phase": "closing",
            }
        )
        inventory = _make_inventory()
        # Block all 4 adjacent tiles around (105,100)
        terrain = FakeTerrainMap(
            terrain_data={
                (106, 100): "W",
                (104, 100): "W",
                (105, 101): "W",
                (105, 99): "W",
            }
        )

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        # Should block target 50 and switch to target 60
        assert decision["command"]["cmd_type"] == "map_open"
        assert "Reachable" in decision["behavior"]["reason"]
        # Target 50 should be in blocked_combat_targets
        blocked = decision["updated_ai_state"]["blocked_combat_targets"]
        assert "50" in blocked

    def test_no_landing_tile_blocks_target_with_no_alternatives(self) -> None:
        """When landing fails and no other threats exist, falls back to generic search."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=100,
                team=2,
                rank=1,
                name="Boxed",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "combat_target_id": 50,
                "combat_target_x": 105,
                "combat_target_y": 100,
                "combat_phase": "closing",
            }
        )
        inventory = _make_inventory()
        terrain = FakeTerrainMap(
            terrain_data={
                (106, 100): "W",
                (104, 100): "W",
                (105, 101): "W",
                (105, 99): "W",
            }
        )

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        # No viable alternatives → generic fallback
        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "find_enemies"
        blocked = decision["updated_ai_state"]["blocked_combat_targets"]
        assert "50" in blocked

    def test_blocked_target_expires_after_ttl(self) -> None:
        """Blocked combat targets expire after kill_cooldown_ms TTL."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=103,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        config = make_default_ai_config()
        # Blocked at 50000, now=100000, kill_cooldown=30000 → expired (50000 ago > 30000)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "config": config,
                "blocked_combat_targets": {"50": 50000},
            }
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # Expired → target is viable again → combat engages
        assert decision["behavior"]["mode"] == "HUNT"
        assert "Enemy" in decision["behavior"]["reason"]


class TestDecideBlockedEdgeSearch:
    """Tests for blocked viewport-edge scouting paths."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_low_fuel_blocked_edge_search_falls_through(self) -> None:
        """Blocked edge scouting with no landing tile falls through to hunt fallback."""
        world, self_state = _make_world(self_x=100, self_y=100, fuel=300)
        ai_state = AIStateDict(**{**_scanned_ai_state(), "last_scan_ms": 99999})
        inventory = _make_inventory()
        # Block viewport edge target (108,108) and all its neighbors
        terrain_data: dict[tuple[int, int], str] = {
            (108, 108): "W",
            (109, 108): "W",
            (107, 108): "W",
            (108, 109): "#",
            (108, 107): "#",
        }
        terrain = FakeTerrainMap(terrain_data=terrain_data)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["command"]["cmd_type"] == "map_open"
