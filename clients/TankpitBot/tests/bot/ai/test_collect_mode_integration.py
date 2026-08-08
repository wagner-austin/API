"""Integration tests for COLLECT-mode equipment priority.

``test_collect_mode_integration.py`` was 779 lines; the search-recovery
suite is now a sibling.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.types import (
    AIStateDict,
)
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import (
    ContainerStateDict,
)
from tests.bot.ai._collect_integration_fixtures import _enemy
from tests.bot.ai._support import (
    make_container,
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestRecoverEquipmentPriority:
    """Tests for top-level recovery priority ordering."""

    def test_equipment_before_combat(self) -> None:
        """Critical equipment recovery preempts combat when supplies are depleted."""
        ws = WorldService()
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

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["command"]["cmd_type"] == "pickup_equipment"

    def test_no_equipment_when_none_visible(self) -> None:
        """Critical equipment depletion searches when no actionable target is visible."""
        ws = WorldService()
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

        decision = decide(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            None,
            map_fuel_dots=((140, 100),),
            ws=ws,
        )

        assert decision["behavior"]["reason_kind"] == "search_collect_local"
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 140
        assert decision["command"]["target_y"] == 100

    def test_equipment_at_break_threshold_relocates_in_scanned_viewport(self) -> None:
        """Break-threshold equipment depletion still enters recovery."""
        ws = WorldService()
        world, self_state = make_world(fuel=800, tanks={"50": _enemy(x=120, y=100)})
        inventory = make_inventory(dual_count=3, dual_enabled=True, default_count=30)
        inventory["extra_radars"]["count"] = 30
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
            }
        )

        decision = decide(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            None,
            map_fuel_dots=((140, 100),),
            ws=ws,
        )

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["command"]["cmd_type"] == "teleport"

    def test_critical_homing_shots_interrupts_for_equipment(self) -> None:
        """Critical homing depletion uses the same recovery path as dual/radar."""
        ws = WorldService()
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

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            inventory,
            100000,
            None,
            ws=ws,
        )

        assert decision["behavior"]["reason_kind"] == "equipment_restock"
        assert decision["command"]["cmd_type"] == "pickup_equipment"

    def test_homing_at_break_threshold_triggers_equipment_recovery(self) -> None:
        """Break-threshold homing count still enters equipment recovery."""
        ws = WorldService()
        world, self_state = make_world(fuel=800, tanks={"50": _enemy(x=120, y=100)})
        inventory = make_inventory(default_count=30)
        inventory["homing_shots"]["count"] = 3

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            inventory,
            100000,
            None,
            ws=ws,
        )

        assert decision["behavior"]["mode"] == "COLLECT"

    def test_active_combat_interrupted_by_critical_equipment(self) -> None:
        """Locked combat yields to critical equipment recovery."""
        ws = WorldService()
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

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["reason_kind"] == "equipment_restock"
        assert decision["command"]["cmd_type"] == "pickup_equipment"

    def test_active_combat_with_subcritical_fuel_interrupts_for_collection(self) -> None:
        """Fuel recovery outranks combat once fuel drops below the threshold.

        The held combat lock (``combat_target_id``) is preserved across
        the recovery cycle so HUNT can resume the same engagement after
        refueling rather than re-acquiring fresh. The recovery branch
        still owns this tick -- the bot dispatches ``pickup_fuel``, not
        ``shoot`` -- but the lock survives for the post-recovery
        cascade.
        """
        ws = WorldService()
        containers: dict[str, ContainerStateDict] = {
            "104,104": make_container(104, 104, 900, True),
        }
        world, self_state = make_world(fuel=150, containers=containers, tanks={"50": _enemy()})
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

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["command"]["cmd_type"] == "pickup_fuel"
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_blocked_equipment_uses_final_pickup_command(self) -> None:
        """Blocked equipment in view still uses the final pickup target."""
        ws = WorldService()
        containers: dict[str, ContainerStateDict] = {
            "128,126": make_container(128, 126, 0, False),
        }
        world, self_state = make_world(self_x=129, self_y=125, fuel=800, containers=containers)
        ai_state = make_scanned_ai_state(landing_scan_viewport="121,117")
        inventory = make_inventory(default_count=3)
        terrain = InMemoryTerrainMap(
            terrain_data={
                (128, 126): "W",
                (127, 126): "W",
                (128, 127): "W",
            }
        )

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain, ws=ws)

        assert decision["command"]["cmd_type"] == "pickup_equipment"
        assert decision["command"]["target_x"] == 128
        assert decision["command"]["target_y"] == 126

    def test_water_locked_equipment_is_skipped(self) -> None:
        """Water-locked equipment is skipped; bot searches elsewhere."""
        ws = WorldService()
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

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            inventory,
            100000,
            terrain,
            ws=ws,
        )

        assert decision["behavior"]["reason_kind"] != "equipment_restock"

    def test_adjacent_blocked_equipment_uses_pickup_move(self) -> None:
        """Adjacent blocked equipment still produces a pickup command."""
        ws = WorldService()
        containers: dict[str, ContainerStateDict] = {
            "128,126": make_container(128, 126, 0, False),
        }
        world, self_state = make_world(self_x=129, self_y=126, fuel=800, containers=containers)
        inventory = make_inventory(default_count=5)
        terrain = InMemoryTerrainMap(terrain_data={(128, 126): "W"})

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(landing_scan_viewport="121,118"),
            inventory,
            100000,
            terrain,
            ws=ws,
        )

        assert decision["command"]["cmd_type"] == "pickup_equipment"

    def test_locked_combat_with_critical_fuel_collects_fuel(self) -> None:
        """Critical fuel interrupts a locked combat phase."""
        ws = WorldService()
        containers: dict[str, ContainerStateDict] = {
            "101,100": make_container(101, 100, 700, True),
        }
        world, self_state = make_world(fuel=150, containers=containers, tanks={"50": _enemy()})
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

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["command"]["cmd_type"] == "pickup_fuel"

    def test_equipment_ranks_ahead_of_fuel_even_at_critical_fuel(self) -> None:
        """COLLECT picks visible equipment before fuel, regardless of fuel level.

        The user's gameplay loop is "pick up all equipment, then maybe the
        biggest fuel container, then hop". The unified cascade enforces
        that ordering at every fuel level. Walking an extra tile or two
        for equipment costs 1 fuel/tile -- a rounding error against the
        viewport-fuel a few ticks later.
        """
        ws = WorldService()
        containers: dict[str, ContainerStateDict] = {
            "101,100": make_container(101, 100, 700, True),
            "102,100": make_container(102, 100, 0, False),
        }
        world, self_state = make_world(fuel=150, containers=containers)
        inventory = make_inventory(dual_count=3, default_count=30)
        inventory["extra_radars"]["count"] = 4

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            inventory,
            100000,
            None,
            ws=ws,
        )

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "equipment_restock"
        assert decision["command"]["cmd_type"] == "pickup_equipment"
