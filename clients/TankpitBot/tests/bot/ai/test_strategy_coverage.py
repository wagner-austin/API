"""Tests for ai_strategy coverage gaps — locked targets, radar search, exploration."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import (
    AIConfigDict,
    AIStateDict,
    make_default_ai_config,
    make_initial_ai_state,
)
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.sniffer.world_state import reset_world_state, update_world_state_from_position
from tankpit_bot.state.types import ContainerStateDict, SelfStateDict, TankStateDict, WorldStateDict


def _make_world(
    self_x: int = 100,
    self_y: int = 100,
    fuel: int = 800,
    containers: dict[str, ContainerStateDict] | None = None,
    tanks: dict[str, TankStateDict] | None = None,
    scanned: bool = True,
) -> tuple[WorldStateDict, SelfStateDict]:
    """Build a world state for testing."""
    self_state = SelfStateDict(
        tank_id=1,
        x=self_x,
        y=self_y,
        team=1,
        rank=0,
        fuel=fuel,
        leaderboard_position=0,
    )
    scanned_viewports: dict[str, int] = {}
    if scanned:
        vp_left = self_x - 8
        vp_top = self_y - 8
        scanned_viewports[f"{vp_left},{vp_top}"] = 100000
    return (
        WorldStateDict(
            self_state=self_state,
            tanks=tanks or {},
            containers=containers or {},
            mines={},
            terrain={},
            viewport={"left": self_x - 8, "top": self_y - 8, "width": 16, "height": 16},
            scanned_viewports=scanned_viewports,
            map_fuel_dots={},
            timestamp_ms=100000,
        ),
        self_state,
    )


def _c(x: int, y: int, volume: int, is_fuel: bool) -> ContainerStateDict:
    """Create a container state."""
    from tankpit_bot.state.types import make_container_state

    return make_container_state(
        x=x,
        y=y,
        is_fuel=is_fuel,
        volume=volume,
        timestamp_ms=100000,
        failed_pickups=0,
    )


def _make_inventory(
    dual_count: int = 30,
    default_count: int = 30,
    radar_count: int = 30,
) -> InventoryState:
    """Build an inventory."""
    item = InventoryItem(count=default_count, enabled=True)
    return InventoryState(
        armor_shields=item,
        dual_shots=InventoryItem(count=dual_count, enabled=True),
        missile_shots=item,
        homing_shots=item,
        extra_radars=InventoryItem(count=radar_count, enabled=True),
    )


def _scanned_ai_state() -> AIStateDict:
    """Build a scanned AI state."""
    return make_initial_ai_state()


class TestLockedEquipmentTarget:
    """Tests for locked equipment target continuation."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_continues_locked_equipment_target(self) -> None:
        """Locked equipment target is continued when still actionable."""
        containers = {"105,105": _c(105, 105, 0, False)}
        world, self_state = _make_world(containers=containers)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "RECOVER_EQUIPMENT",
                "mode_state": "APPROACH",
                "mode_started_ms": 90000,
                "resource_target_kind": "equipment",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        # default_count=15: below low (20) but above break (12) → _try_collect_equipment path
        inventory = _make_inventory(default_count=15)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["behavior"]["reason"] == "equipment_locked"

    def test_locked_equipment_target_clears_when_unexecutable(self) -> None:
        """Locked equipment target is cleared when walk_or_teleport fails."""
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        # Target on water with all adjacent tiles also water — no landing
        containers = {"105,105": _c(105, 105, 0, False)}
        world, self_state = _make_world(containers=containers, fuel=800)
        terrain_data: dict[tuple[int, int], str] = {
            (105, 105): "W",
            (104, 105): "W",
            (106, 105): "W",
            (105, 104): "W",
            (105, 106): "W",
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "RECOVER_EQUIPMENT",
                "mode_state": "APPROACH",
                "mode_started_ms": 90000,
                "resource_target_kind": "equipment",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        # default_count=15: below low but above break
        inventory = _make_inventory(default_count=15)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        # Should NOT keep equipment_locked — target is on water with no landing
        assert decision["behavior"]["reason"] != "equipment_locked"


class TestLockedFuelTarget:
    """Tests for locked fuel target continuation."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_continues_locked_fuel_target(self) -> None:
        """Locked fuel target is continued when still actionable."""
        containers = {"105,105": _c(105, 105, 700, True)}
        world, self_state = _make_world(fuel=400, containers=containers)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "resource_target_kind": "fuel",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"
        assert "fuel=700" in decision["behavior"]["reason"]


class TestRadarForEquipment:
    """Tests for radar-for-equipment in unscanned viewport."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_radar_for_equipment_in_unscanned_viewport(self) -> None:
        """Equipment recovery radars in unscanned viewport when stock available."""
        world, self_state = _make_world(fuel=800, scanned=False)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "RECOVER_EQUIPMENT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
        # default_count=15: below low (20) but above break (12) → _try_collect_equipment
        # radar_count=15: above break (12) so _try_search_critical doesn't fire
        inventory = _make_inventory(default_count=15, radar_count=15)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["behavior"]["reason"] == "radar_for_equipment"
        assert decision["command"]["cmd_type"] == "radar"


class TestExplorationSkipsTeleportLowFuel:
    """Tests for exploration rejecting teleport when fuel too low."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_exploration_skips_teleport_when_cant_afford(self) -> None:
        """Exploration skips teleport candidates when fuel reserve too low."""
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.bot.ai.movement import select_exploration_command
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        # All edge tiles are water — only teleport is possible, but fuel is too low
        terrain_data: dict[tuple[int, int], str] = {}
        for x in range(92, 108):
            for y in range(92, 108):
                terrain_data[(x, y)] = "W"
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)

        world, self_state = _make_world(fuel=140)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "RECOVER_EQUIPMENT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
        inventory = _make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = select_exploration_command(ctx)

        # All candidates blocked or teleport-unaffordable → None
        assert result is None


class TestEquipmentSearchHopFallback:
    """Tests for equipment search hop when no target and no radar."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_equipment_search_hop_when_viewport_scanned_no_radar(self) -> None:
        """Equipment search hops to fresh sector when viewport scanned and no radar."""
        world, self_state = _make_world(fuel=800, scanned=True)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "RECOVER_EQUIPMENT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
        # default_count=15: below low but above break; radar=0 so no scan
        # radar_count=13: above break (12) so critical path doesn't fire; radar stock
        # doesn't matter for this test since viewport is already scanned
        inventory = _make_inventory(default_count=15, radar_count=13)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["behavior"]["reason"] == "search_equipment_local"
        assert decision["command"]["cmd_type"] == "teleport"

    def test_equipment_search_walks_edge_when_teleport_unaffordable(self) -> None:
        """Durable equipment recovery edge-walks when the search hop is unaffordable.

        Regression guard for live run 20260610-000x: the owner used to
        raise here, killing the bot process mid-game.
        """
        # fuel=550: above critical (500) so fuel recovery doesn't fire first
        world, self_state = _make_world(fuel=550, scanned=True)
        base_config = make_default_ai_config()
        # hop_distance=90: floor(6 * 90)=540, so 550 < 540+100 reserve.
        config = AIConfigDict(**{**base_config, "equip_search_hop_distance": 90})
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "config": config,
                "mode": "RECOVER_EQUIPMENT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
        # default_count=15, radar_count=13: above break, viewport scanned → search hop path
        inventory = _make_inventory(default_count=15, radar_count=13)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["behavior"]["reason"] == "edge_for_equipment"
        assert decision["command"]["cmd_type"] == "move"


class TestCriticalEquipmentLockedTarget:
    """Tests for critical equipment locked target paths."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_continues_locked_critical_equipment_target(self) -> None:
        """Critical locked equipment target is continued when executable."""
        containers = {"105,105": _c(105, 105, 0, False)}
        world, self_state = _make_world(containers=containers)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "resource_target_kind": "equipment",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        # default_count=5: below break (12) → critical path
        inventory = _make_inventory(default_count=5)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["behavior"]["reason"] == "equipment_locked"

    def test_clears_locked_critical_equipment_when_unexecutable(self) -> None:
        """Critical locked equipment target is cleared when not executable."""
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        containers = {"105,105": _c(105, 105, 0, False)}
        world, self_state = _make_world(containers=containers)
        terrain_data: dict[tuple[int, int], str] = {
            (105, 105): "W",
            (104, 105): "W",
            (106, 105): "W",
            (105, 104): "W",
            (105, 106): "W",
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "resource_target_kind": "equipment",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        # default_count=5: below break → critical path
        inventory = _make_inventory(default_count=5)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["reason"] != "equipment_locked"


class TestFuelSearchFallbacks:
    """Tests for fuel search fallback paths."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_locked_fuel_clears_when_unexecutable(self) -> None:
        """Locked fuel target is cleared when walk_or_teleport fails."""
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        containers = {"105,105": _c(105, 105, 700, True)}
        world, self_state = _make_world(fuel=400, containers=containers)
        terrain_data: dict[tuple[int, int], str] = {
            (105, 105): "W",
            (104, 105): "W",
            (106, 105): "W",
            (105, 104): "W",
            (105, 106): "W",
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "resource_target_kind": "fuel",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        # Fuel target was blocked — should clear and search
        assert "fuel=700" not in decision["behavior"]["reason"]

    def test_fuel_search_hop_when_scanned_no_visible_fuel(self) -> None:
        """Fuel search hops to fresh sector when scanned but no fuel found."""
        world, self_state = _make_world(fuel=400, scanned=True)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"
        assert decision["behavior"]["reason"] == "search_fuel_local"

    def test_fuel_edge_walk_when_teleport_unaffordable(self) -> None:
        """Fuel recovery walks to viewport edge when teleport too expensive."""
        world, self_state = _make_world(fuel=140, scanned=True)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"
        assert decision["behavior"]["reason"] == "edge_for_fuel"

    def test_fuel_recovery_opens_map_when_all_paths_are_blocked(self) -> None:
        """Durable fuel recovery opens the map for intel instead of crashing.

        With every viewport tile water, radar exhausted, and no
        affordable hop, the owner's terminal fallback is the free
        map-intel action -- raising here used to kill the bot process.
        """
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        terrain_data: dict[tuple[int, int], str] = {}
        for x in range(92, 108):
            for y in range(92, 108):
                terrain_data[(x, y)] = "W"
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        world, self_state = _make_world(fuel=140, scanned=True)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"
        assert decision["behavior"]["reason"] == "map_intel_for_fuel"
        assert decision["command"]["cmd_type"] == "map_open"
