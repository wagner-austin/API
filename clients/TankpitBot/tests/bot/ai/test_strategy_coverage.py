"""Tests for ai_strategy coverage gaps — locked targets, radar search, exploration."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import (
    AIConfigDict,
    AIStateDict,
    make_default_ai_config,
    make_initial_ai_state,
)
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.bot.session_exit import SessionExitError
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
    vp_left = self_x - 8
    vp_top = self_y - 8
    scanned_tiles: dict[str, int] = (
        {
            f"{x},{y}": 100000
            for y in range(vp_top, vp_top + 16)
            for x in range(vp_left, vp_left + 16)
        }
        if scanned
        else {}
    )
    return (
        WorldStateDict(
            self_state=self_state,
            tanks=tanks or {},
            containers=containers or {},
            mines={},
            terrain={},
            viewport={"left": vp_left, "top": vp_top, "width": 16, "height": 16},
            scanned_tiles=scanned_tiles,
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
                "mode": "COLLECT",
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

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason"] == "equipment_locked"

    def test_locked_equipment_target_on_water_releases_lock(self) -> None:
        """A water-locked equipment target is released by the lock-continuation.

        User contract (2026-06-26): ``walk_or_teleport`` returns
        ``None`` for non-walk-reachable pickup targets. The lock
        continuation path treats that as "no longer executable"
        and releases the lock, dropping the bot through to forage.
        """
        from tests.in_memory_terrain_map import InMemoryTerrainMap

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
                "mode": "COLLECT",
                "mode_state": "APPROACH",
                "mode_started_ms": 90000,
                "resource_target_kind": "equipment",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        inventory = _make_inventory(default_count=15)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["reason"] != "equipment_locked"
        assert decision["command"]["cmd_type"] != "pickup_equipment"

    def test_locked_equipment_target_clears_when_teleport_unaffordable(self) -> None:
        """Locked equipment target is cleared when teleport is unaffordable."""
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        # Far target outside viewport — all viewport tiles water so no walkable approach,
        # and the direct teleport is unaffordable.
        containers = {"200,200": _c(200, 200, 0, False)}
        world, self_state = _make_world(containers=containers, fuel=550)
        terrain_data: dict[tuple[int, int], str] = {
            (x, y): "W" for x in range(92, 108) for y in range(92, 108)
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "APPROACH",
                "mode_started_ms": 90000,
                "resource_target_kind": "equipment",
                "resource_target_x": 200,
                "resource_target_y": 200,
            },
        )
        # default_count=15: below low but above break
        inventory = _make_inventory(default_count=15)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        # Locked target cleared — teleport to (200,200) costs 1200 fuel, only have 550
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
        world, self_state = _make_world(fuel=150, containers=containers)
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

        assert decision["behavior"]["mode"] == "COLLECT"
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

    def test_forage_radar_in_unscanned_viewport(self) -> None:
        """Equipment recovery forages with a radar in an unscanned viewport.

        The forager owns the scan path regardless of extras count: it
        fires the radar when any viewport tile is unscanned and the
        radar fuel cost is payable.
        """
        world, self_state = _make_world(fuel=800, scanned=False)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
        # default_count=15: below low (20) but above break (12) → _try_collect_equipment
        # radar_count=15: above break (12) so _try_search_critical doesn't fire
        inventory = _make_inventory(default_count=15, radar_count=15)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason"] == "scan_on_landing"
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

        # Fuel below the short-hop cost so even cheap exploration teleports
        # are unaffordable (reserve gate dropped 2026-06-24; the new floor
        # is raw teleport cost).
        world, self_state = _make_world(fuel=30)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
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
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
        # default_count=15: below low but above break; radar=0 so no scan
        # radar_count=13: above break (12) so critical path doesn't fire; radar stock
        # doesn't matter for this test since viewport is already scanned
        inventory = _make_inventory(default_count=15, radar_count=13)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason"] == "search_collect_local"
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
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
        # default_count=15, radar_count=13: above break, viewport scanned → search hop path
        inventory = _make_inventory(default_count=15, radar_count=13)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason"] == "search_collect_local"
        assert decision["command"]["cmd_type"] == "teleport"


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
        inventory = _make_inventory(default_count=3)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason"] == "equipment_locked"

    def test_locked_critical_equipment_target_drives_recovery_owner(self) -> None:
        """A locked critical equipment target stays under COLLECT.

        Earlier the test asserted the lock would clear when the
        underlying tile was water-locked, but the actual behaviour
        under the 2026-06-22 resume-threshold mode-entry rule is
        that COLLECT owns the tick, dispatches a pickup
        attempt at the locked tile, and lets the server's reject
        (e.g. ``Empty container`` / ``You can't go there!``) clear
        the lock via the `_clear_command_error` path. The decision
        on this tick is therefore a normal recovery action, not a
        crash, and the mode label is the contract being asserted.
        """
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
        # default_count=5: every counter below resume → mode-entry triggers.
        inventory = _make_inventory(default_count=5)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "COLLECT"


class TestFuelSearchFallbacks:
    """Tests for fuel search fallback paths."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_locked_fuel_on_water_releases_lock(self) -> None:
        """A water-locked fuel target is released by the lock-continuation.

        User contract (2026-06-26): no pickup dispatched at a target
        the bot cannot walk to. The lock-continuation releases the
        lock and falls through to forage / search-hop.
        """
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        containers = {"105,105": _c(105, 105, 700, True)}
        world, self_state = _make_world(fuel=150, containers=containers)
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

        assert "fuel=700" not in decision["behavior"]["reason"]
        assert decision["command"]["cmd_type"] != "pickup_fuel"

    def test_fuel_search_hop_when_scanned_no_visible_fuel(self) -> None:
        """Fuel search hops to fresh sector when viewport tiles fully swept."""
        world, self_state = _make_world(fuel=150, scanned=True)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason"] == "search_collect_local"

    def test_fuel_raises_when_no_hop_affordable_and_no_atlas_dot(self) -> None:
        """Fuel recovery raises when no productive action remains.

        The viewport-edge walk fallback was removed 2026-06-22 (per-tile
        fuel cost for no visibility gain) and the map_intel terminal
        was removed the same day. With no atlas-known dot, no
        affordable search hop, and a scanned viewport, the bot has
        nothing legal to do; raising surfaces the wedged state loudly.
        """
        import pytest

        # Fuel below the short-hop cost (8 * 6 = 48). The
        # ``hunt_min_fuel`` reserve drop (2026-06-24) means stranding
        # now requires fuel < raw teleport cost.
        world, self_state = _make_world(fuel=30, scanned=True)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        with pytest.raises(SessionExitError, match="COLLECT owner produced no decision"):
            decide(world, self_state, ai_state, inventory, 100000, None)

    def test_fuel_recovery_raises_when_all_paths_are_blocked(self) -> None:
        """Durable fuel recovery raises when boxed in.

        With every viewport tile already scanned (bot-side coverage),
        every viewport map tile water, no affordable hop, and no fuel
        dots, the bot is genuinely wedged. The map_intel terminal
        was removed 2026-06-22; the owner now raises so the wedged
        state can't be missed in production logs.
        """
        import pytest

        from tests.in_memory_terrain_map import InMemoryTerrainMap

        terrain_data: dict[tuple[int, int], str] = {}
        for x in range(92, 108):
            for y in range(92, 108):
                terrain_data[(x, y)] = "W"
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        # Fuel below the short-hop cost so no teleport is affordable.
        world, self_state = _make_world(fuel=30, scanned=True)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        with pytest.raises(SessionExitError, match="COLLECT owner produced no decision"):
            decide(world, self_state, ai_state, inventory, 100000, terrain)
