"""Tests for locked-target behaviour in the strategy cascade.

Equipment and fuel locks, including the critical-equipment override.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.types import (
    AIStateDict,
)
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.sniffer.world_service import WorldService
from tests.bot.ai._strategy_fixtures import (
    _c,
    _make_inventory,
    _make_world,
    _scanned_ai_state,
)


class TestLockedEquipmentTarget:
    """Tests for locked equipment target continuation."""

    def setup_method(self) -> None:
        """Reset world state."""
        self.ws = WorldService()
        self.ws.update_world_state_from_position(100, 100)

    def test_continues_locked_equipment_target(self) -> None:
        """Locked equipment target is continued when still actionable."""
        ws = self.ws
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

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "equipment_locked"

    def test_locked_equipment_target_on_water_holds_plan(self) -> None:
        """A water-locked equipment target holds through the tick.

        Committed-intent law ([[committed-intent]], run
        bot-20260730-032x ticks 361/366/371): transient
        inexecutability is not invalidity. The continuation yields
        the tick to the rest of the cascade but KEEPS the plan — a
        ferry or a better approach can serve it later, and only a
        structural verdict (the move-failed mark, unservability)
        releases it. A fresh ferry floats ON the target's pond so the
        ride lane keeps the target servable and the hold genuinely
        transient.
        """
        from tankpit_bot.state.types import make_terrain_tile
        from tankpit_bot.types.constants import TERRAIN_FERRY
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = self.ws
        containers = {"105,105": _c(105, 105, 0, False)}
        world, self_state = _make_world(containers=containers, fuel=800)
        world["terrain"]["104,105"] = make_terrain_tile(104, 105, TERRAIN_FERRY, observed_ms=100000)
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

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain, ws=ws)

        assert decision["behavior"]["reason_kind"] != "equipment_locked"
        assert decision["command"]["cmd_type"] != "pickup_equipment"
        assert decision["updated_ai_state"]["resource_target_kind"] == "equipment"

    def test_locked_equipment_target_holds_when_teleport_unaffordable(self) -> None:
        """An unaffordable approach holds the plan instead of dropping it.

        Fuel recovers; the plan survives until a genuine release
        gate (superior candidate, validity, move-failed mark) fires.
        """
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        # Far target outside viewport — all viewport tiles water so no walkable approach,
        # and the direct teleport is unaffordable.
        ws = self.ws
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

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain, ws=ws)

        # Teleport to (200,200) costs 1200 fuel with only 550 held: the
        # tick goes elsewhere but the plan is kept.
        assert decision["behavior"]["reason_kind"] != "equipment_locked"
        assert decision["updated_ai_state"]["resource_target_kind"] == "equipment"

    def test_locked_equipment_target_releases_on_move_failed_mark(self) -> None:
        """The server-confirmed move-failed mark is the structural release.

        A cant_go-marked destination means the approach is dead on
        server truth, so the continuation releases the plan with the
        ``not_executable`` reason instead of holding it.
        """
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = self.ws
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
        ws.mark_move_target_failed(105, 105, 99000)
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

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain, ws=ws)

        assert decision["behavior"]["reason_kind"] != "equipment_locked"
        assert decision["updated_ai_state"]["resource_target_kind"] == ""


class TestLockedFuelTarget:
    """Tests for locked fuel target continuation."""

    def setup_method(self) -> None:
        """Reset world state."""
        self.ws = WorldService()
        self.ws.update_world_state_from_position(100, 100)

    def test_continues_locked_fuel_target(self) -> None:
        """Locked fuel target is continued when still actionable."""
        ws = self.ws
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

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "fuel_locked"
        assert decision["behavior"]["reason_context"]["volume"] == 700

    def test_water_locked_fuel_target_holds_plan(self) -> None:
        """Transient fuel-lock inexecutability keeps the plan.

        Same committed-intent law as the equipment lock: the
        continuation yields the tick but the plan survives for a
        later ferry or approach.
        """
        from tankpit_bot.bot.ai.collect_locks import continue_or_release_fuel_lock
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.state.types import make_terrain_tile
        from tankpit_bot.types.constants import TERRAIN_FERRY
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = self.ws
        containers = {"105,105": _c(105, 105, 700, True)}
        world, self_state = _make_world(fuel=800, containers=containers)
        world["terrain"]["104,105"] = make_terrain_tile(104, 105, TERRAIN_FERRY, observed_ms=100000)
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
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            _make_inventory(),
            100000,
            terrain,
            "",
            ws=ws,
        )

        decision, state = continue_or_release_fuel_lock(
            ctx, ctx.base, ctx.filtered["containers"]["105,105"]
        )

        assert decision is None
        assert state["resource_target_kind"] == "fuel"

    def test_move_failed_fuel_target_releases_plan(self) -> None:
        """The move-failed mark structurally releases the fuel plan."""
        from tankpit_bot.bot.ai.collect_locks import continue_or_release_fuel_lock
        from tankpit_bot.bot.ai.context import DecideCtx
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = self.ws
        containers = {"105,105": _c(105, 105, 700, True)}
        world, self_state = _make_world(fuel=800, containers=containers)
        terrain_data: dict[tuple[int, int], str] = {
            (105, 105): "W",
            (104, 105): "W",
            (106, 105): "W",
            (105, 104): "W",
            (105, 106): "W",
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        ws.mark_move_target_failed(105, 105, 99000)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "resource_target_kind": "fuel",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            _make_inventory(),
            100000,
            terrain,
            "",
            ws=ws,
        )

        decision, state = continue_or_release_fuel_lock(
            ctx, ctx.base, ctx.filtered["containers"]["105,105"]
        )

        assert decision is None
        assert state["resource_target_kind"] == ""


class TestCriticalEquipmentLockedTarget:
    """Tests for critical equipment locked target paths."""

    def setup_method(self) -> None:
        """Reset world state."""
        self.ws = WorldService()
        self.ws.update_world_state_from_position(100, 100)

    def test_continues_locked_critical_equipment_target(self) -> None:
        """Critical locked equipment target is continued when executable."""
        ws = self.ws
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

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "equipment_locked"

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

        ws = self.ws
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

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"
