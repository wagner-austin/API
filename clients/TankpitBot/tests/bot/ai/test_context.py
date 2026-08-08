"""Tests for ai.context module — resource target helpers and equipment checks."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport,
    locked_resource_target,
    set_resource_target,
    teleport_fuel_cost_to,
)
from tankpit_bot.bot.ai.types import make_initial_ai_state
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import (
    SelfStateDict,
    WorldStateDict,
    make_container_state,
    make_self_state,
    make_viewport_state,
)


def _world_with_container(
    x: int,
    y: int,
    is_fuel: bool,
    volume: int,
    failed_pickups: int = 0,
) -> WorldStateDict:
    """Build a minimal world state with one container."""
    key = f"{x},{y}"
    return WorldStateDict(
        self_state=None,
        tanks={},
        containers={
            key: make_container_state(
                x=x,
                y=y,
                is_fuel=is_fuel,
                volume=volume,
                timestamp_ms=0,
                failed_pickups=failed_pickups,
            )
        },
        mines={},
        terrain={},
        viewport=make_viewport_state(left=0, top=0, width=16, height=16),
        scanned_tiles={},
        timestamp_ms=0,
    )


class TestLockedResourceTarget:
    """Tests for locked_resource_target via DecideCtx."""

    def test_ctx_exposes_durable_mode_fields(self) -> None:
        """DecideCtx exposes the durable top-level mode and substate."""
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.bot.ai.types import AIStateDict

        ws = WorldService()
        world = _world_with_container(10, 20, True, 500)
        self_state = _self_state()
        world["self_state"] = self_state
        ai_state = AIStateDict(
            **{
                **make_initial_ai_state(),
                "mode": "HUNT",
                "mode_state": "ACQUIRE",
                "mode_started_ms": 999,
            }
        )

        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            _dummy_inventory(),
            100000,
            None,
            "",
            ws=ws,
        )

        assert ctx.mode == "HUNT"
        assert ctx.mode_state == "ACQUIRE"
        assert ctx.mode_started_ms == 999

    def test_returns_none_for_wrong_kind(self) -> None:
        """Locked target returns None when kind doesn't match."""
        from tankpit_bot.bot.ai.context import DecideCtx

        ws = WorldService()
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)
        world = _world_with_container(10, 20, True, 500)
        self_state = _self_state()
        world["self_state"] = self_state
        ctx = DecideCtx(world, self_state, state, _dummy_inventory(), 100000, None, "", ws=ws)
        _, target = locked_resource_target(ctx, "equipment")
        assert target is None

    def test_clears_when_container_missing(self) -> None:
        """Locked target clears when container not in filtered world."""
        from tankpit_bot.bot.ai.context import DecideCtx

        ws = WorldService()
        state = set_resource_target(make_initial_ai_state(), "fuel", 99, 99)
        world = _world_with_container(10, 20, True, 500)
        self_state = _self_state()
        world["self_state"] = self_state
        ctx = DecideCtx(world, self_state, state, _dummy_inventory(), 100000, None, "", ws=ws)
        base_state, target = locked_resource_target(ctx, "fuel")
        assert target is None
        assert base_state["resource_target_kind"] == ""

    def test_clears_when_kind_mismatch_fuel(self) -> None:
        """Locked fuel target clears when container is equipment."""
        from tankpit_bot.bot.ai.context import DecideCtx

        ws = WorldService()
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)
        world = _world_with_container(10, 20, False, 0)
        self_state = _self_state()
        world["self_state"] = self_state
        ctx = DecideCtx(world, self_state, state, _dummy_inventory(), 100000, None, "", ws=ws)
        _, target = locked_resource_target(ctx, "fuel")
        assert target is None

    def test_clears_when_kind_mismatch_equipment(self) -> None:
        """Locked equipment target clears when container is fuel."""
        from tankpit_bot.bot.ai.context import DecideCtx

        ws = WorldService()
        state = set_resource_target(make_initial_ai_state(), "equipment", 10, 20)
        world = _world_with_container(10, 20, True, 500)
        self_state = _self_state()
        world["self_state"] = self_state
        ctx = DecideCtx(world, self_state, state, _dummy_inventory(), 100000, None, "", ws=ws)
        _, target = locked_resource_target(ctx, "equipment")
        assert target is None

    def test_clears_when_failed_pickups(self) -> None:
        """Locked target clears when container has failed pickups."""
        from tankpit_bot.bot.ai.context import DecideCtx

        ws = WorldService()
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)
        world = _world_with_container(10, 20, True, 500, failed_pickups=2)
        self_state = _self_state()
        world["self_state"] = self_state
        ctx = DecideCtx(world, self_state, state, _dummy_inventory(), 100000, None, "", ws=ws)
        _, target = locked_resource_target(ctx, "fuel")
        assert target is None


class TestLockedResourceTargetMissingContainer:
    """Test locked_resource_target when container absent from filtered world."""

    def test_raises_when_container_absent_from_filtered(self) -> None:
        """Mutating filtered after normalization breaks the invariant loudly.

        ``ctx.base`` is normalized against ``ctx.filtered`` at
        construction, so a surviving lock kind guarantees the container
        exists; the lookup deliberately raises instead of silently
        clearing when that invariant is violated.
        """
        import pytest

        from tankpit_bot.bot.ai.context import DecideCtx

        ws = WorldService()
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)
        world = _world_with_container(10, 20, True, 500)
        self_state = _self_state()
        world["self_state"] = self_state
        ctx = DecideCtx(world, self_state, state, _dummy_inventory(), 100000, None, "", ws=ws)
        assert ctx.base["resource_target_kind"] == "fuel"
        ctx.filtered = WorldStateDict(**{**ctx.filtered, "containers": {}})
        with pytest.raises(KeyError):
            locked_resource_target(ctx, "fuel")


class TestMakePickupCommandError:
    """Test _make_pickup_command raises on unknown kind."""

    def test_unknown_kind_raises(self) -> None:
        """Unknown pickup kind raises ValueError."""
        import pytest

        from tankpit_bot.bot.ai.movement import _make_pickup_command

        with pytest.raises(ValueError, match="Unknown pickup kind"):
            _make_pickup_command("unknown", 10, 20)


def _self_state() -> SelfStateDict:
    """Build a dummy self state for testing."""
    return make_self_state(
        tank_id=1,
        x=100,
        y=100,
        team=1,
        rank=0,
        fuel=500,
        leaderboard_position=0,
    )


def _dummy_inventory() -> InventoryState:
    """Build a dummy inventory for testing."""
    item = InventoryItem(count=30, enabled=True)
    return InventoryState(
        armor_shields=item,
        dual_shots=item,
        missile_shots=item,
        homing_shots=item,
        extra_radars=item,
    )


class TestTeleportFuelHelpers:
    """Tests for teleport fuel cost and affordability helpers."""

    def test_reports_exact_axis_aligned_teleport_cost(self) -> None:
        """Teleport helper returns the exact cost for an axis-aligned jump."""
        ws = WorldService()
        world = _world_with_container(10, 20, True, 100)
        self_state = make_self_state(
            tank_id=1,
            x=93,
            y=106,
            team=1,
            rank=0,
            fuel=800,
            leaderboard_position=0,
        )
        world["self_state"] = self_state
        ctx = DecideCtx(
            world,
            self_state,
            make_initial_ai_state(),
            _dummy_inventory(),
            100000,
            None,
            "",
            ws=ws,
        )

        assert teleport_fuel_cost_to(ctx, 3, 106) == 540

    def test_reports_exact_diagonal_teleport_cost(self) -> None:
        """Teleport helper matches the sniffed long diagonal sample."""
        ws = WorldService()
        world = _world_with_container(10, 20, True, 100)
        self_state = make_self_state(
            tank_id=1,
            x=6,
            y=172,
            team=1,
            rank=0,
            fuel=1100,
            leaderboard_position=0,
        )
        world["self_state"] = self_state
        ctx = DecideCtx(
            world,
            self_state,
            make_initial_ai_state(),
            _dummy_inventory(),
            100000,
            None,
            "",
            ws=ws,
        )

        assert teleport_fuel_cost_to(ctx, 86, 90) == 687

    def test_can_afford_teleport_uses_exact_cost(self) -> None:
        """Exact affordability accepts only when current fuel covers the jump."""
        ws = WorldService()
        world = _world_with_container(10, 20, True, 100)
        self_state = make_self_state(
            tank_id=1,
            x=96,
            y=95,
            team=1,
            rank=0,
            fuel=12,
            leaderboard_position=0,
        )
        world["self_state"] = self_state
        ctx = DecideCtx(
            world,
            self_state,
            make_initial_ai_state(),
            _dummy_inventory(),
            100000,
            None,
            "",
            ws=ws,
        )

        assert can_afford_teleport(ctx, 96, 97) is True
        assert can_afford_teleport(ctx, 94, 107) is False
