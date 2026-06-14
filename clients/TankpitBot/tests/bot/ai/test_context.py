"""Tests for ai.context module — resource target helpers and equipment checks."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport,
    can_afford_teleport_search,
    locked_resource_target,
    normalize_resource_target,
    set_resource_target,
    teleport_fuel_cost_to,
)
from tankpit_bot.bot.ai.types import make_initial_ai_state
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.state.types import SelfStateDict, WorldStateDict, make_container_state


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
        viewport={"left": 0, "top": 0, "width": 16, "height": 16},
        scanned_viewports={},
        map_fuel_dots={},
        timestamp_ms=0,
    )


class TestNormalizeResourceTarget:
    """Tests for normalize_resource_target."""

    def test_clears_invalid_kind(self) -> None:
        """Non-fuel/equipment kind is cleared."""
        state = make_initial_ai_state()
        state = set_resource_target(state, "bogus", 10, 20)
        result = normalize_resource_target(
            state,
            _world_with_container(10, 20, True, 100),
            now_ms=0,
        )
        assert result["resource_target_kind"] == ""

    def test_clears_missing_container(self) -> None:
        """Target cleared when container no longer in world."""
        state = set_resource_target(make_initial_ai_state(), "fuel", 99, 99)
        world = _world_with_container(10, 20, True, 100)
        result = normalize_resource_target(state, world, now_ms=0)
        assert result["resource_target_kind"] == ""

    def test_clears_fuel_targeting_equipment(self) -> None:
        """Fuel target pointing at equipment container is cleared."""
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)
        world = _world_with_container(10, 20, False, 0)
        result = normalize_resource_target(state, world, now_ms=0)
        assert result["resource_target_kind"] == ""

    def test_clears_equipment_targeting_fuel(self) -> None:
        """Equipment target pointing at fuel container is cleared."""
        state = set_resource_target(make_initial_ai_state(), "equipment", 10, 20)
        world = _world_with_container(10, 20, True, 500)
        result = normalize_resource_target(state, world, now_ms=0)
        assert result["resource_target_kind"] == ""

    def test_clears_failed_pickup(self) -> None:
        """Target with failed_pickups > 0 is cleared."""
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)
        world = _world_with_container(10, 20, True, 500, failed_pickups=1)
        result = normalize_resource_target(state, world, now_ms=0)
        assert result["resource_target_kind"] == ""

    def test_clears_stale_target(self) -> None:
        """A lock on a container past the freshness TTL is cleared.

        The lock previously skipped the freshness check that candidate
        selection applies, so live run 20260610 walked across the map
        to a long-drained container.
        """
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)
        world = _world_with_container(10, 20, True, 500)
        result = normalize_resource_target(state, world, 31000)
        assert result["resource_target_kind"] == ""

    def test_preserves_valid_fuel_target(self) -> None:
        """Valid fuel target is preserved."""
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)
        world = _world_with_container(10, 20, True, 500)
        result = normalize_resource_target(state, world, now_ms=0)
        assert result["resource_target_kind"] == "fuel"
        assert result["resource_target_x"] == 10

    def test_preserves_valid_equipment_target(self) -> None:
        """Valid equipment target is preserved."""
        state = set_resource_target(make_initial_ai_state(), "equipment", 10, 20)
        world = _world_with_container(10, 20, False, 0)
        result = normalize_resource_target(state, world, now_ms=0)
        assert result["resource_target_kind"] == "equipment"


class TestLockedResourceTarget:
    """Tests for locked_resource_target via DecideCtx."""

    def test_ctx_exposes_durable_mode_fields(self) -> None:
        """DecideCtx exposes the durable top-level mode and substate."""
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.bot.ai.types import AIStateDict
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
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

        ctx = DecideCtx(world, self_state, ai_state, _dummy_inventory(), 100000, None, "")

        assert ctx.mode == "HUNT"
        assert ctx.mode_state == "ACQUIRE"
        assert ctx.mode_started_ms == 999
        reset_world_state()

    def test_returns_none_for_wrong_kind(self) -> None:
        """Locked target returns None when kind doesn't match."""
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)
        world = _world_with_container(10, 20, True, 500)
        self_state = _self_state()
        world["self_state"] = self_state
        ctx = DecideCtx(world, self_state, state, _dummy_inventory(), 100000, None, "")
        _, target = locked_resource_target(ctx, "equipment")
        assert target is None
        reset_world_state()

    def test_clears_when_container_missing(self) -> None:
        """Locked target clears when container not in filtered world."""
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        state = set_resource_target(make_initial_ai_state(), "fuel", 99, 99)
        world = _world_with_container(10, 20, True, 500)
        self_state = _self_state()
        world["self_state"] = self_state
        ctx = DecideCtx(world, self_state, state, _dummy_inventory(), 100000, None, "")
        base_state, target = locked_resource_target(ctx, "fuel")
        assert target is None
        assert base_state["resource_target_kind"] == ""
        reset_world_state()

    def test_clears_when_kind_mismatch_fuel(self) -> None:
        """Locked fuel target clears when container is equipment."""
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)
        world = _world_with_container(10, 20, False, 0)
        self_state = _self_state()
        world["self_state"] = self_state
        ctx = DecideCtx(world, self_state, state, _dummy_inventory(), 100000, None, "")
        _, target = locked_resource_target(ctx, "fuel")
        assert target is None
        reset_world_state()

    def test_clears_when_kind_mismatch_equipment(self) -> None:
        """Locked equipment target clears when container is fuel."""
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        state = set_resource_target(make_initial_ai_state(), "equipment", 10, 20)
        world = _world_with_container(10, 20, True, 500)
        self_state = _self_state()
        world["self_state"] = self_state
        ctx = DecideCtx(world, self_state, state, _dummy_inventory(), 100000, None, "")
        _, target = locked_resource_target(ctx, "equipment")
        assert target is None
        reset_world_state()

    def test_clears_when_failed_pickups(self) -> None:
        """Locked target clears when container has failed pickups."""
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)
        world = _world_with_container(10, 20, True, 500, failed_pickups=2)
        self_state = _self_state()
        world["self_state"] = self_state
        ctx = DecideCtx(world, self_state, state, _dummy_inventory(), 100000, None, "")
        _, target = locked_resource_target(ctx, "fuel")
        assert target is None
        reset_world_state()


class TestLockedResourceTargetMissingContainer:
    """Test locked_resource_target when container absent from filtered world."""

    def test_clears_when_container_absent_from_filtered(self) -> None:
        """Container present in world but removed from filtered → cleared."""
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)
        world = _world_with_container(10, 20, True, 500)
        self_state = _self_state()
        world["self_state"] = self_state
        ctx = DecideCtx(world, self_state, state, _dummy_inventory(), 100000, None, "")
        # Simulate container missing from filtered (e.g. race condition)
        ctx.filtered = WorldStateDict(**{**ctx.filtered, "containers": {}})
        base_state, target = locked_resource_target(ctx, "fuel")
        assert target is None
        assert base_state["resource_target_kind"] == ""
        reset_world_state()


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
    return SelfStateDict(
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
        world = _world_with_container(10, 20, True, 100)
        self_state = SelfStateDict(
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
        )

        assert teleport_fuel_cost_to(ctx, 3, 106) == 540

    def test_reports_exact_diagonal_teleport_cost(self) -> None:
        """Teleport helper matches the sniffed long diagonal sample."""
        world = _world_with_container(10, 20, True, 100)
        self_state = SelfStateDict(
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
        )

        assert teleport_fuel_cost_to(ctx, 86, 90) == 687

    def test_can_afford_teleport_uses_exact_cost(self) -> None:
        """Exact affordability accepts only when current fuel covers the jump."""
        world = _world_with_container(10, 20, True, 100)
        self_state = SelfStateDict(
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
        )

        assert can_afford_teleport(ctx, 96, 97) is True
        assert can_afford_teleport(ctx, 94, 107) is False

    def test_can_afford_teleport_search_applies_operating_reserve(self) -> None:
        """Search affordability includes the configured post-teleport reserve."""
        world = _world_with_container(10, 20, True, 100)
        self_state = SelfStateDict(
            tank_id=1,
            x=100,
            y=100,
            team=1,
            rank=0,
            fuel=279,
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
        )

        assert can_afford_teleport_search(ctx, 130, 100) is False
