"""Tests for the committed-intent layer (collect-plan semantics)."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.bot.ai.intent import (
    CollectPlanDict,
    current_collect_plan,
    decode_collect_plan,
    encode_collect_plan,
    plan_completes_here,
    release_collect_plan,
    set_resource_target,
    validate_collect_plan,
)
from tankpit_bot.bot.ai.types import make_initial_ai_state
from tankpit_bot.state.types import (
    WorldStateDict,
    make_container_state,
    make_viewport_state,
)


def _world_with_container(
    x: int,
    y: int,
    is_fuel: bool,
    volume: int,
    failed_pickups: int = 0,
) -> WorldStateDict:
    """Build a minimal world state with one container.

    Args:
        x: Container X.
        y: Container Y.
        is_fuel: Whether the container is fuel.
        volume: Container volume.
        failed_pickups: Failed pickup count for the container.

    Returns:
        World state holding exactly that container.
    """
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


class TestCollectPlanCodecs:
    """Tests for encode_collect_plan / decode_collect_plan."""

    def test_fuel_plan_round_trips(self) -> None:
        """A fuel plan survives encode -> decode unchanged."""
        plan = CollectPlanDict(kind="fuel", target_x=12, target_y=34)

        assert decode_collect_plan(encode_collect_plan(plan)) == plan

    def test_equipment_plan_round_trips(self) -> None:
        """An equipment plan survives encode -> decode unchanged."""
        plan = CollectPlanDict(kind="equipment", target_x=7, target_y=9)

        assert decode_collect_plan(encode_collect_plan(plan)) == plan

    def test_decode_rejects_unknown_kind(self) -> None:
        """A kind outside the closed vocabulary raises."""
        with pytest.raises(JSONTypeError, match="kind must be one of"):
            decode_collect_plan({"kind": "bogus", "target_x": 1, "target_y": 2})

    def test_decode_rejects_missing_coordinate(self) -> None:
        """A missing target field raises instead of defaulting."""
        with pytest.raises(JSONTypeError):
            decode_collect_plan({"kind": "fuel", "target_x": 1})


class TestCurrentCollectPlan:
    """Tests for reading the held plan out of AI state."""

    def test_no_lock_reads_as_no_plan(self) -> None:
        """The initial state holds no plan."""
        assert current_collect_plan(make_initial_ai_state()) is None

    def test_fuel_lock_reads_as_fuel_plan(self) -> None:
        """A held fuel lock is the fuel plan."""
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)

        plan = current_collect_plan(state)

        assert plan == CollectPlanDict(kind="fuel", target_x=10, target_y=20)

    def test_equipment_lock_reads_as_equipment_plan(self) -> None:
        """A held equipment lock is the equipment plan."""
        state = set_resource_target(make_initial_ai_state(), "equipment", 3, 4)

        plan = current_collect_plan(state)

        assert plan == CollectPlanDict(kind="equipment", target_x=3, target_y=4)

    def test_unknown_kind_reads_as_no_plan(self) -> None:
        """A lock kind outside the vocabulary is not a plan."""
        state = set_resource_target(make_initial_ai_state(), "bogus", 10, 20)

        assert current_collect_plan(state) is None


class TestPlanCompletesHere:
    """Tests for the serve-reach completion predicate."""

    def test_standing_on_the_target_completes(self) -> None:
        """Distance zero is inside the serve reach."""
        plan = CollectPlanDict(kind="equipment", target_x=100, target_y=100)

        assert plan_completes_here(plan, 100, 100) is True

    def test_cardinal_adjacency_completes(self) -> None:
        """Distance one (the auto-pick reach) completes."""
        plan = CollectPlanDict(kind="fuel", target_x=101, target_y=100)

        assert plan_completes_here(plan, 100, 100) is True

    def test_two_tiles_out_does_not_complete(self) -> None:
        """Distance two is travel, not completion."""
        plan = CollectPlanDict(kind="fuel", target_x=101, target_y=101)

        assert plan_completes_here(plan, 100, 100) is False


class TestReleaseCollectPlan:
    """Tests for the sanctioned release path."""

    def test_held_plan_is_cleared(self) -> None:
        """Releasing a held plan zeroes the lock fields."""
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)

        result = release_collect_plan(state, reason="superior_candidate")

        assert result["resource_target_kind"] == ""
        assert result["resource_target_x"] == 0
        assert result["resource_target_y"] == 0

    def test_release_without_a_plan_passes_through(self) -> None:
        """Releasing nothing is nothing: the same state comes back."""
        state = make_initial_ai_state()

        assert release_collect_plan(state, reason="landing_scan_reset") is state


class TestValidateCollectPlan:
    """Tests for the per-tick plan validity pass (lifted from context)."""

    def test_releases_invalid_kind(self) -> None:
        """Non-fuel/equipment kind is released."""
        state = set_resource_target(make_initial_ai_state(), "bogus", 10, 20)

        result = validate_collect_plan(state, _world_with_container(10, 20, True, 100))

        assert result["resource_target_kind"] == ""

    def test_releases_missing_container(self) -> None:
        """Plan released when its container is no longer in the world."""
        state = set_resource_target(make_initial_ai_state(), "fuel", 99, 99)

        result = validate_collect_plan(state, _world_with_container(10, 20, True, 100))

        assert result["resource_target_kind"] == ""

    def test_releases_fuel_plan_targeting_equipment(self) -> None:
        """Fuel plan pointing at an equipment container is released."""
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)

        result = validate_collect_plan(state, _world_with_container(10, 20, False, 0))

        assert result["resource_target_kind"] == ""

    def test_releases_equipment_plan_targeting_fuel(self) -> None:
        """Equipment plan pointing at a fuel container is released."""
        state = set_resource_target(make_initial_ai_state(), "equipment", 10, 20)

        result = validate_collect_plan(state, _world_with_container(10, 20, True, 500))

        assert result["resource_target_kind"] == ""

    def test_releases_failed_pickup_target(self) -> None:
        """Plan on a container with failed pickups is released."""
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)

        result = validate_collect_plan(
            state,
            _world_with_container(10, 20, True, 500, failed_pickups=1),
        )

        assert result["resource_target_kind"] == ""

    def test_preserves_old_target(self) -> None:
        """A plan on an old but tracked container survives.

        The 30 s freshness TTL was removed 2026-07-06: in-viewport
        containers are wire-truthful under the truth layer, so age
        alone never invalidates a plan.
        """
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)
        world = _world_with_container(10, 20, True, 500)
        world["timestamp_ms"] = 31000

        result = validate_collect_plan(state, world)

        assert result["resource_target_kind"] == "fuel"

    def test_preserves_valid_fuel_plan(self) -> None:
        """A valid fuel plan is preserved with its coordinates."""
        state = set_resource_target(make_initial_ai_state(), "fuel", 10, 20)

        result = validate_collect_plan(state, _world_with_container(10, 20, True, 500))

        assert result["resource_target_kind"] == "fuel"
        assert result["resource_target_x"] == 10

    def test_preserves_valid_equipment_plan(self) -> None:
        """A valid equipment plan is preserved."""
        state = set_resource_target(make_initial_ai_state(), "equipment", 10, 20)

        result = validate_collect_plan(state, _world_with_container(10, 20, False, 0))

        assert result["resource_target_kind"] == "equipment"

    def test_no_plan_passes_through_unchanged(self) -> None:
        """A state without a plan is returned as-is, no rebuild."""
        state = make_initial_ai_state()

        result = validate_collect_plan(state, _world_with_container(10, 20, True, 500))

        assert result is state
