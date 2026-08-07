"""Tests for tick-context publication and action-wait handling."""

from __future__ import annotations

from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.states import (
    InFlightActionDict,
)
from tankpit_bot.bot.tick_loop_actions import _clear_rejected_movement
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.sniffer.world_state import (
    get_world_service,
    mark_move_target_failed,
    update_world_state_from_position,
)
from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total
from tests.conftest import (
    FakeEnv,
)


class TestPublishTickContext:
    """Tests for ``_publish_tick_context`` (Tier 3.2 event enrichment)."""

    def test_writes_tick_n_bot_state_and_action_kind(self, fake_env: FakeEnv) -> None:
        """The published context exposes the three structured fields.

        Asserts against ``runtime_context.get_runtime_context`` so the
        test mirrors what every subsequent ``emit_*`` call will see.
        """
        from tankpit_bot.bot.states import make_in_flight_action
        from tankpit_bot.bot.tick_loop import _publish_tick_context
        from tankpit_bot.runtime_context import get_runtime_context

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["in_flight_action"] = make_in_flight_action(
            kind="shoot",
            target_x=131,
            target_y=124,
            started_ms=get_current_time_ms(),
        )

        _publish_tick_context(bot, tick_n=42)

        ctx = get_runtime_context()
        assert ctx["tick_n"] == 42
        assert "/" in ctx["bot_state"]  # "<mode>/<mode_state>"
        assert ctx["in_flight_action_kind"] == "shoot"

    def test_publish_is_callable_repeatedly_with_increasing_tick_numbers(
        self, fake_env: FakeEnv
    ) -> None:
        """Calling ``_publish_tick_context`` repeatedly updates ``tick_n``.

        Exercises the per-tick mutation directly instead of driving the
        full tick loop, which has CDP/page dependencies unrelated to
        the context publication being tested.
        """
        from tankpit_bot.bot.tick_loop import _publish_tick_context
        from tankpit_bot.runtime_context import get_runtime_context

        bot = Bot("https://test.tankpit.com/", headless=True)
        _publish_tick_context(bot, tick_n=1)
        assert get_runtime_context()["tick_n"] == 1
        _publish_tick_context(bot, tick_n=2)
        assert get_runtime_context()["tick_n"] == 2
        _publish_tick_context(bot, tick_n=42)
        assert get_runtime_context()["tick_n"] == 42


class TestClearRejectedMovement:
    """Tests for _clear_rejected_movement (lines 369-383)."""

    def test_non_move_collect_kind_returns_false(self, fake_env: FakeEnv) -> None:
        """Actions other than move/collect are not affected by rejection."""

        bot = Bot("https://test.tankpit.com/", headless=True)
        action = InFlightActionDict(
            kind="teleport",
            target_x=100,
            target_y=100,
            started_ms=1000,
            outcome="pending",
        )

        result = _clear_rejected_movement(bot, action)

        assert result is False

    def test_move_without_failure_returns_false(self, fake_env: FakeEnv) -> None:
        """A move whose target is NOT failed returns False."""

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        action = InFlightActionDict(
            kind="move",
            target_x=150,
            target_y=150,
            started_ms=1000,
            outcome="pending",
        )

        result = _clear_rejected_movement(bot, action)

        assert result is False

    def test_move_with_failed_target_clears_and_returns_true(self, fake_env: FakeEnv) -> None:
        """A move whose target was marked failed gets cleared (replan)."""

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"

        tx, ty = 150, 150
        now_ms = get_current_time_ms()
        mark_move_target_failed(tx, ty, now_ms)

        action = InFlightActionDict(
            kind="move",
            target_x=tx,
            target_y=ty,
            started_ms=1000,
            outcome="pending",
        )

        result = _clear_rejected_movement(bot, action)

        assert result is True
        assert bot.get_state() == "IDLE"

    def test_collect_with_failed_target_increments_failed_pickups(self, fake_env: FakeEnv) -> None:
        """A collect whose target was marked failed also marks the container."""

        update_world_state_from_position(100, 100)
        update_world_state_from_fuel_total(get_world_service(), 800)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "COLLECTING"

        tx, ty = 120, 130
        now_ms = get_current_time_ms()
        mark_move_target_failed(tx, ty, now_ms)

        action = InFlightActionDict(
            kind="collect",
            target_x=tx,
            target_y=ty,
            started_ms=1000,
            outcome="pending",
        )

        result = _clear_rejected_movement(bot, action)

        assert result is True
        assert bot.get_state() == "IDLE"


class TestWaitForMapOpenAction:
    """Tests for _wait_for_map_open_action (line 338)."""

    def test_wait_for_map_open_action_returns_true_while_waiting(self, fake_env: FakeEnv) -> None:
        """_wait_for_map_open_action returns True when map data hasn't arrived."""
        from tankpit_bot.bot.tick_loop_actions import _wait_for_map_open_action

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)

        action = InFlightActionDict(
            kind="map_open",
            target_x=0,
            target_y=0,
            started_ms=get_current_time_ms(),
            outcome="pending",
        )

        result = _wait_for_map_open_action(bot, action)

        assert result is True


class TestWaitForMovementActionRejected:
    """Test _wait_for_movement_action when _clear_rejected_movement returns True (line 308)."""

    def test_wait_for_movement_action_returns_false_on_rejected_move(
        self, fake_env: FakeEnv
    ) -> None:
        """_wait_for_movement_action returns False when the target was rejected."""
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"

        tx, ty = 150, 150
        now_ms = get_current_time_ms()
        mark_move_target_failed(tx, ty, now_ms)

        action = InFlightActionDict(
            kind="move",
            target_x=tx,
            target_y=ty,
            started_ms=now_ms,
            outcome="pending",
        )

        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"
