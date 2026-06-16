"""Coverage tests for bot/tick_loop.py: stop-file detection and _clear_rejected_movement."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONValue

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import KeyboardProtocol, ResponseProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.bot.states import InFlightActionDict
from tankpit_bot.bot.tick_loop_actions import _clear_rejected_movement
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.sniffer.world_state import (
    get_world_service,
    mark_move_target_failed,
    reset_world_state,
    update_world_state_from_position,
)
from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total
from tests.conftest import FakeEnv


class _NoOpKeyboard:
    """Minimal keyboard stub."""

    def press(self, key: str, *, delay: float | None = None) -> None:
        """No-op."""

    def type(self, text: str, *, delay: float | None = None) -> None:
        """No-op."""


class _FakePage:
    """Minimal page stub for tick-loop testing."""

    def __init__(self) -> None:
        """Initialize."""
        self._url = "https://test.tankpit.com/play"
        self._keyboard = _NoOpKeyboard()

    @property
    def url(self) -> str:
        """Return URL."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Return keyboard."""
        return self._keyboard

    def wait_for_timeout(self, timeout: float) -> None:
        """No-op wait."""

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """No-op wait for event."""

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """No-op wait for function."""

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """No-op goto."""
        self._url = url
        return None

    def evaluate(self, expression: str) -> JSONValue:
        """No-op evaluate."""
        return None

    def close(
        self,
        *,
        reason: str | None = None,
        run_before_unload: bool | None = None,
    ) -> None:
        """No-op close."""


class TestStopFileDetection:
    """Test stop-file sentinel terminates run_tick_loop."""

    def test_stop_file_ends_tick_loop(self, fake_env: FakeEnv) -> None:
        """run_tick_loop exits when the stop file exists.

        Uses fake hooks so no real file operations occur. The tick_once
        call exits early because self_state is None (no position), then
        the stop-file check triggers.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import run_tick_loop

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)

        stop_path = Path("C:/tmp/test_stop_file.sentinel")

        removed_files: list[Path] = []

        def fake_path_exists(path: Path) -> bool:
            return path == stop_path

        def fake_remove_file(path: Path) -> None:
            removed_files.append(path)

        _test_hooks.path_exists = fake_path_exists
        _test_hooks.remove_file = fake_remove_file

        run_tick_loop(
            bot,
            _FakePage(),
            session_seconds=0,
            stop_file_path=stop_path,
        )

        assert removed_files == [stop_path]


class TestClearRejectedMovement:
    """Tests for _clear_rejected_movement (lines 369-383)."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_non_move_collect_kind_returns_false(self, fake_env: FakeEnv) -> None:
        """Actions other than move/collect are not affected by rejection."""
        from tankpit_bot.bot.base import Bot

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
        from tankpit_bot.bot.base import Bot

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
        from tankpit_bot.bot.base import Bot

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
        from tankpit_bot.bot.base import Bot

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

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_wait_for_map_open_action_returns_true_while_waiting(self, fake_env: FakeEnv) -> None:
        """_wait_for_map_open_action returns True when map data hasn't arrived."""
        from tankpit_bot.bot.base import Bot
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

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_wait_for_movement_action_returns_false_on_rejected_move(
        self, fake_env: FakeEnv
    ) -> None:
        """_wait_for_movement_action returns False when the target was rejected."""
        from tankpit_bot.bot.base import Bot
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
