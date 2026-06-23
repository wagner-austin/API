"""Coverage tests for bot/tick_loop.py: stop-file detection and _clear_rejected_movement."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import KeyboardProtocol, ResponseProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.states import ActionKind, InFlightActionDict
from tankpit_bot.bot.tick_loop_actions import _clear_rejected_movement
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.sniffer.world_state import (
    get_world_service,
    mark_move_target_failed,
    reset_world_state,
    update_world_state_from_position,
)
from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total
from tests.conftest import FakeEnv, FakeFileSystem


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


class TestExtractStampFromArchivePath:
    """Tests for ``_extract_stamp_from_archive_path``."""

    def test_returns_stamp_for_canonical_path(self) -> None:
        """The stamp segment between ``bot-`` and ``.events.jsonl`` is returned."""
        from tankpit_bot.bot.tick_loop import _extract_stamp_from_archive_path

        result = _extract_stamp_from_archive_path("runs/bot/bot-20260620-150138.events.jsonl")
        assert result == "20260620-150138"

    def test_returns_stamp_for_windows_separator_path(self) -> None:
        """Windows separators in the input do not affect stamp extraction."""
        from tankpit_bot.bot.tick_loop import _extract_stamp_from_archive_path

        result = _extract_stamp_from_archive_path(r"runs\bot\bot-20260620-150138.events.jsonl")
        assert result == "20260620-150138"

    def test_raises_when_prefix_missing(self) -> None:
        """Archive paths that don't start with ``bot-`` raise."""
        from tankpit_bot.bot.tick_loop import _extract_stamp_from_archive_path

        with pytest.raises(ValueError, match="does not match bot-"):
            _extract_stamp_from_archive_path("runs/bot/probe-20260620-150138.events.jsonl")

    def test_raises_when_suffix_missing(self) -> None:
        """Archive paths that don't end with ``.events.jsonl`` raise."""
        from tankpit_bot.bot.tick_loop import _extract_stamp_from_archive_path

        with pytest.raises(ValueError, match="does not match bot-"):
            _extract_stamp_from_archive_path("runs/bot/bot-20260620-150138.log")


class TestInterruptedExitReason:
    """The interrupt flag should produce ``exit_reason=interrupted`` rows."""

    def setup_method(self) -> None:
        """Reset world state + interrupt flag before each test."""
        from tankpit_bot.bot.tick_loop import reset_interrupt_flag

        reset_world_state()
        reset_interrupt_flag()

    def teardown_method(self) -> None:
        """Reset world state + interrupt flag after each test."""
        from tankpit_bot.bot.tick_loop import reset_interrupt_flag

        reset_world_state()
        reset_interrupt_flag()

    def test_pre_set_interrupt_records_interrupted_row(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """Pre-setting the flag exits at tick 1 with ``interrupted``."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import request_interrupt, run_tick_loop
        from tankpit_bot.diagnostics.runs_index import DEFAULT_INDEX_PATH, decode_row
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        configure_bot_runtime_logging("20260620-150138")
        bot = Bot("https://test.tankpit.com/", headless=True)
        request_interrupt()

        run_tick_loop(
            bot,
            _FakePage(),
            session_seconds=0,
            stop_file_path=Path("C:/tmp/never_exists.sentinel"),
        )

        text = fake_fs.get_written_files()[str(DEFAULT_INDEX_PATH)]
        data_lines = [line for line in text.splitlines() if line and not line.startswith("stamp\t")]
        if len(data_lines) != 1:
            raise AssertionError(f"expected 1 index row, got {len(data_lines)}")
        row = decode_row(data_lines[0])
        assert row["exit_reason"] == "interrupted"
        assert row["ticks"] == 1


class TestAppendIndexRowEndToEnd:
    """End-to-end test of ``_emit_session_scorecard`` -> ``_append_index_row``.

    Configures bot runtime logging so :func:`get_bot_runtime_artifacts`
    returns a real artifact bundle, then asserts the index row landed
    in the fake filesystem with the expected stamp + exit reason.
    """

    def test_emit_session_scorecard_appends_index_row(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """A bot session end writes a row matching the active artifacts."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import _emit_session_scorecard
        from tankpit_bot.diagnostics.runs_index import (
            DEFAULT_INDEX_PATH,
            HEADER_LINE,
            decode_row,
        )
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        configure_bot_runtime_logging("20260620-150138")
        bot = Bot("https://test.tankpit.com/", headless=True)

        _emit_session_scorecard(bot, ticks=30, exit_reason="completed")

        text = fake_fs.get_written_files()[str(DEFAULT_INDEX_PATH)]
        assert text.startswith(HEADER_LINE)
        rows = [
            decode_row(line)
            for line in text.splitlines()
            if line and not line.startswith("stamp\t")
        ]
        if len(rows) != 1:
            raise AssertionError(f"expected 1 index row, got {len(rows)}")
        row = rows[0]
        assert row["stamp"] == "20260620-150138"
        assert row["exit_reason"] == "completed"
        assert row["ticks"] == 30
        # 30 ticks * 2000ms / 1000 = 60 seconds (TICK_RATE_MS=2000).
        assert row["duration_s"] == 60


class TestPublishTickContext:
    """Tests for ``_publish_tick_context`` (Tier 3.2 event enrichment)."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_writes_tick_n_bot_state_and_action_kind(self, fake_env: FakeEnv) -> None:
        """The published context exposes the three structured fields.

        Asserts against ``runtime_logging.get_runtime_context`` so the
        test mirrors what every subsequent ``emit_*`` call will see.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.states import make_in_flight_action
        from tankpit_bot.bot.tick_loop import _publish_tick_context
        from tankpit_bot.runtime_logging import get_runtime_context

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
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import _publish_tick_context
        from tankpit_bot.runtime_logging import get_runtime_context

        bot = Bot("https://test.tankpit.com/", headless=True)
        _publish_tick_context(bot, tick_n=1)
        assert get_runtime_context()["tick_n"] == 1
        _publish_tick_context(bot, tick_n=2)
        assert get_runtime_context()["tick_n"] == 2
        _publish_tick_context(bot, tick_n=42)
        assert get_runtime_context()["tick_n"] == 42


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


class _BrowserClosedPage(_FakePage):
    """Page stub whose ``wait_for_timeout`` raises ``TargetClosedError``.

    Models the operator closing the browser while the bot is sleeping
    between ticks. The first tick runs to completion; the wait afterward
    is what fails. Lets the tick-loop graceful-shutdown path execute
    end-to-end without launching Playwright.
    """

    def wait_for_timeout(self, timeout: float) -> None:
        """Raise to model the browser being closed between ticks."""
        from playwright._impl._errors import TargetClosedError

        raise TargetClosedError("Page.wait_for_timeout: target closed")


class _TickRaisesBrowserClosedPage(_FakePage):
    """Page stub whose first ``set_content`` (used in tick_once) raises.

    Used to exercise the ``except TargetClosedError`` around
    ``_tick_once`` itself, the in-loop path. Modeled on a Playwright
    call failing because the browser shut mid-tick.
    """


def _fail_tick_once_with_browser_closed(bot: Bot) -> None:
    """Drop-in ``_tick_once`` that simulates the browser closing mid-tick."""
    _ = bot
    from playwright._impl._errors import TargetClosedError

    raise TargetClosedError("Page.goto: target closed mid-tick")


class TestBrowserClosedExit:
    """Browser closure mid-run records ``browser_closed`` and exits cleanly."""

    def setup_method(self) -> None:
        """Reset world state + interrupt flag before each test."""
        from tankpit_bot.bot.tick_loop import reset_interrupt_flag

        reset_world_state()
        reset_interrupt_flag()

    def teardown_method(self) -> None:
        """Reset world state + interrupt flag after each test."""
        from tankpit_bot.bot.tick_loop import reset_interrupt_flag

        reset_world_state()
        reset_interrupt_flag()

    def test_browser_closed_between_ticks_records_browser_closed(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """Closing the browser between ticks exits with ``browser_closed``."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import run_tick_loop
        from tankpit_bot.diagnostics.runs_index import DEFAULT_INDEX_PATH, decode_row
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        configure_bot_runtime_logging("20260620-191000")
        bot = Bot("https://test.tankpit.com/", headless=True)

        run_tick_loop(
            bot,
            _BrowserClosedPage(),
            session_seconds=0,
            stop_file_path=Path("C:/tmp/never_exists.sentinel"),
        )

        text = fake_fs.get_written_files()[str(DEFAULT_INDEX_PATH)]
        data_lines = [line for line in text.splitlines() if line and not line.startswith("stamp\t")]
        if len(data_lines) != 1:
            raise AssertionError(f"expected 1 index row, got {len(data_lines)}")
        row = decode_row(data_lines[0])
        assert row["exit_reason"] == "browser_closed"

    def test_browser_closed_during_tick_records_browser_closed(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """A ``TargetClosedError`` raised from inside ``_tick_once`` exits cleanly."""
        from tankpit_bot.bot import tick_loop as tick_loop_module
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import run_tick_loop
        from tankpit_bot.diagnostics.runs_index import DEFAULT_INDEX_PATH, decode_row
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        configure_bot_runtime_logging("20260620-191000")
        bot = Bot("https://test.tankpit.com/", headless=True)

        saved_tick_once = tick_loop_module._tick_once
        tick_loop_module._tick_once = _fail_tick_once_with_browser_closed
        run_tick_loop(
            bot,
            _FakePage(),
            session_seconds=0,
            stop_file_path=Path("C:/tmp/never_exists.sentinel"),
        )
        tick_loop_module._tick_once = saved_tick_once

        text = fake_fs.get_written_files()[str(DEFAULT_INDEX_PATH)]
        data_lines = [line for line in text.splitlines() if line and not line.startswith("stamp\t")]
        if len(data_lines) != 1:
            raise AssertionError(f"expected 1 index row, got {len(data_lines)}")
        row = decode_row(data_lines[0])
        assert row["exit_reason"] == "browser_closed"


class TestClearCommandError:
    """The Supervisor (0x52) error code clears every in-flight action kind."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def _make_pending_action(
        self,
        kind: ActionKind,
        *,
        target_x: int = 100,
        target_y: int = 100,
    ) -> InFlightActionDict:
        """Build a pending in-flight action of the requested kind."""
        return InFlightActionDict(
            kind=kind,
            target_x=target_x,
            target_y=target_y,
            started_ms=get_current_time_ms(),
            outcome="pending",
        )

    def test_command_error_clears_collect_action(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``Empty container`` aborts a pending collect in < 1 s.

        Without the hook the bot waited the full
        ``action_stall_timeout_ms`` (10 s) on every server denial; live
        run 20260620-184223 wasted 40 s of session time on four such
        rejections.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.sniffer.world_state import get_world_service

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("collect", target_x=150, target_y=150)

        get_world_service().last_command_error = 4  # "Empty container"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"
        assert get_world_service().last_command_error == -1

    def test_command_error_clears_collect_on_inventory_full(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``Inventory full`` (code 7) aborts a pending pickup in < 1 s.

        Empirical guard: live capture 20260620-190728 / 20260620-190830
        delivered ``error_code=7`` over the wire after pickup dispatches
        at full inventory (see ``runs/sniff/latest.events.jsonl``); the
        ``[GAME:EQUIPMENT] Inventory full`` log lines in the same capture
        cross-confirm the reject. Without code 7 in
        ``_ACTION_BLOCKING_COMMAND_ERRORS`` the collect would idle the
        full ``action_stall_timeout_ms`` (10 s) before replanning. The
        container's ``failed_pickups`` counter is bumped on the same
        path so downstream blacklisting kicks in.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.sniffer.world_state import get_world_service
        from tankpit_bot.state.types import WorldStateDict, make_container_state

        update_world_state_from_position(100, 100)
        ws = get_world_service()
        ws.world_state = WorldStateDict(
            **{
                **ws.world_state,
                "containers": {
                    "150,150": make_container_state(
                        x=150,
                        y=150,
                        is_fuel=False,
                        volume=0,
                        timestamp_ms=get_current_time_ms(),
                        failed_pickups=0,
                    )
                },
            }
        )
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("collect", target_x=150, target_y=150)

        ws.last_command_error = 7  # "Inventory full"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"
        assert ws.last_command_error == -1
        container = ws.world_state["containers"]["150,150"]
        assert container["failed_pickups"] == 1

    def test_command_error_clears_teleport_action(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``You can't go there!`` aborts a pending teleport in < 1 s."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.sniffer.world_state import get_world_service

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "TELEPORTING"
        action = self._make_pending_action("teleport", target_x=200, target_y=200)

        get_world_service().last_command_error = 1  # "You can't go there!"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"

    def test_command_error_clears_scan_action(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``Insufficient fuel`` aborts a pending radar scan in < 1 s."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_scan_action
        from tankpit_bot.sniffer.world_state import get_world_service

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "SCANNING"
        action = self._make_pending_action("scan")

        get_world_service().last_command_error = 8  # "Insufficient fuel"
        result = _wait_for_scan_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"

    def test_command_error_clears_map_open_action(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``You can't do this`` aborts a pending map_open in < 1 s."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_map_open_action
        from tankpit_bot.sniffer.world_state import get_world_service

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        # Map-open dispatch fires from IDLE; IDLE -> IDLE is a valid
        # transition so the rejection path replans without leaving the
        # ready state.
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        action = self._make_pending_action("map_open")

        get_world_service().last_command_error = 0  # "You can't do this"
        result = _wait_for_map_open_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"

    def test_no_command_error_lets_wait_continue(self, fake_env: FakeEnv) -> None:
        """No 0x52 error pending -> normal wait machinery runs."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("move", target_x=150, target_y=150)

        # No error code set; default -1 means no rejection pending.
        result = _wait_for_movement_action(bot, action)

        # The action is still in-flight (not rejected, not stalled, not
        # blocked) so wait returns True to continue waiting.
        assert result is True
