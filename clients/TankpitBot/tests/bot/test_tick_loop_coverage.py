"""Coverage tests for bot/tick_loop.py: stop-file detection and _clear_rejected_movement."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import KeyboardProtocol, ResponseProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.bot.ai.types import AIStateDict
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
from tankpit_bot.state.types import make_self_state
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


class TestWindDown:
    """Tests for the session wind-down flag."""

    def test_bounded_session_raises_wind_down_flag_in_final_stretch(self) -> None:
        """A >2-window session flips ``wind_down`` 60 s before the budget.

        User request 2026-07-26: "run and then collect and exit
        cleanly, instead of the program killing it mid action."
        """
        from tankpit_bot.bot.tick_loop import run_tick_loop

        bot = Bot("https://test.tankpit.com/", headless=True)
        run_tick_loop(
            bot,
            _FakePage(),
            session_seconds=126,
            stop_file_path=Path("C:/tmp/absent.sentinel"),
        )
        assert bot._ai_state["wind_down"] is True

    def test_kill_target_triggers_wind_down(self) -> None:
        """Reaching the kill bound flips ``wind_down`` at the kill boundary.

        User request 2026-07-26: "maybe if we put it for kills instead
        of time based" — the kill boundary is the natural clean-exit
        point (no fight is ever interrupted).
        """
        from tankpit_bot.bot.tick_loop import run_tick_loop

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["session_kill_count"] = 2
        run_tick_loop(
            bot,
            _FakePage(),
            session_seconds=4,
            session_kills=2,
            stop_file_path=Path("C:/tmp/absent.sentinel"),
        )
        assert bot._ai_state["wind_down"] is True

    def test_short_diagnostic_session_never_winds_down(self) -> None:
        """Sessions of two windows or less keep the full loop active."""
        from tankpit_bot.bot.tick_loop import run_tick_loop

        bot = Bot("https://test.tankpit.com/", headless=True)
        run_tick_loop(
            bot,
            _FakePage(),
            session_seconds=120,
            stop_file_path=Path("C:/tmp/absent.sentinel"),
        )
        assert bot._ai_state["wind_down"] is False


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
        from tankpit_bot.ledger.decision import record_decision
        from tankpit_bot.ledger.outcome.map_open import emit_map_open_data_processed
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        configure_bot_runtime_logging("20260620-150138")
        bot = Bot("https://test.tankpit.com/", headless=True)

        # One resolved decision (outcome-counts line) plus one still
        # pending at shutdown (unresolved-decisions line).
        record_decision(
            action_kind="map_open",
            cmd_type="map_open",
            mode="HUNT",
            score=800,
            reason_kind="find_enemies",
            reason_context={},
            target_x=0,
            target_y=0,
            target_id=0,
        )
        emit_map_open_data_processed(duration_ms=500)
        record_decision(
            action_kind="scan",
            cmd_type="radar",
            mode="HUNT",
            score=700,
            reason_kind="scan_on_landing",
            reason_context={},
            target_x=1,
            target_y=2,
            target_id=0,
        )

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


def _fail_tick_once_with_session_exit(bot: Bot) -> None:
    """Drop-in ``_tick_once`` that simulates a decision-owner exit request."""
    _ = bot
    from tankpit_bot.bot.session_exit import SessionExitError

    raise SessionExitError("no_viable_targets", "fresh map snapshot has no affordable enemy")


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

    def test_browser_closed_during_live_view_sync_records_browser_closed(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """A ``TargetClosedError`` from the caster toggle exits cleanly.

        The demand sync runs inside ``_tick_once``'s guard; a viewer
        subscribing in the same instant the operator closes the
        browser makes the caster's ``Runtime.evaluate`` the first
        call to observe the dead target.
        """
        from collections.abc import Callable

        from platform_core.json_utils import JSONObject
        from playwright._impl._errors import TargetClosedError

        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import run_tick_loop
        from tankpit_bot.diagnostics.runs_index import DEFAULT_INDEX_PATH, decode_row
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging
        from tankpit_bot.service.frame_bus import FrameBus

        class _ClosedTargetCDP:
            def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
                _ = params
                if method == "Runtime.evaluate":
                    raise TargetClosedError(f"{method}: target closed")
                return {}

            def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
                _ = (event, handler)

            def detach(self) -> None:
                raise AssertionError("never detached in this test")

        configure_bot_runtime_logging("20260728-120000")
        frames = FrameBus()
        bot = Bot("https://test.tankpit.com/", headless=True, frame_bus=frames)
        bot._cdp = _ClosedTargetCDP()
        frames.subscribe()  # viewer demand → the sync attempts a start

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
        assert row["exit_reason"] == "browser_closed"

    def test_session_exit_request_records_its_reason(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """A ``SessionExitError`` from a decision owner ends the run with its reason.

        User contract (2026-07-02): when the bot cannot do its job
        (no viable targets, out of fuel) it exits cleanly with an
        analyzable ``exit_reason`` instead of crashing or looping.
        """
        from tankpit_bot.bot import tick_loop as tick_loop_module
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import run_tick_loop
        from tankpit_bot.diagnostics.runs_index import DEFAULT_INDEX_PATH, decode_row
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        configure_bot_runtime_logging("20260702-101500")
        bot = Bot("https://test.tankpit.com/", headless=True)

        saved_tick_once = tick_loop_module._tick_once
        tick_loop_module._tick_once = _fail_tick_once_with_session_exit
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
        assert row["exit_reason"] == "no_viable_targets"


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
        """A 0x52 ``You can't do this`` (code 0) aborts a pending collect in < 1 s.

        Without the hook the bot waited the full
        ``action_stall_timeout_ms`` (10 s) on every server denial; live
        run 20260620-184223 wasted 40 s of session time on four such
        rejections. Illegal geometry blacklists the container position
        via ``failed_pickups`` (unlike code 4, which removes the
        belief outright).
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.sniffer.world_state import get_world_service

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("collect", target_x=150, target_y=150)

        get_world_service().last_command_error = 0  # "You can't do this"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"
        assert get_world_service().last_command_error == -1

    def test_cant_go_on_collect_records_a_movement_rejection(self, fake_env: FakeEnv) -> None:
        """A cant_go rejecting a walk-pickup lands in the movement record.

        Run bot-20260730-110x ticks 95-107: twelve consecutive
        rejected walk-pickups under fire were invisible to the
        per-tile move marks because collect rejections only fed
        ``failed_pickups`` — the escape's movement-dead detector
        needs the shared "the server refused a move" fact regardless
        of the command kind.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.sniffer.world_state import (
            get_world_service,
            recent_movement_rejections,
        )

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("collect", target_x=150, target_y=150)

        get_world_service().last_command_error = 1  # "You can't go there!"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert recent_movement_rejections(get_current_time_ms(), 10000) == 1

    def test_non_movement_rejection_is_not_recorded(self, fake_env: FakeEnv) -> None:
        """A code-0 collect rejection is not a movement refusal."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.sniffer.world_state import (
            get_world_service,
            recent_movement_rejections,
        )

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("collect", target_x=150, target_y=150)

        get_world_service().last_command_error = 0  # "You can't do this"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert recent_movement_rejections(get_current_time_ms(), 10000) == 0

    def test_command_error_clears_collect_on_inventory_full(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``Inventory full`` (code 7) aborts the pickup, keeps the container.

        Empirical guard: live capture 20260620-190728 / 20260620-190830
        delivered ``error_code=7`` over the wire after pickup dispatches
        at full inventory. Without code 7 in the blocking set the
        collect would idle the full ``action_stall_timeout_ms`` (10 s)
        before replanning. User mechanic (2026-07-18): containers fill
        whatever is empty and code 7 fires only at all-slots-full --
        the container is NOT blacklisted (it is fine; the tank is
        full) and every slot belief reconciles up to capacity, the
        rejection being an authoritative absolute inventory statement.
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
        assert container["failed_pickups"] == 0
        # No self_state rank in this fixture-free world? position update
        # created one at rank 0 -> capacity applies; all slots snapped up.
        from tankpit_bot.physics.capacity import inventory_capacity

        rank = ws.world_state["self_state"]["rank"] if ws.world_state["self_state"] else 0
        cap = inventory_capacity(rank)
        inv = ws.inventory_state
        assert inv["armor_shields"]["count"] >= cap
        assert inv["dual_shots"]["count"] >= cap
        assert inv["missile_shots"]["count"] >= cap
        assert inv["homing_shots"]["count"] >= cap
        assert inv["extra_radars"]["count"] >= cap
        from tankpit_bot.ledger.ring import outcome_counts

        assert outcome_counts("collect") == {"inventory_full": 1}

    def test_command_error_tank_full_does_not_mark_failed_pickup(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``Tank full`` (code 5) clears the action WITHOUT blacklisting.

        Bug 0.3 (2026-07-06): a code=5 rejection means the container
        was not empty -- the server refused the transfer because the
        tank could not accept it. Under Bug 0.2's pre-dispatch gate (now ``_pickup_not_worth_walk``)
        pre-dispatch gate the overflow scenario cannot occur in the
        normal flow, so a surviving code=5 is a race between
        planner-time and dispatch-time fuel state. Blacklisting a
        still-full container is wrong -- next tick with headroom will
        successfully consume it. The in-flight action is still
        cleared (the planner replans this tick) but ``failed_pickups``
        stays at 0 so the container remains a candidate. Pre-fix
        behavior: the 22:37 fuel-loop's four consecutive
        partial-transfer + code=5 events blacklisted four still-full
        fuel containers.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.sniffer.world_state import get_world_service
        from tankpit_bot.state.types import (
            WorldStateDict,
            make_container_state,
        )

        update_world_state_from_position(100, 100)
        ws = get_world_service()
        ws.world_state = WorldStateDict(
            **{
                **ws.world_state,
                "self_state": make_self_state(
                    tank_id=1,
                    x=100,
                    y=100,
                    team=1,
                    rank=0,
                    fuel=1000,
                    leaderboard_position=0,
                ),
                "containers": {
                    "150,150": make_container_state(
                        x=150,
                        y=150,
                        is_fuel=True,
                        volume=400,
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

        ws.last_command_error = 5  # "Tank full"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"
        assert ws.last_command_error == -1
        container = ws.world_state["containers"]["150,150"]
        assert container["failed_pickups"] == 0
        from tankpit_bot.ledger.ring import outcome_counts

        assert outcome_counts("collect") == {"clamped_transfer": 1}

    def test_command_error_empty_container_removes_belief(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``Empty container`` (code 4) deletes the container belief.

        The server says the container is drained, so the volume the
        planner acted on is contradicted -- the belief is removed
        outright rather than blacklisted. (Until 2026-07-19 this
        removal was done by the DOM game-log "Empty container"
        consumer one or two ticks later; the wire code is the same
        signal, earlier, and the DOM channel is now witness-only.)
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.sniffer.world_state import get_world_service
        from tankpit_bot.state.types import (
            WorldStateDict,
            make_container_state,
        )

        update_world_state_from_position(100, 100)
        ws = get_world_service()
        ws.world_state = WorldStateDict(
            **{
                **ws.world_state,
                "self_state": make_self_state(
                    tank_id=1,
                    x=100,
                    y=100,
                    team=1,
                    rank=0,
                    fuel=1000,
                    leaderboard_position=0,
                ),
                "containers": {
                    "150,150": make_container_state(
                        x=150,
                        y=150,
                        is_fuel=True,
                        volume=400,
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

        ws.last_command_error = 4  # "Empty container"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"
        assert ws.last_command_error == -1
        assert ws.world_state["containers"] == {}
        # The disproof also marks the container memory desynced so the
        # collect cascade radars before pursuing further remembered
        # stock (user ruling 2026-07-30: one stale item = one radar).
        assert ws.container_desync_ms > 0
        from tankpit_bot.ledger.ring import outcome_counts

        assert outcome_counts("collect") == {"pickup_empty": 1}

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

    def test_scan_wait_drops_orphan_error_and_stays_pending(self, fake_env: FakeEnv) -> None:
        """A 0x52 code arriving during a scan wait is an orphan and is dropped.

        Radar dispatch (``CMD_RADAR`` 0x66, client ``Mb``) is not
        server-side rejectable: the server accepts every scan and
        replies with a ``0x4F`` result. Any 0x52 that lands during the
        scan wait belongs to a PRIOR action (typically one that already
        completed via a different wire signal like
        ``container_consumed``). The wait discards the orphan code and
        stays pending so the scan can complete normally.
        """
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

        assert result is True
        assert bot.get_state() == "SCANNING"
        assert get_world_service().last_command_error == -1

    def test_map_open_wait_drops_orphan_error_and_stays_pending(self, fake_env: FakeEnv) -> None:
        """A 0x52 code arriving during a map_open wait is an orphan and is dropped.

        Regression guard for live run 2026-07-06 20:20:59: a late-
        arriving ``code=4`` from a collect that already completed via
        ``container_consumed`` was misattributed to the following
        ``map_open``. HUNT could not acquire, session exited
        ``no_viable_targets`` at fuel 531 with a fully-stocked tank.
        Map_open dispatch (``CMD_MAP_OPEN`` 0x6C, client ``Nb``) is
        server-side unconditional, so no 0x52 code is ever a legitimate
        map_open rejection.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_map_open_action
        from tankpit_bot.sniffer.world_state import get_world_service

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        action = self._make_pending_action("map_open")

        get_world_service().last_command_error = 4  # "Empty container"
        result = _wait_for_map_open_action(bot, action)

        assert result is True
        assert bot.get_state() == "IDLE"
        assert get_world_service().last_command_error == -1

    def test_teleport_wait_drops_orphan_empty_container(self, fake_env: FakeEnv) -> None:
        """A code=4 during a teleport wait is an orphan; teleport stays pending.

        Teleport (``CMD_MAP_TELEPORT`` 0x74) can draw codes 0/1/8; an
        ``Empty container`` (4) can only originate from a pickup and so
        must belong to a prior collect.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.sniffer.world_state import get_world_service

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "TELEPORTING"
        action = self._make_pending_action("teleport", target_x=200, target_y=200)

        get_world_service().last_command_error = 4  # "Empty container"
        result = _wait_for_movement_action(bot, action)

        assert result is True
        assert bot.get_state() == "TELEPORTING"
        assert get_world_service().last_command_error == -1

    def test_move_wait_drops_orphan_tank_full(self, fake_env: FakeEnv) -> None:
        """A code=5 (tank full) during a move wait is orphaned.

        Move (``CMD_MOVE`` 0x70) can draw codes 0/1/8; ``Tank full`` (5)
        can only originate from a fuel pickup.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.sniffer.world_state import get_world_service

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("move", target_x=150, target_y=150)

        get_world_service().last_command_error = 5  # "Tank full"
        result = _wait_for_movement_action(bot, action)

        assert result is True
        assert bot.get_state() == "MOVING"
        assert get_world_service().last_command_error == -1

    def test_orphan_command_error_emits_diagnostic(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """The orphan-drop path emits an ``orphan_command_error`` diagnostic.

        Observability guard: without the diagnostic, a wire race that
        drops an orphan code is invisible in the events stream. This
        test drives the map_open orphan path and asserts a single
        diagnostic with the action_kind and error_code fields.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_map_open_action
        from tankpit_bot.diagnostics.event_stream import load_event_records
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging
        from tankpit_bot.sniffer.world_state import get_world_service

        artifacts = configure_bot_runtime_logging("20260706-202100")
        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        action = self._make_pending_action("map_open")

        get_world_service().last_command_error = 4  # "Empty container"
        _wait_for_map_open_action(bot, action)

        records = [
            record
            for record in load_event_records(Path(artifacts["latest_events_path"]))
            if record["fields"].get("diagnostic_kind") == "orphan_command_error"
        ]
        assert len(records) == 1
        assert records[0]["fields"] == {
            "diagnostic_kind": "orphan_command_error",
            "action_kind": "map_open",
            "error_code": 4,
        }

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

    def test_scan_wait_with_no_error_stays_pending(self, fake_env: FakeEnv) -> None:
        """The scan drain path is a no-op when no 0x52 code is pending."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_scan_action
        from tankpit_bot.sniffer.world_state import get_world_service

        update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "SCANNING"
        action = self._make_pending_action("scan")

        assert get_world_service().last_command_error == -1
        result = _wait_for_scan_action(bot, action)

        assert result is True
        assert bot.get_state() == "SCANNING"

    def test_scan_and_map_open_whitelists_are_empty(self) -> None:
        """Whitelist invariant: scan and map_open are never rejected by any 0x52 code.

        Radar (``CMD_RADAR`` 0x66) and map_open (``CMD_MAP_OPEN`` 0x6C)
        are server-side unconditional. If a future change adds a code
        to either whitelist,
        :func:`~tankpit_bot.bot.tick_loop_actions._wait_for_scan_action`
        and :func:`~tankpit_bot.bot.tick_loop_actions._wait_for_map_open_action`
        must be updated to check the applicable-rejection outcome and
        transition the action -- currently they only call
        :func:`~tankpit_bot.bot.tick_loop_actions._drain_orphan_command_error`
        which never transitions.
        """
        from tankpit_bot.bot.tick_loop_actions import _COMMAND_ERROR_APPLICABILITY

        assert _COMMAND_ERROR_APPLICABILITY["scan"] == frozenset()
        assert _COMMAND_ERROR_APPLICABILITY["map_open"] == frozenset()
        assert _COMMAND_ERROR_APPLICABILITY["none"] == frozenset()
        assert _COMMAND_ERROR_APPLICABILITY["shoot"] == frozenset()


class TestEarlyWakeSleep:
    """Tests for the early-wake between-tick sleep."""

    def test_idle_wait_sleeps_the_full_window(self) -> None:
        """With no in-flight action the sleep is one uninterrupted window."""
        from tankpit_bot.bot.tick_loop import _wait_between_ticks
        from tankpit_bot.protocol.commands import TICK_RATE_MS

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._cdp_message_buffer = []
        waits: list[float] = []

        class _RecordingPage(_FakePage):
            def wait_for_timeout(self, timeout: float) -> None:
                waits.append(timeout)

        waited = _wait_between_ticks(bot, _RecordingPage())

        assert waited == TICK_RATE_MS
        assert waits == [float(TICK_RATE_MS)]

    def test_in_flight_wait_wakes_on_fresh_wire_traffic(self) -> None:
        """Fresh CDP traffic during an in-flight action ends the sleep early.

        The Artax-era fixed sleep waited out the whole 2 s window after
        a completion message arrived, drifting one server tick behind a
        human who acts the moment the tank arrives (user observation
        2026-07-30: "you can click as soon as it reaches it's
        destination and it'll go instantly to the next action").
        """
        from tankpit_bot.bot.states import make_in_flight_action
        from tankpit_bot.bot.tick_loop import _WAKE_SLICE_MS, _wait_between_ticks
        from tankpit_bot.browser import get_current_time_ms

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._cdp_message_buffer = []
        bot._state_data["in_flight_action"] = make_in_flight_action(
            "scan", 0, 0, get_current_time_ms()
        )
        waits: list[float] = []

        class _TrafficPage(_FakePage):
            def wait_for_timeout(self, timeout: float) -> None:
                waits.append(timeout)
                bot._cdp_message_buffer.append("traffic")

        waited = _wait_between_ticks(bot, _TrafficPage())

        assert waited == _WAKE_SLICE_MS
        assert waits == [float(_WAKE_SLICE_MS)]

    def test_in_flight_wait_without_traffic_runs_the_window(self) -> None:
        """A quiet wire keeps the in-flight sleep at the full window."""
        from tankpit_bot.bot.states import make_in_flight_action
        from tankpit_bot.bot.tick_loop import _WAKE_SLICE_MS, _wait_between_ticks
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.protocol.commands import TICK_RATE_MS

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._cdp_message_buffer = []
        bot._state_data["in_flight_action"] = make_in_flight_action(
            "scan", 0, 0, get_current_time_ms()
        )
        waits: list[float] = []

        class _QuietPage(_FakePage):
            def wait_for_timeout(self, timeout: float) -> None:
                waits.append(timeout)

        waited = _wait_between_ticks(bot, _QuietPage())

        assert waited == TICK_RATE_MS
        assert waits == [float(_WAKE_SLICE_MS)] * (TICK_RATE_MS // _WAKE_SLICE_MS)


class TestWireSilenceWatchdog:
    """Tests for the connection-lost wire-silence watchdog."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_disarmed_before_first_game_message(self) -> None:
        """A zero stamp (boot, lobby) never trips the watchdog."""
        from tankpit_bot.bot.tick_loop import _check_wire_silence

        bot = Bot("https://test.tankpit.com/", headless=True)
        assert get_world_service().last_game_message_ms == 0

        _check_wire_silence(bot)

    def test_fresh_traffic_passes(self) -> None:
        """A recent game message keeps the session alive."""
        from tankpit_bot.bot.tick_loop import _check_wire_silence

        bot = Bot("https://test.tankpit.com/", headless=True)
        get_world_service().last_game_message_ms = get_current_time_ms() - 1_000

        _check_wire_silence(bot)

    def test_silence_past_limit_raises_connection_lost(self) -> None:
        """Wire silence past the limit ends the session with a receipt.

        Session 3 of run 20260730: the game socket died at 11:58:32,
        the page auto-reconnected to the lobby (socket OPEN, so the
        ws-ready gate passed), and the bot injected map_open into a
        dead session for 43 minutes -- 243 consecutive stalls, zero
        inbound world messages. The watchdog turns that zombie into a
        90-second clean exit the harness can relaunch from.
        """
        from tankpit_bot.bot.session_exit import SessionExitError
        from tankpit_bot.bot.tick_loop import _WIRE_SILENCE_LIMIT_MS, _check_wire_silence

        bot = Bot("https://test.tankpit.com/", headless=True)
        get_world_service().last_game_message_ms = (
            get_current_time_ms() - _WIRE_SILENCE_LIMIT_MS - 1
        )

        with pytest.raises(SessionExitError) as exc_info:
            _check_wire_silence(bot)

        assert exc_info.value.reason == "connection_lost"
        assert "no game wire message" in exc_info.value.detail


class TestFriendlyFireDisproof:
    """Tests for consuming err=3 friendly_fire as target disproof."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_friendly_fire_blocks_target_and_releases_matching_lock(self) -> None:
        """One err=3 blocklists the id and clears the matching combat lock.

        Session 4 of run 20260730 (20:36): Yuppler left the game, the
        0x58 grace kept his registry entry, every map open re-stamped
        the ghost's freshness, and the bot fired 43 consecutive
        rejected shots. The disproof turns the first rejection into a
        block + lock release.
        """
        from tankpit_bot.bot.tick_loop import _disprove_target_by_friendly_fire

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state = AIStateDict(
            **{
                **bot._ai_state,
                "combat_target_id": 1229,
                "combat_target_x": 245,
                "combat_target_y": 76,
            }
        )

        _disprove_target_by_friendly_fire(bot, 1229, "Yuppler")

        assert "1229" in bot._ai_state["blocked_combat_targets"]
        assert bot._ai_state["combat_target_id"] == -1
        assert bot._ai_state["combat_target_x"] == 0

    def test_friendly_fire_keeps_unrelated_lock(self) -> None:
        """Disproving a non-locked target leaves the held lock alone."""
        from tankpit_bot.bot.tick_loop import _disprove_target_by_friendly_fire

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state = AIStateDict(
            **{
                **bot._ai_state,
                "combat_target_id": 514,
                "combat_target_x": 117,
                "combat_target_y": 139,
            }
        )

        _disprove_target_by_friendly_fire(bot, 1229, "Yuppler")

        assert "1229" in bot._ai_state["blocked_combat_targets"]
        assert bot._ai_state["combat_target_id"] == 514
