"""Tests for Bot run method and game loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot._test_hooks import AutoscrollEnforcerProtocol
from tests.conftest import FakeEnv, FakeFileSystem

_STOP = Path("__nonexistent_stop_file__")


class TestBotGameLoop:
    """Tests for Bot._game_loop method."""

    def test_game_loop_exits_on_keyboard_interrupt(
        self, fake_env: FakeEnv, fake_fs: FakeFileSystem
    ) -> None:
        """Test _game_loop exits cleanly on KeyboardInterrupt."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession, FakePageInterrupting

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp

        # Use the FakePageInterrupting from fakes.py that raises after 1 wait
        interrupting_page = FakePageInterrupting(interrupt_after=1)

        # _game_loop will exit when KeyboardInterrupt is raised
        with pytest.raises(KeyboardInterrupt):
            bot._game_loop(interrupting_page, session_seconds=0, stop_file_path=_STOP)

    def test_game_loop_returns_at_session_tick_budget(
        self, fake_env: FakeEnv, fake_fs: FakeFileSystem
    ) -> None:
        """A positive TANKPIT_BOT_SESSION_SECONDS ends the loop cleanly.

        With TICK_RATE_MS=2000, 4 seconds is exactly 2 ticks: the loop
        must return (not raise) after the second tick, before any
        inter-tick wait would run for the final tick.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.protocol.commands import TICK_RATE_MS
        from tests.fakes import FakeCDPSession, FakePageInterrupting

        fake_env.set("TANKPIT_BOT_SESSION_SECONDS", "4")
        assert 4 * 1000 // TICK_RATE_MS == 2

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp

        # Interrupt as a backstop: a correct budget exit never waits twice.
        page = FakePageInterrupting(interrupt_after=2)

        bot._game_loop(page, session_seconds=4, stop_file_path=_STOP)

        assert page._wait_count == 1

    def test_game_loop_invalid_session_seconds_raises(self, fake_env: FakeEnv) -> None:
        """A non-integer TANKPIT_BOT_SESSION_SECONDS propagates ValueError."""
        from tankpit_bot.bot.entry import resolve_session_seconds

        with pytest.raises(ValueError):
            resolve_session_seconds([], "soon")


def _stub_autoscroll_hook() -> tuple[AutoscrollEnforcerProtocol, list[int]]:
    """Replace the autoscroll enforcement hook with a call recorder.

    Returns:
        Tuple of (original hook, recorded call count list) for
        save-and-restore in the caller's ``finally``.
    """
    from tankpit_bot import _test_hooks
    from tankpit_bot._test_hooks import AutoscrollPageProtocol
    from tankpit_bot.types.message import CapturedMessage

    calls: list[int] = []
    original = _test_hooks.ensure_autoscroll_off

    def _recorder(page: AutoscrollPageProtocol, messages: list[CapturedMessage]) -> None:
        del page, messages
        calls.append(1)

    _test_hooks.ensure_autoscroll_off = _recorder
    return original, calls


class TestBotRunMethod:
    """Tests for Bot.run method."""

    def test_run_raises_without_playwright(self, fake_env: FakeEnv) -> None:
        """Test run() raises PlaywrightNotInstalledError when not available."""
        from tankpit_bot import _test_hooks
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.browser import PlaywrightNotInstalledError

        # Save original and set to None
        original = _test_hooks.sync_playwright
        _test_hooks.sync_playwright = None
        try:
            bot = Bot("https://test.tankpit.com/", headless=True)
            with pytest.raises(PlaywrightNotInstalledError):
                bot.run(session_seconds=0, stop_file_path=_STOP)
        finally:
            _test_hooks.sync_playwright = original

    def test_run_success_path(self, fake_env: FakeEnv, fake_fs: FakeFileSystem) -> None:
        """Test run() goes through the success path and handles KeyboardInterrupt.

        This covers lines 603-614 in run(). The run() method catches
        KeyboardInterrupt internally and returns normally after cleanup.
        """
        from tankpit_bot import _test_hooks
        from tankpit_bot.bot.base import Bot
        from tests.fakes import fake_sync_playwright_bot

        # Set up the fake Playwright that will raise KeyboardInterrupt in game loop
        original = _test_hooks.sync_playwright
        _test_hooks.sync_playwright = fake_sync_playwright_bot
        original_autoscroll, autoscroll_calls = _stub_autoscroll_hook()

        try:
            bot = Bot("https://test.tankpit.com/", headless=True)
            # run() catches KeyboardInterrupt internally and returns normally
            bot.run(session_seconds=0, stop_file_path=_STOP)
            # After cleanup, _cdp and _page should be None
            assert bot._cdp is None
            assert bot._page is None
            # The session-start enforcement fired exactly once.
            assert autoscroll_calls == [1]
        finally:
            _test_hooks.sync_playwright = original
            _test_hooks.ensure_autoscroll_off = original_autoscroll

    def test_run_maximises_via_cdp_on_streamed_display(
        self, fake_env: FakeEnv, fake_fs: FakeFileSystem
    ) -> None:
        """Bot.run() flips the window to maximised via CDP on the streamed display.

        When Vibeshine's launcher sets ``SUNSHINE_STREAM_DISPLAY_*``,
        ``_chrome_stream_no_viewport`` becomes ``True`` and the
        post-launch maximise branch in :meth:`Bot.run` fires. The fake
        CDP session returns a well-formed ``windowId`` so the run
        completes without raising.
        """
        from tankpit_bot import _test_hooks
        from tankpit_bot.bot.base import Bot
        from tests.fakes import fake_sync_playwright_bot

        original = _test_hooks.sync_playwright
        fake_env.set("SUNSHINE_STREAM_DISPLAY_X", "0")
        fake_env.set("SUNSHINE_STREAM_DISPLAY_Y", "0")
        fake_env.set("SUNSHINE_STREAM_DISPLAY_W", "1920")
        fake_env.set("SUNSHINE_STREAM_DISPLAY_H", "1080")
        _test_hooks.sync_playwright = fake_sync_playwright_bot
        original_autoscroll, _autoscroll_calls = _stub_autoscroll_hook()

        try:
            bot = Bot("https://test.tankpit.com/", headless=True)
            bot.run(session_seconds=0, stop_file_path=_STOP)
            assert bot._cdp is None
            assert bot._page is None
        finally:
            _test_hooks.sync_playwright = original
            _test_hooks.ensure_autoscroll_off = original_autoscroll

    def test_run_saves_capture_session(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """Test run() saves a capture session file when runtime logging is configured.

        Configures bot runtime logging so get_bot_runtime_artifacts() is non-None,
        then verifies the capture session is written to the canonical paths.
        """
        from tankpit_bot import _test_hooks
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging
        from tests.fakes import fake_sync_playwright_bot

        original = _test_hooks.sync_playwright
        _test_hooks.sync_playwright = fake_sync_playwright_bot
        original_autoscroll, _autoscroll_calls = _stub_autoscroll_hook()

        try:
            configure_bot_runtime_logging(stamp="20260404-000000")
            bot = Bot("https://test.tankpit.com/", headless=True)
            bot.run(session_seconds=0, stop_file_path=_STOP)
            written_files = fake_fs.get_written_files()
            has_capture = False
            for path in written_files:
                if "capture_session.json" in path:
                    has_capture = True
            assert has_capture
        finally:
            _test_hooks.sync_playwright = original
            _test_hooks.ensure_autoscroll_off = original_autoscroll

    def test_send_graceful_quit_uses_current_cdp(self, fake_env: FakeEnv) -> None:
        """Teardown quit binds the live CDP session and sends quit_game."""
        from tankpit_bot._test_hooks import CDPSessionProtocol
        from tankpit_bot.bot.base import Bot
        from tests.fakes.base import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        sent: list[tuple[str, bytes]] = []

        def _record(cdp: CDPSessionProtocol, data: bytes, label: str) -> str:
            sent.append((label, data))
            return ""

        bot._commands._send_ws_bytes = _record
        bot._cdp = FakeCDPSession()

        bot._send_graceful_quit()

        assert sent == [("quit_game", b"\x01\x00-")]

    def test_send_graceful_quit_absorbs_closed_browser(self, fake_env: FakeEnv) -> None:
        """A dead browser makes the courtesy quit a no-op, not a crash.

        Run bot-20260729-215151: the browser died at 19 kills, the
        scorecard wrote cleanly, and the teardown quit raised
        ``TargetClosedError`` through an otherwise-handled shutdown
        (exit code 2). The socket drop already told the server we
        left, so the send path absorbs the error with a log line.
        """
        from playwright._impl._errors import TargetClosedError

        from tankpit_bot._test_hooks import CDPSessionProtocol
        from tankpit_bot.bot.base import Bot
        from tests.fakes.base import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)

        def _raise_closed(cdp: CDPSessionProtocol, data: bytes, label: str) -> str:
            _ = (cdp, data, label)
            raise TargetClosedError("browser is gone")

        bot._commands._send_ws_bytes = _raise_closed
        bot._cdp = FakeCDPSession()

        bot._send_graceful_quit()

        assert bot._commands.cdp is bot._cdp

    def test_save_capture_session_returns_when_runtime_artifacts_missing(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """Saving a capture session is a no-op when runtime logging is disabled."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)

        bot._save_capture_session()

        assert all("capture_session.json" not in path for path in fake_fs.get_written_files())

    def test_record_game_log_witness_timestamps_entries(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Polled game-log entries land in the witness list with timestamps."""
        from tankpit_bot import _test_hooks
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.browser.dom_scraper import GameLogEntry
        from tankpit_bot.types import GameLogEntryWithTimestamp

        bot = Bot("https://test.tankpit.com/", headless=True)
        original_clock = _test_hooks.get_current_time_ms
        _test_hooks.get_current_time_ms = lambda: 1234567
        try:
            bot._record_game_log_witness(
                [
                    GameLogEntry(text="purple-8 has been deactivated by you", category="combat"),
                    GameLogEntry(text="Empty container", category="other"),
                ]
            )
            bot._record_game_log_witness([])
        finally:
            _test_hooks.get_current_time_ms = original_clock

        assert bot._game_log_witness == [
            GameLogEntryWithTimestamp(
                timestamp_ms=1234567,
                text="purple-8 has been deactivated by you",
                category="combat",
            ),
            GameLogEntryWithTimestamp(
                timestamp_ms=1234567,
                text="Empty container",
                category="other",
            ),
        ]


class TestBotBaseMain:
    """Tests for bot.base.main function."""

    def test_main_creates_and_runs_bot(self, fake_env: FakeEnv, fake_fs: FakeFileSystem) -> None:
        """Test main() creates Bot and calls run()."""
        from tankpit_bot import _test_hooks
        from tankpit_bot._test_hooks import SyncPlaywrightContextManagerProtocol
        from tests.fakes import FakeSyncPlaywrightContextManagerBot

        # Track if sync_playwright factory was called
        factory_called = False

        def fake_sync_playwright_factory() -> SyncPlaywrightContextManagerProtocol:
            """Return fake sync_playwright that exits via KeyboardInterrupt."""
            nonlocal factory_called
            factory_called = True
            return FakeSyncPlaywrightContextManagerBot(interrupt_after=2)

        # Set up fakes
        original_pw = _test_hooks.sync_playwright
        original_argv = _test_hooks.get_argv
        _test_hooks.sync_playwright = fake_sync_playwright_factory
        _test_hooks.get_argv = lambda: ["tankpit-bot"]

        try:
            from tankpit_bot.bot import entry

            with pytest.raises(KeyboardInterrupt):
                entry.main()
        finally:
            _test_hooks.sync_playwright = original_pw
            _test_hooks.get_argv = original_argv

        if not factory_called:
            raise AssertionError("Expected sync_playwright factory to be called")
        written_files = fake_fs.get_written_files()
        if "runs\\bot\\latest.log" not in written_files:
            raise AssertionError("Expected bot runtime latest log artifact")
        if "runs\\bot\\latest.events.jsonl" not in written_files:
            raise AssertionError("Expected bot runtime latest events artifact")

    def test_main_sets_sync_playwright_when_none(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """Test main() sets sync_playwright when it is None.

        This covers line 672 where sync_playwright is set from get_sync_playwright().
        """
        from tankpit_bot import _test_hooks
        from tankpit_bot._test_hooks import SyncPlaywrightFactoryProtocol
        from tests.fakes import FakeSyncPlaywrightContextManagerBot

        # Save originals
        original_pw = _test_hooks.sync_playwright
        original_get_pw = _test_hooks.get_sync_playwright
        original_argv = _test_hooks.get_argv

        # Set sync_playwright to None so main() will call get_sync_playwright()
        _test_hooks.sync_playwright = None
        _test_hooks.get_argv = lambda: ["tankpit-bot"]

        # Track if get_sync_playwright was called
        get_called = False

        def fake_get_sync_playwright() -> SyncPlaywrightFactoryProtocol:
            """Fake get_sync_playwright that returns our test factory."""
            nonlocal get_called
            get_called = True

            def factory() -> FakeSyncPlaywrightContextManagerBot:
                return FakeSyncPlaywrightContextManagerBot(interrupt_after=2)

            return factory

        _test_hooks.get_sync_playwright = fake_get_sync_playwright

        try:
            from tankpit_bot.bot import entry

            with pytest.raises(KeyboardInterrupt):
                entry.main()
        finally:
            _test_hooks.sync_playwright = original_pw
            _test_hooks.get_sync_playwright = original_get_pw
            _test_hooks.get_argv = original_argv

        if not get_called:
            raise AssertionError("Expected get_sync_playwright to be called")
        written_files = fake_fs.get_written_files()
        if "runs\\bot\\latest.log" not in written_files:
            raise AssertionError("Expected bot runtime latest log artifact")


class TestBotGameLoopStates:
    """Tests for Bot._game_loop AI-driven state handling."""

    def test_game_loop_ai_tick_no_self_state(
        self, fake_env: FakeEnv, fake_fs: FakeFileSystem
    ) -> None:
        """Game loop returns early when tick-loop state has no self tank."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state
        from tests.fakes import FakeCDPSession, FakePageInterrupting

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp

        interrupting_page = FakePageInterrupting(interrupt_after=3)

        with pytest.raises(KeyboardInterrupt):
            bot._game_loop(interrupting_page, session_seconds=0, stop_file_path=_STOP)

        # AI state unchanged — no self_state to act on
        assert bot._ai_state["mode"] == "UNSET"
        assert bot._ai_state["mode_state"] == ""

    def test_game_loop_ai_tick_reaches_ready_state_and_starts_equipment_search(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """Game loop reaches IDLE and starts equipment-search radar flow.

        Args:
            fake_env: Installed fake environment fixture.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.sniffer.world_state import (
            get_world_service,
            reset_world_state,
            update_world_state_from_position,
        )
        from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total
        from tests.fakes import FakeCDPSession, FakePageInterrupting

        reset_world_state()
        update_world_state_from_position(100, 100)
        update_world_state_from_fuel_total(get_world_service(), 800)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._magic = "test_magic"

        interrupting_page = FakePageInterrupting(interrupt_after=3)

        with pytest.raises(KeyboardInterrupt):
            bot._game_loop(interrupting_page, session_seconds=0, stop_file_path=_STOP)

        runtime_calls = [m for m in fake_cdp._sent_methods if m == "Runtime.evaluate"]
        # CDP calls: snapshot read + structure survey + radar dispatch + overlay update.
        assert runtime_calls == ["Runtime.evaluate"] * 4
        assert bot._ai_state["last_scan_ms"] > 0
        assert bot.get_state() == "SCANNING"
