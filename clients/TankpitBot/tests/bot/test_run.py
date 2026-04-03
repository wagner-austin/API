"""Tests for Bot run method and game loop."""

from __future__ import annotations

import pytest

from tests.conftest import FakeEnv, FakeFileSystem


class TestBotGameLoop:
    """Tests for Bot._game_loop method."""

    def test_game_loop_exits_on_keyboard_interrupt(self, fake_env: FakeEnv) -> None:
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
            bot._game_loop(interrupting_page)


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
                bot.run()
        finally:
            _test_hooks.sync_playwright = original

    def test_run_success_path(self, fake_env: FakeEnv) -> None:
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

        try:
            bot = Bot("https://test.tankpit.com/", headless=True)
            # run() catches KeyboardInterrupt internally and returns normally
            bot.run()
            # After cleanup, _cdp and _page should be None
            assert bot._cdp is None
            assert bot._page is None
        finally:
            _test_hooks.sync_playwright = original


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
        _test_hooks.sync_playwright = fake_sync_playwright_factory

        try:
            from tankpit_bot.bot import base

            with pytest.raises(KeyboardInterrupt):
                base.main()
        finally:
            _test_hooks.sync_playwright = original_pw

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

        # Set sync_playwright to None so main() will call get_sync_playwright()
        _test_hooks.sync_playwright = None

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
            from tankpit_bot.bot import base

            with pytest.raises(KeyboardInterrupt):
                base.main()
        finally:
            _test_hooks.sync_playwright = original_pw
            _test_hooks.get_sync_playwright = original_get_pw

        if not get_called:
            raise AssertionError("Expected get_sync_playwright to be called")
        written_files = fake_fs.get_written_files()
        if "runs\\bot\\latest.log" not in written_files:
            raise AssertionError("Expected bot runtime latest log artifact")


class TestBotGameLoopStates:
    """Tests for Bot._game_loop AI-driven state handling."""

    def test_game_loop_ai_tick_no_self_state(self, fake_env: FakeEnv) -> None:
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
            bot._game_loop(interrupting_page)

        # AI state unchanged — no self_state to act on
        assert bot._ai_state["active_mode"] == "HUNT"
        assert bot._ai_state["combat_phase"] == "none"

    def test_game_loop_ai_tick_reaches_ready_state_and_starts_equipment_search(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Game loop reaches IDLE and starts equipment-search teleport flow.

        Args:
            fake_env: Installed fake environment fixture.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession, FakePageInterrupting

        reset_world_state()
        update_world_state_from_position(100, 100)
        update_world_state_from_fuel_total(800)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._magic = "test_magic"

        interrupting_page = FakePageInterrupting(interrupt_after=3)

        with pytest.raises(KeyboardInterrupt):
            bot._game_loop(interrupting_page)

        runtime_calls = [m for m in fake_cdp._sent_methods if m == "Runtime.evaluate"]
        assert runtime_calls == ["Runtime.evaluate", "Runtime.evaluate"]
        assert bot._ai_state["patrol_waypoint_index"] == 1
        assert bot.get_state() == "TELEPORTING"
