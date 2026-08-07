"""Coverage tests for bot/base.py: _dispatch_keypress, maybe_capture_account_stats_once,
and resolve_session_seconds."""

from __future__ import annotations

import pytest

from tests.conftest import FakeEnv, FakeFileSystem


class TestDispatchKeypressWithoutCDP:
    """Test _dispatch_keypress raises RuntimeError when no CDP session."""

    def test_raises_without_cdp(self, fake_env: FakeEnv) -> None:
        """_dispatch_keypress with no CDP session raises RuntimeError."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot._cdp is None

        with pytest.raises(RuntimeError, match="requires an attached CDP session"):
            bot._dispatch_keypress("m")


class TestMaybeCaptureAccountStatsOnce:
    """Test the max-attempts guard in maybe_capture_account_stats_once."""

    def test_already_captured_returns_immediately(self, fake_env: FakeEnv) -> None:
        """When stats are already captured, no further attempts are made."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._account_stats_captured = True
        initial_attempts = bot._account_stats_attempts

        bot.maybe_capture_account_stats_once()

        assert bot._account_stats_attempts == initial_attempts

    def test_max_attempts_guard_returns_without_capture(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """After max attempts, no further capture is tried.

        Exercises the early return when ``_account_stats_attempts``
        has reached ``_ACCOUNT_STATS_MAX_CAPTURE_ATTEMPTS``.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.game_log_witness import _ACCOUNT_STATS_MAX_CAPTURE_ATTEMPTS

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._account_stats_attempts = _ACCOUNT_STATS_MAX_CAPTURE_ATTEMPTS
        bot._account_stats_captured = False

        bot.maybe_capture_account_stats_once()

        assert bot._account_stats_captured is False
        assert bot._account_stats_attempts == _ACCOUNT_STATS_MAX_CAPTURE_ATTEMPTS


class TestResolveSessionSeconds:
    """Tests for resolve_session_seconds (formerly _parse_session_seconds)."""

    def test_resolve_session_kills_defaults_to_zero(self) -> None:
        """No env value means no kill bound."""
        from tankpit_bot.bot.entry import resolve_session_kills

        assert resolve_session_kills(None) == 0

    def test_resolve_session_kills_parses_env(self) -> None:
        """A set env value becomes the kill target."""
        from tankpit_bot.bot.entry import resolve_session_kills

        assert resolve_session_kills("5") == 5

    def test_help_flag_raises_system_exit_zero(self) -> None:
        """--help raises SystemExit(0) after writing usage."""
        from tankpit_bot.bot.entry import resolve_session_seconds

        with pytest.raises(SystemExit) as exc_info:
            resolve_session_seconds(["--help"], None)
        assert exc_info.value.code == 0

    def test_h_flag_raises_system_exit_zero(self) -> None:
        """-h raises SystemExit(0) after writing usage."""
        from tankpit_bot.bot.entry import resolve_session_seconds

        with pytest.raises(SystemExit) as exc_info:
            resolve_session_seconds(["-h"], None)
        assert exc_info.value.code == 0

    def test_seconds_flag_returns_value(self) -> None:
        """--seconds N returns the integer N."""
        from tankpit_bot.bot.entry import resolve_session_seconds

        result = resolve_session_seconds(["--seconds", "300"], None)
        assert result == 300

    def test_no_argv_with_env_returns_env_value(self) -> None:
        """No CLI args with env_value returns int(env_value)."""
        from tankpit_bot.bot.entry import resolve_session_seconds

        result = resolve_session_seconds([], "600")
        assert result == 600

    def test_no_argv_no_env_returns_zero(self) -> None:
        """No CLI args and no env_value returns 0 (run until stopped)."""
        from tankpit_bot.bot.entry import resolve_session_seconds

        result = resolve_session_seconds([], None)
        assert result == 0

    def test_unrecognized_args_raises_system_exit_one(self) -> None:
        """Unrecognized argument shape raises SystemExit with usage."""
        from tankpit_bot.bot.entry import resolve_session_seconds

        with pytest.raises(SystemExit) as exc_info:
            resolve_session_seconds(["--bogus"], None)
        assert "unrecognized arguments" in str(exc_info.value)

    def test_seconds_alone_raises_system_exit(self) -> None:
        """--seconds without a value raises SystemExit."""
        from tankpit_bot.bot.entry import resolve_session_seconds

        with pytest.raises(SystemExit):
            resolve_session_seconds(["--seconds"], None)
