"""Tests for :mod:`tankpit_bot.bot.config`.

Covers the two env-driven bot-launch resolvers used by both
:mod:`tankpit_bot.bot.entry` (one-shot ``tankpit-bot``) and
:mod:`tankpit_bot.service.service_main` (long-running
``tankpit-bot-service``).
"""

from __future__ import annotations

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.bot.config import (
    DEFAULT_TARGET_URL,
    resolve_idle_exit_seconds,
    resolve_prefer_account,
    resolve_target_url,
)
from tests.conftest import FakeEnv


class TestDefaultTargetURL:
    """The canonical URL used when no env override applies."""

    def test_default_matches_the_public_production_url(self) -> None:
        """The default targets the live tankpit.com host."""
        assert DEFAULT_TARGET_URL == "https://tankpit.com/"


class TestResolveTargetURL:
    """``resolve_target_url`` env override contract."""

    def test_defaults_when_env_unset(self) -> None:
        """No ``TANKPIT_URL`` env var yields the canonical public URL."""
        _test_hooks.get_env = FakeEnv({})
        assert resolve_target_url() == "https://tankpit.com/"

    def test_env_override_wins(self) -> None:
        """A non-empty ``TANKPIT_URL`` env var replaces the default."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_URL": "https://staging.tankpit.com/"})
        assert resolve_target_url() == "https://staging.tankpit.com/"

    def test_empty_string_env_falls_back_to_default(self) -> None:
        """An empty-string override is treated as unset."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_URL": ""})
        assert resolve_target_url() == "https://tankpit.com/"


class TestResolveIdleExitSeconds:
    """``resolve_idle_exit_seconds`` env override contract (2026-07-29)."""

    def test_missing_env_returns_the_default_window(self) -> None:
        """No env var yields the 1800 s default idle window."""
        from tankpit_bot.service.constants import SERVICE_IDLE_EXIT_SECONDS

        _test_hooks.get_env = FakeEnv({})
        assert resolve_idle_exit_seconds() == SERVICE_IDLE_EXIT_SECONDS

    def test_zero_disables_the_idle_exit(self) -> None:
        """``"0"`` resolves to 0.0 — the always-on deployment mode."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_SERVICE_IDLE_EXIT_SECONDS": "0"})
        assert resolve_idle_exit_seconds() == 0.0

    def test_custom_window_wins(self) -> None:
        """A numeric override replaces the default."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_SERVICE_IDLE_EXIT_SECONDS": "600"})
        assert resolve_idle_exit_seconds() == 600.0

    def test_non_numeric_value_raises(self) -> None:
        """A malformed value fails loudly instead of picking a default."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_SERVICE_IDLE_EXIT_SECONDS": "forever"})
        with pytest.raises(ValueError):
            resolve_idle_exit_seconds()


class TestResolveVideoSettings:
    """``resolve_video_fps`` / ``resolve_video_quality`` contracts (2026-07-29)."""

    def test_fps_defaults_to_twelve(self) -> None:
        """No env var yields the 12 fps monitoring default."""
        from tankpit_bot.bot.config import resolve_video_fps

        _test_hooks.get_env = FakeEnv({})
        assert resolve_video_fps() == 12.0

    def test_fps_override_wins(self) -> None:
        """A numeric override replaces the default."""
        from tankpit_bot.bot.config import resolve_video_fps

        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_VIDEO_FPS": "20"})
        assert resolve_video_fps() == 20.0

    def test_fps_non_numeric_raises(self) -> None:
        """A malformed value fails loudly instead of picking a default."""
        from tankpit_bot.bot.config import resolve_video_fps

        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_VIDEO_FPS": "fast"})
        with pytest.raises(ValueError):
            resolve_video_fps()

    def test_quality_defaults_to_point_eight(self) -> None:
        """No env var yields the 0.8 JPEG-quality default."""
        from tankpit_bot.bot.config import resolve_video_quality

        _test_hooks.get_env = FakeEnv({})
        assert resolve_video_quality() == 0.8

    def test_quality_override_wins(self) -> None:
        """A numeric override replaces the default."""
        from tankpit_bot.bot.config import resolve_video_quality

        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_VIDEO_QUALITY": "0.6"})
        assert resolve_video_quality() == 0.6

    def test_quality_non_numeric_raises(self) -> None:
        """A malformed value fails loudly instead of picking a default."""
        from tankpit_bot.bot.config import resolve_video_quality

        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_VIDEO_QUALITY": "crisp"})
        with pytest.raises(ValueError):
            resolve_video_quality()


class TestResolvePreferAccount:
    """``resolve_prefer_account`` env override contract."""

    def test_missing_env_returns_false(self) -> None:
        """No env var yields guest login."""
        _test_hooks.get_env = FakeEnv({})
        assert resolve_prefer_account() is False

    def test_true_string_returns_true(self) -> None:
        """``"true"`` selects account login."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_PREFER_ACCOUNT": "true"})
        assert resolve_prefer_account() is True

    def test_one_string_returns_true(self) -> None:
        """``"1"`` selects account login."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_PREFER_ACCOUNT": "1"})
        assert resolve_prefer_account() is True

    def test_yes_string_returns_true(self) -> None:
        """``"yes"`` selects account login."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_PREFER_ACCOUNT": "yes"})
        assert resolve_prefer_account() is True

    def test_case_insensitive(self) -> None:
        """The comparison is case-insensitive."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_PREFER_ACCOUNT": "TRUE"})
        assert resolve_prefer_account() is True

    def test_other_string_returns_false(self) -> None:
        """Anything else stays on the guest login path."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_PREFER_ACCOUNT": "false"})
        assert resolve_prefer_account() is False
