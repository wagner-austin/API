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
    resolve_env_flag,
    resolve_headless,
    resolve_prefer_account,
    resolve_target_url,
    resolve_weapon_resume_slack,
)
from tankpit_bot.service.config import resolve_idle_exit_seconds
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

    def test_fps_defaults_to_sixty(self) -> None:
        """No env var yields the 60 fps capture default.

        12 -> 30 -> 60, each step taken because a measurement said the
        previous one was still the floor. At 30 the public stream's
        MEDIAN inter-frame gap was 28 ms, faster than the 33 ms
        interval, with a third of gaps pinned at it -- the signature of
        sampling slower than the source paints.
        """
        from tankpit_bot.bot.config import resolve_video_fps

        _test_hooks.get_env = FakeEnv({})
        assert resolve_video_fps() == 60.0

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


class TestResolveHeadless:
    """``resolve_headless`` env override contract.

    The default is the load-bearing case. Until this resolver existed the
    bot path passed ``headless=False`` as a literal, so a containerized
    fleet child launched a headed Chromium against a machine with no X
    server and exited 1 about five seconds after spawn -- while
    ``docker-compose.yml`` had set ``TANKPIT_HEADLESS: "true"`` all along
    and nothing read it.
    """

    def test_missing_env_keeps_the_window(self) -> None:
        """Unset means headed: a desktop run exists to be watched."""
        _test_hooks.get_env = FakeEnv({})
        assert resolve_headless() is False

    def test_true_string_returns_true(self) -> None:
        """``"true"`` selects a windowless browser."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_HEADLESS": "true"})
        assert resolve_headless() is True

    def test_one_string_returns_true(self) -> None:
        """``"1"`` selects a windowless browser."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_HEADLESS": "1"})
        assert resolve_headless() is True

    def test_yes_string_returns_true(self) -> None:
        """``"yes"`` selects a windowless browser."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_HEADLESS": "yes"})
        assert resolve_headless() is True

    def test_case_insensitive(self) -> None:
        """The comparison is case-insensitive."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_HEADLESS": "TRUE"})
        assert resolve_headless() is True

    def test_other_string_keeps_the_window(self) -> None:
        """Anything else stays headed rather than guessing."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_HEADLESS": "false"})
        assert resolve_headless() is False

    def test_empty_string_keeps_the_window(self) -> None:
        """An empty value is not a request for headless."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_HEADLESS": ""})
        assert resolve_headless() is False

    def test_the_value_compose_actually_sets_is_honoured(self) -> None:
        """The literal string in docker-compose.yml resolves to headless.

        Asserted verbatim because the compose value and the resolver are
        the two halves that have to agree, and for the life of the
        container image they did not.
        """
        _test_hooks.get_env = FakeEnv({"TANKPIT_HEADLESS": "true"})
        assert resolve_headless() is True

    def test_both_booleans_accept_the_same_spellings(self) -> None:
        """One shared truthy set, so the two flags cannot drift apart."""
        for value in ("true", "1", "yes", "YES"):
            _test_hooks.get_env = FakeEnv({"TANKPIT_HEADLESS": value})
            headless = resolve_headless()
            _test_hooks.get_env = FakeEnv({"TANKPIT_PREFER_ACCOUNT": value})
            assert headless is resolve_prefer_account()


class TestResolveEnvFlag:
    """The one boolean parser every flag in the process shares.

    Tested directly rather than only through its callers: it is now the
    single place a spelling becomes a truth value, so a change here
    silently moves every flag at once.
    """

    def test_missing_env_is_false(self) -> None:
        """An unset variable is not an affirmative."""
        _test_hooks.get_env = FakeEnv({})
        assert resolve_env_flag("ANY_FLAG") is False

    def test_every_affirmative_spelling_is_accepted(self) -> None:
        """``true``, ``1`` and ``yes`` all mean yes, in any case."""
        for value in ("true", "1", "yes", "TRUE", "Yes"):
            _test_hooks.get_env = FakeEnv({"ANY_FLAG": value})
            assert resolve_env_flag("ANY_FLAG") is True

    def test_anything_else_is_false(self) -> None:
        """No other spelling is guessed at."""
        for value in ("false", "0", "no", "", "on"):
            _test_hooks.get_env = FakeEnv({"ANY_FLAG": value})
            assert resolve_env_flag("ANY_FLAG") is False

    def test_it_reads_the_variable_it_was_given(self) -> None:
        """The name is the argument, not a hardcoded one."""
        _test_hooks.get_env = FakeEnv({"FIRST": "true", "SECOND": "false"})
        assert (resolve_env_flag("FIRST"), resolve_env_flag("SECOND")) == (True, False)


class TestWeaponResumeSlack:
    """Contract for ``resolve_weapon_resume_slack``."""

    def test_default_is_zero_full_stock_contract(self) -> None:
        """Unset env keeps the verbatim full-stock resume bar."""
        _test_hooks.get_env = FakeEnv({})
        assert resolve_weapon_resume_slack() == 0

    def test_positive_slack_parses(self) -> None:
        """A configured slack of 5 mirrors the radar cap-5 shape."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_WEAPON_RESUME_SLACK": "5"})
        assert resolve_weapon_resume_slack() == 5

    def test_negative_slack_is_rejected_loudly(self) -> None:
        """A negative slack is a config error, not a clamp."""
        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_WEAPON_RESUME_SLACK": "-1"})
        with pytest.raises(ValueError, match="must be >= 0"):
            resolve_weapon_resume_slack()
