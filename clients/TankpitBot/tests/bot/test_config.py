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
from tankpit_bot.stream.types import StreamConfigDict
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


def _resolved(config: StreamConfigDict | None) -> StreamConfigDict:
    """Fail loudly when streaming was expected to resolve.

    Args:
        config: What ``resolve_stream_config`` returned.

    Returns:
        The configuration.

    Raises:
        AssertionError: The resolver said streaming is off — these
            tests all set ``TANKPIT_STREAM_VIDEO``, so ``None`` is the
            test being wrong rather than a case to tolerate.
    """
    if config is None:
        raise AssertionError("expected streaming to resolve a configuration")
    return config


class TestResolveHudOverlay:
    """The in-page diagnostic HUD switch — default ON, demo turns it off."""

    def test_unset_means_the_hud_renders(self) -> None:
        """The desktop operator watching the browser is the default case."""
        from tankpit_bot.bot.config import resolve_hud_overlay

        _test_hooks.get_env = FakeEnv({})
        assert resolve_hud_overlay() is True

    def test_false_turns_the_hud_off(self) -> None:
        """The demo fleet's compose value disables the card."""
        from tankpit_bot.bot.config import resolve_hud_overlay

        _test_hooks.get_env = FakeEnv({"TANKPIT_HUD_OVERLAY": "false"})
        assert resolve_hud_overlay() is False

    def test_truthy_spellings_keep_it_on(self) -> None:
        """The shared yes-spellings work here like every other flag."""
        from tankpit_bot.bot.config import resolve_hud_overlay

        _test_hooks.get_env = FakeEnv({"TANKPIT_HUD_OVERLAY": "YES"})
        assert resolve_hud_overlay() is True


class TestResolveStreamConfig:
    """The display-capture configuration switch and its derivations."""

    def test_unset_means_no_capture(self) -> None:
        """No ``TANKPIT_STREAM_VIDEO`` resolves to None, reading nothing else."""
        from tankpit_bot.bot.config import resolve_stream_config

        _test_hooks.get_env = FakeEnv({})
        assert resolve_stream_config() is None

    def test_display_override_wins(self) -> None:
        """``TANKPIT_STREAM_DISPLAY`` beats the service-port derivation."""
        from tankpit_bot.bot.config import resolve_stream_config

        _test_hooks.get_env = FakeEnv(
            {
                "TANKPIT_STREAM_VIDEO": "true",
                "TANKPIT_STREAM_DISPLAY": "77",
                "TANKPIT_BOT_SERVICE_PORT": "27301",
            }
        )
        config = _resolved(resolve_stream_config())
        assert config["display"] == 77

    def test_display_derives_from_the_service_port(self) -> None:
        """The fleet's unique per-child port doubles as the display number."""
        from tankpit_bot.bot.config import resolve_stream_config

        _test_hooks.get_env = FakeEnv(
            {"TANKPIT_STREAM_VIDEO": "true", "TANKPIT_BOT_SERVICE_PORT": "27301"}
        )
        config = _resolved(resolve_stream_config())
        assert config["display"] == 27301

    def test_no_display_source_refuses(self) -> None:
        """Streaming on with no unique number to claim is refused loudly."""
        from tankpit_bot.bot.config import resolve_stream_config

        _test_hooks.get_env = FakeEnv({"TANKPIT_STREAM_VIDEO": "true"})
        with pytest.raises(ValueError, match="no unique display number"):
            resolve_stream_config()

    def test_defaults_fill_the_rest(self) -> None:
        """Geometry, rate, bitrate and segment length come from the constants."""
        from tankpit_bot.bot.config import (
            DEFAULT_STREAM_BITRATE_KBPS,
            DEFAULT_STREAM_FPS,
            DEFAULT_STREAM_HEIGHT,
            DEFAULT_STREAM_SEGMENT_SECONDS,
            DEFAULT_STREAM_WIDTH,
            resolve_stream_config,
        )

        _test_hooks.get_env = FakeEnv(
            {"TANKPIT_STREAM_VIDEO": "true", "TANKPIT_STREAM_DISPLAY": "9"}
        )
        config = _resolved(resolve_stream_config())
        assert config["width"] == DEFAULT_STREAM_WIDTH
        assert config["height"] == DEFAULT_STREAM_HEIGHT
        assert config["fps"] == DEFAULT_STREAM_FPS
        assert config["bitrate_kbps"] == DEFAULT_STREAM_BITRATE_KBPS
        assert config["segment_seconds"] == DEFAULT_STREAM_SEGMENT_SECONDS

    def test_fps_and_bitrate_overrides_win(self) -> None:
        """Numeric overrides replace the defaults."""
        from tankpit_bot.bot.config import resolve_stream_config

        _test_hooks.get_env = FakeEnv(
            {
                "TANKPIT_STREAM_VIDEO": "true",
                "TANKPIT_STREAM_DISPLAY": "9",
                "TANKPIT_STREAM_FPS": "24",
                "TANKPIT_STREAM_BITRATE_KBPS": "800",
            }
        )
        config = _resolved(resolve_stream_config())
        assert config["fps"] == 24
        assert config["bitrate_kbps"] == 800

    def test_non_numeric_display_raises(self) -> None:
        """A malformed number fails loudly instead of picking a default."""
        from tankpit_bot.bot.config import resolve_stream_config

        _test_hooks.get_env = FakeEnv(
            {"TANKPIT_STREAM_VIDEO": "true", "TANKPIT_STREAM_DISPLAY": "primary"}
        )
        with pytest.raises(ValueError):
            resolve_stream_config()

    def test_domain_validation_is_the_codec_s(self) -> None:
        """An out-of-domain value is refused by ``decode_stream_config``."""
        from tankpit_bot.bot.config import resolve_stream_config

        _test_hooks.get_env = FakeEnv(
            {
                "TANKPIT_STREAM_VIDEO": "true",
                "TANKPIT_STREAM_DISPLAY": "9",
                "TANKPIT_STREAM_FPS": "0",
            }
        )
        with pytest.raises(ValueError, match="fps must be positive"):
            resolve_stream_config()

    def test_hls_dir_is_the_instance_run_dir(self) -> None:
        """The segments land under the instance's own artifact namespace."""
        from tankpit_bot.bot.config import resolve_stream_config
        from tankpit_bot.runtime_artifacts import bot_run_dir

        _test_hooks.get_env = FakeEnv(
            {
                "TANKPIT_STREAM_VIDEO": "true",
                "TANKPIT_STREAM_DISPLAY": "9",
                "TANKPIT_BOT_INSTANCE": "demo-1",
            }
        )
        config = _resolved(resolve_stream_config())
        assert config["hls_dir"] == str(bot_run_dir("demo-1") / "hls")


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
