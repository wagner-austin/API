"""Tests for config module.

Note: Tests for environment variable parsing behavior (invalid floats, invalid ints, etc.)
belong in libs/platform_core/tests/ since that's where load_discordbot_settings is implemented.
DiscordBot tests should use the hooks pattern to inject test settings directly.
"""

from __future__ import annotations

from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.config import load_discordbot_settings


def test_load_settings_returns_settings_from_hook() -> None:
    """Verify that load_discordbot_settings uses the hook."""
    expected = build_settings(redis_url="redis://test-url")
    _test_hooks.load_settings = lambda: expected
    result = load_discordbot_settings()
    assert result["redis"]["redis_url"] == "redis://test-url"


def test_load_settings_with_blank_youtube_key() -> None:
    """Test that settings can have None youtube_api_key."""
    cfg = build_settings()
    _test_hooks.load_settings = lambda: cfg
    result = load_discordbot_settings()
    assert result["transcript"]["youtube_api_key"] is None


def test_load_settings_with_blank_redis_url() -> None:
    """Test that settings can have None redis_url."""
    cfg = build_settings(redis_url=None)
    _test_hooks.load_settings = lambda: cfg
    result = load_discordbot_settings()
    assert result["redis"]["redis_url"] is None
