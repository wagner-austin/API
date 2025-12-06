from __future__ import annotations

import pytest

from clubbot.config import load_discordbot_settings as load_config


def test_float_parsing_raises_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    # Strict validation: invalid floats should raise ValueError
    monkeypatch.setenv("TRANSCRIPT_STT_RTF", "bad")
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    with pytest.raises(ValueError):
        load_config()


def test_int_parsing_raises_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    # Strict validation: invalid integers should raise ValueError
    monkeypatch.setenv("QRCODE_RATE_LIMIT", "bad")
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    with pytest.raises(ValueError):
        load_config()


def test_youtube_key_blank_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    cfg = load_config()
    assert cfg["transcript"]["youtube_api_key"] is None


def test_redis_url_none_when_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    monkeypatch.setenv("REDIS_URL", " ")
    cfg = load_config()
    assert cfg["redis"]["redis_url"] is None
