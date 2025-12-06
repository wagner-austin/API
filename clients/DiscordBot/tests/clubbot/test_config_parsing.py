from __future__ import annotations

import logging

import pytest
from tests.support.settings import build_settings

from clubbot.config import require_discord_token as require_token


def test_retry_intervals_respected_without_env() -> None:
    """Test that retry intervals are set correctly in nested config."""
    cfg = build_settings()
    # The retry intervals are in the redis nested config
    assert cfg["redis"]["rq_transcript_retry_intervals_sec"] == (60, 300)


def test_boolean_flags_taken_from_config() -> None:
    """Test boolean flags in nested configs."""
    cfg = build_settings()
    # QR public_responses is in qr nested config
    assert cfg["qr"]["public_responses"] is False
    # Transcript public_responses is in transcript nested config
    assert cfg["transcript"]["public_responses"] is False


def test_guild_ids_in_discord_config() -> None:
    """Test guild_ids are in the discord nested config."""
    cfg = build_settings()
    # Guild IDs are in the discord nested config
    assert cfg["discord"]["guild_ids"] == []


def test_require_token_raises_when_missing() -> None:
    """Test that require_discord_token raises when token is empty."""
    cfg = build_settings(discord_token="")
    with pytest.raises(RuntimeError):
        require_token(cfg)


logger = logging.getLogger(__name__)
