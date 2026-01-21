"""Tests for SnifferConfig and BotConfig TypedDicts."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.types import (
    BotConfig,
    SnifferConfig,
    decode_bot_config,
    decode_sniffer_config,
    encode_bot_config,
    encode_sniffer_config,
)

# =============================================================================
# SnifferConfig Tests
# =============================================================================


def test_encode_sniffer_config() -> None:
    """Test encoding SnifferConfig to JSON."""
    config = SnifferConfig(
        target_url="https://tankpit.com",
        output_path="output.json",
        headless=True,
        capture_duration_ms=30000,
    )
    result = encode_sniffer_config(config)
    assert result["target_url"] == "https://tankpit.com"
    assert result["headless"] is True


def test_decode_sniffer_config() -> None:
    """Test decoding SnifferConfig from JSON."""
    data: JSONObject = {
        "target_url": "https://tankpit.com",
        "output_path": "output.json",
        "headless": False,
        "capture_duration_ms": 30000,
    }
    result = decode_sniffer_config(data)
    assert result["target_url"] == "https://tankpit.com"
    assert result["headless"] is False


def test_decode_sniffer_config_missing_headless() -> None:
    """Test decoding SnifferConfig with missing headless raises error."""
    data: JSONObject = {
        "target_url": "https://tankpit.com",
        "output_path": "output.json",
        "capture_duration_ms": 30000,
    }
    with pytest.raises(JSONTypeError, match="Missing required field 'headless'"):
        decode_sniffer_config(data)


def test_decode_sniffer_config_invalid_headless() -> None:
    """Test decoding SnifferConfig with invalid headless type raises error."""
    data: JSONObject = {
        "target_url": "https://tankpit.com",
        "output_path": "output.json",
        "headless": "yes",
        "capture_duration_ms": 30000,
    }
    with pytest.raises(JSONTypeError, match="'headless' must be a boolean"):
        decode_sniffer_config(data)


# =============================================================================
# BotConfig Tests
# =============================================================================


def test_encode_bot_config() -> None:
    """Test encoding BotConfig to JSON."""
    config = BotConfig(
        ws_url="wss://tankpit.com/game",
        username="testbot",
        game_id="abc123",
    )
    result = encode_bot_config(config)
    assert result["ws_url"] == "wss://tankpit.com/game"
    assert result["username"] == "testbot"
    assert result["game_id"] == "abc123"


def test_decode_bot_config() -> None:
    """Test decoding BotConfig from JSON."""
    data: JSONObject = {
        "ws_url": "wss://tankpit.com/game",
        "username": "testbot",
        "game_id": "abc123",
    }
    result = decode_bot_config(data)
    assert result["ws_url"] == "wss://tankpit.com/game"
    assert result["username"] == "testbot"
