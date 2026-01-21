"""Tests for CapturedMessage TypedDict."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.types import (
    CapturedMessage,
    decode_captured_message,
    encode_captured_message,
)


def test_encode_captured_message() -> None:
    """Test encoding CapturedMessage to JSON."""
    msg = CapturedMessage(
        timestamp_ms=1234567890,
        direction="sent",
        payload='{"type":"move"}',
        ws_url="wss://tankpit.com/game",
    )
    result = encode_captured_message(msg)
    assert result["timestamp_ms"] == 1234567890
    assert result["direction"] == "sent"
    assert result["payload"] == '{"type":"move"}'
    assert result["ws_url"] == "wss://tankpit.com/game"


def test_decode_captured_message_sent() -> None:
    """Test decoding CapturedMessage with sent direction."""
    data: JSONObject = {
        "timestamp_ms": 1234567890,
        "direction": "sent",
        "payload": '{"type":"move"}',
        "ws_url": "wss://tankpit.com/game",
    }
    result = decode_captured_message(data)
    assert result["timestamp_ms"] == 1234567890
    assert result["direction"] == "sent"
    assert result["payload"] == '{"type":"move"}'
    assert result["ws_url"] == "wss://tankpit.com/game"


def test_decode_captured_message_received() -> None:
    """Test decoding CapturedMessage with received direction."""
    data: JSONObject = {
        "timestamp_ms": 1234567890,
        "direction": "received",
        "payload": '{"type":"state"}',
        "ws_url": "wss://tankpit.com/game",
    }
    result = decode_captured_message(data)
    assert result["direction"] == "received"


def test_decode_captured_message_invalid_direction() -> None:
    """Test decoding CapturedMessage with invalid direction raises error."""
    data: JSONObject = {
        "timestamp_ms": 1234567890,
        "direction": "invalid",
        "payload": "test",
        "ws_url": "wss://example.com",
    }
    with pytest.raises(JSONTypeError, match="must be 'sent' or 'received'"):
        decode_captured_message(data)


def test_decode_captured_message_missing_field() -> None:
    """Test decoding CapturedMessage with missing field raises error."""
    data: JSONObject = {
        "timestamp_ms": 1234567890,
        "direction": "sent",
        # missing payload and ws_url
    }
    with pytest.raises(JSONTypeError, match="Missing required field"):
        decode_captured_message(data)
