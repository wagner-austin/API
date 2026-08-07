"""Tests for CaptureSession TypedDict."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.types import (
    CapturedMessage,
    CaptureSession,
    GameLogEntryWithTimestamp,
    decode_capture_session,
    encode_capture_session,
)


def test_encode_capture_session() -> None:
    """Test encoding CaptureSession to JSON."""
    msg = CapturedMessage(
        timestamp_ms=1000,
        direction="sent",
        payload="test",
        ws_url="wss://example.com",
    )
    session = CaptureSession(
        session_id="abc-123",
        start_timestamp_ms=0,
        end_timestamp_ms=5000,
        base_url="https://example.com",
        messages=[msg],
        magic="test_magic_key",
        game_log=[],
        tank_names={},
    )
    result = encode_capture_session(session)
    assert result["session_id"] == "abc-123"
    assert result["start_timestamp_ms"] == 0
    assert result["end_timestamp_ms"] == 5000
    assert result["base_url"] == "https://example.com"
    assert result["magic"] == "test_magic_key"
    messages_list = result["messages"]
    assert type(messages_list) is list
    assert len(messages_list) == 1
    assert result["game_log"] == []
    assert result["tank_names"] == {}


def test_encode_capture_session_with_none_end() -> None:
    """Test encoding CaptureSession with None end timestamp."""
    session = CaptureSession(
        session_id="abc-123",
        start_timestamp_ms=0,
        end_timestamp_ms=None,
        base_url="https://example.com",
        messages=[],
        magic=None,
        game_log=[],
        tank_names={},
    )
    result = encode_capture_session(session)
    assert result["end_timestamp_ms"] is None
    assert result["magic"] is None


def test_decode_capture_session() -> None:
    """Test decoding CaptureSession from JSON."""
    data: JSONObject = {
        "session_id": "abc-123",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 5000,
        "base_url": "https://example.com",
        "messages": [
            {
                "timestamp_ms": 1000,
                "direction": "sent",
                "payload": "test",
                "ws_url": "wss://example.com",
            }
        ],
        "magic": "decoded_magic_key",
        "game_log": [],
        "tank_names": {},
    }
    result = decode_capture_session(data)
    assert result["session_id"] == "abc-123"
    assert len(result["messages"]) == 1
    assert result["messages"][0]["direction"] == "sent"
    assert result["magic"] == "decoded_magic_key"


def test_decode_capture_session_with_none_magic() -> None:
    """Test decoding CaptureSession with None magic."""
    data: JSONObject = {
        "session_id": "abc-123",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 5000,
        "base_url": "https://example.com",
        "messages": [],
        "magic": None,
        "game_log": [],
        "tank_names": {},
    }
    result = decode_capture_session(data)
    assert result["magic"] is None


def test_decode_capture_session_with_missing_magic() -> None:
    """Test decoding CaptureSession with missing magic field."""
    data: JSONObject = {
        "session_id": "abc-123",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 5000,
        "base_url": "https://example.com",
        "messages": [],
        "game_log": [],
        "tank_names": {},
    }
    result = decode_capture_session(data)
    assert result["magic"] is None


def test_decode_capture_session_invalid_message() -> None:
    """Test decoding CaptureSession with invalid message raises error."""
    data: JSONObject = {
        "session_id": "abc-123",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 5000,
        "base_url": "https://example.com",
        "messages": ["not an object"],
        "magic": None,
    }
    with pytest.raises(JSONTypeError, match=r"messages\[0\] must be an object"):
        decode_capture_session(data)


def test_encode_capture_session_with_game_log_and_tank_names() -> None:
    """Test encoding CaptureSession with non-empty game_log and tank_names."""
    log_entry = GameLogEntryWithTimestamp(
        timestamp_ms=1000,
        text="You hit enemy-1",
        category="combat",
    )
    session = CaptureSession(
        session_id="session-456",
        start_timestamp_ms=0,
        end_timestamp_ms=5000,
        base_url="https://tankpit.com",
        messages=[],
        magic="xor_key_here",
        game_log=[log_entry],
        tank_names={"123": "Player1", "456": "Enemy2"},
    )
    result = encode_capture_session(session)
    # Verify tank_names dict was converted (exercises _str_dict_to_json)
    tank_names = result["tank_names"]
    assert type(tank_names) is dict
    assert tank_names["123"] == "Player1"
    assert tank_names["456"] == "Enemy2"
    # Verify game_log was encoded
    game_log = result["game_log"]
    assert type(game_log) is list
    assert len(game_log) == 1


def test_decode_capture_session_with_game_log_and_tank_names() -> None:
    """Test decoding CaptureSession with game_log and tank_names."""
    data: JSONObject = {
        "session_id": "session-789",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 10000,
        "base_url": "https://tankpit.com",
        "messages": [],
        "magic": "decoded_key",
        "game_log": [
            {
                "timestamp_ms": 500,
                "text": "Enemy destroyed",
                "category": "combat",
            }
        ],
        "tank_names": {"100": "TankA", "200": "TankB"},
    }
    result = decode_capture_session(data)
    # Verify game_log was decoded (exercises lines 307-308)
    assert len(result["game_log"]) == 1
    assert result["game_log"][0]["text"] == "Enemy destroyed"
    # Verify tank_names was decoded (exercises lines 321-322)
    assert result["tank_names"]["100"] == "TankA"
    assert result["tank_names"]["200"] == "TankB"


def test_decode_capture_session_rejects_non_dict_game_log_entry() -> None:
    """A non-object game_log entry is a malformed capture, not one to skip."""
    data: JSONObject = {
        "session_id": "session-skip",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 10000,
        "base_url": "https://tankpit.com",
        "messages": [],
        "magic": None,
        "game_log": [
            "not a dict",
            {
                "timestamp_ms": 500,
                "text": "Valid entry",
                "category": "info",
            },
        ],
        "tank_names": {},
    }
    with pytest.raises(JSONTypeError, match=r"game_log\[0\] must be an object"):
        decode_capture_session(data)


def test_decode_capture_session_rejects_non_string_tank_name_values() -> None:
    """A non-string tank name is a malformed capture, not one to skip."""
    data: JSONObject = {
        "session_id": "session-names",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 10000,
        "base_url": "https://tankpit.com",
        "messages": [],
        "magic": None,
        "game_log": [],
        "tank_names": {"100": "ValidName", "200": 12345, "300": None},
    }
    with pytest.raises(JSONTypeError, match=r"tank_names\['200'\] must be a string"):
        decode_capture_session(data)
