"""Tests for ProbeResult and ProbeSession TypedDicts."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.types import (
    CapturedMessage,
    KeyInput,
    ProbeInput,
    ProbeResult,
    ProbeSession,
    decode_probe_result,
    decode_probe_session,
    encode_probe_result,
    encode_probe_session,
)

# =============================================================================
# ProbeResult Tests
# =============================================================================


def test_encode_probe_result() -> None:
    """Test encoding ProbeResult to JSON."""
    key_inp = KeyInput(key="w")
    probe_inp = ProbeInput(input_type="key", key_input=key_inp, mouse_input=None)
    msg = CapturedMessage(
        timestamp_ms=1000,
        direction="sent",
        payload="test",
        ws_url="wss://example.com",
    )
    result_obj = ProbeResult(
        input=probe_inp,
        timestamp_ms=500,
        messages_before_count=10,
        messages_after=[msg],
    )
    result = encode_probe_result(result_obj)
    assert result["timestamp_ms"] == 500
    assert result["messages_before_count"] == 10
    messages_after = result["messages_after"]
    assert type(messages_after) is list
    assert len(messages_after) == 1


def test_encode_probe_result_empty_messages() -> None:
    """Test encoding ProbeResult with no messages after."""
    key_inp = KeyInput(key="s")
    probe_inp = ProbeInput(input_type="key", key_input=key_inp, mouse_input=None)
    result_obj = ProbeResult(
        input=probe_inp,
        timestamp_ms=500,
        messages_before_count=10,
        messages_after=[],
    )
    result = encode_probe_result(result_obj)
    messages_after = result["messages_after"]
    assert type(messages_after) is list
    assert len(messages_after) == 0


def test_decode_probe_result() -> None:
    """Test decoding ProbeResult from JSON."""
    data: JSONObject = {
        "input": {
            "input_type": "key",
            "key_input": {"key": "w"},
            "mouse_input": None,
        },
        "timestamp_ms": 500,
        "messages_before_count": 10,
        "messages_after": [
            {
                "timestamp_ms": 1000,
                "direction": "sent",
                "payload": "test",
                "ws_url": "wss://example.com",
            }
        ],
    }
    result = decode_probe_result(data)
    assert result["timestamp_ms"] == 500
    assert result["messages_before_count"] == 10
    assert len(result["messages_after"]) == 1
    assert result["input"]["input_type"] == "key"


def test_decode_probe_result_missing_input() -> None:
    """Test decoding ProbeResult with missing input raises error."""
    data: JSONObject = {
        "timestamp_ms": 500,
        "messages_before_count": 10,
        "messages_after": [],
    }
    with pytest.raises(JSONTypeError, match="Missing required field 'input'"):
        decode_probe_result(data)


def test_decode_probe_result_invalid_input() -> None:
    """Test decoding ProbeResult with invalid input type raises error."""
    data: JSONObject = {
        "input": "not an object",
        "timestamp_ms": 500,
        "messages_before_count": 10,
        "messages_after": [],
    }
    with pytest.raises(JSONTypeError, match="'input' must be an object"):
        decode_probe_result(data)


def test_decode_probe_result_invalid_messages_after() -> None:
    """Test decoding ProbeResult with invalid messages_after raises error."""
    data: JSONObject = {
        "input": {
            "input_type": "key",
            "key_input": {"key": "w"},
            "mouse_input": None,
        },
        "timestamp_ms": 500,
        "messages_before_count": 10,
        "messages_after": ["not an object"],
    }
    with pytest.raises(JSONTypeError, match=r"messages_after\[0\] must be an object"):
        decode_probe_result(data)


# =============================================================================
# ProbeSession Tests
# =============================================================================


def test_encode_probe_session() -> None:
    """Test encoding ProbeSession to JSON."""
    key_inp = KeyInput(key="w")
    probe_inp = ProbeInput(input_type="key", key_input=key_inp, mouse_input=None)
    result_obj = ProbeResult(
        input=probe_inp,
        timestamp_ms=500,
        messages_before_count=10,
        messages_after=[],
    )
    session = ProbeSession(
        session_id="probe-123",
        start_timestamp_ms=0,
        end_timestamp_ms=10000,
        base_url="https://example.com",
        results=[result_obj],
    )
    result = encode_probe_session(session)
    assert result["session_id"] == "probe-123"
    assert result["start_timestamp_ms"] == 0
    assert result["end_timestamp_ms"] == 10000
    results_list = result["results"]
    assert type(results_list) is list
    assert len(results_list) == 1


def test_encode_probe_session_empty_results() -> None:
    """Test encoding ProbeSession with no results."""
    session = ProbeSession(
        session_id="probe-empty",
        start_timestamp_ms=0,
        end_timestamp_ms=1000,
        base_url="https://example.com",
        results=[],
    )
    result = encode_probe_session(session)
    results_list = result["results"]
    assert type(results_list) is list
    assert len(results_list) == 0


def test_decode_probe_session() -> None:
    """Test decoding ProbeSession from JSON."""
    data: JSONObject = {
        "session_id": "probe-123",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 10000,
        "base_url": "https://example.com",
        "results": [
            {
                "input": {
                    "input_type": "key",
                    "key_input": {"key": "w"},
                    "mouse_input": None,
                },
                "timestamp_ms": 500,
                "messages_before_count": 10,
                "messages_after": [],
            }
        ],
    }
    result = decode_probe_session(data)
    assert result["session_id"] == "probe-123"
    assert result["start_timestamp_ms"] == 0
    assert len(result["results"]) == 1
    assert result["results"][0]["input"]["input_type"] == "key"


def test_decode_probe_session_invalid_result() -> None:
    """Test decoding ProbeSession with invalid result raises error."""
    data: JSONObject = {
        "session_id": "probe-123",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 10000,
        "base_url": "https://example.com",
        "results": ["not an object"],
    }
    with pytest.raises(JSONTypeError, match=r"results\[0\] must be an object"):
        decode_probe_session(data)
