"""Tests for WebSocketInfo and CDP WebSocket TypedDicts."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.types import (
    CDPWebSocketCreatedEvent,
    CDPWebSocketFrame,
    CDPWebSocketFrameEvent,
    WebSocketInfo,
    decode_cdp_websocket_created_event,
    decode_cdp_websocket_frame,
    decode_cdp_websocket_frame_event,
    decode_websocket_info,
    encode_cdp_websocket_created_event,
    encode_cdp_websocket_frame,
    encode_cdp_websocket_frame_event,
    encode_websocket_info,
)

# =============================================================================
# WebSocketInfo Tests
# =============================================================================


def test_encode_websocket_info() -> None:
    """Test encoding WebSocketInfo to JSON."""
    info = WebSocketInfo(
        url="wss://tankpit.com/game",
        request_id="123.45",
        timestamp_ms=1234567890,
    )
    result = encode_websocket_info(info)
    assert result["url"] == "wss://tankpit.com/game"
    assert result["request_id"] == "123.45"


def test_decode_websocket_info() -> None:
    """Test decoding WebSocketInfo from JSON."""
    data: JSONObject = {
        "url": "wss://tankpit.com/game",
        "request_id": "123.45",
        "timestamp_ms": 1234567890,
    }
    result = decode_websocket_info(data)
    assert result["url"] == "wss://tankpit.com/game"


# =============================================================================
# CDP WebSocket Frame Tests
# =============================================================================


def test_encode_cdp_websocket_frame() -> None:
    """Test encoding CDPWebSocketFrame to JSON."""
    frame = CDPWebSocketFrame(
        opcode=1,
        mask=True,
        payloadData='{"type":"move"}',
    )
    result = encode_cdp_websocket_frame(frame)
    assert result["opcode"] == 1
    assert result["mask"] is True
    assert result["payloadData"] == '{"type":"move"}'


def test_decode_cdp_websocket_frame() -> None:
    """Test decoding CDPWebSocketFrame from JSON."""
    data: JSONObject = {
        "opcode": 1,
        "mask": False,
        "payloadData": "test payload",
    }
    result = decode_cdp_websocket_frame(data)
    assert result["opcode"] == 1
    assert result["mask"] is False
    assert result["payloadData"] == "test payload"


# =============================================================================
# CDP WebSocket Frame Event Tests
# =============================================================================


def test_encode_cdp_websocket_frame_event() -> None:
    """Test encoding CDPWebSocketFrameEvent to JSON."""
    frame = CDPWebSocketFrame(opcode=1, mask=True, payloadData="test")
    event = CDPWebSocketFrameEvent(
        requestId="123.45",
        timestamp=12345.678,
        response=frame,
    )
    result = encode_cdp_websocket_frame_event(event)
    assert result["requestId"] == "123.45"
    assert result["timestamp"] == 12345.678
    response = result["response"]
    assert type(response) is dict


def test_decode_cdp_websocket_frame_event() -> None:
    """Test decoding CDPWebSocketFrameEvent from JSON."""
    data: JSONObject = {
        "requestId": "123.45",
        "timestamp": 12345.678,
        "response": {
            "opcode": 1,
            "mask": False,
            "payloadData": "test",
        },
    }
    result = decode_cdp_websocket_frame_event(data)
    assert result["requestId"] == "123.45"
    assert result["timestamp"] == 12345.678
    assert result["response"]["payloadData"] == "test"


def test_decode_cdp_websocket_frame_event_missing_response() -> None:
    """Test decoding CDPWebSocketFrameEvent with missing response raises error."""
    data: JSONObject = {
        "requestId": "123.45",
        "timestamp": 12345.678,
    }
    with pytest.raises(JSONTypeError, match="Missing required field 'response'"):
        decode_cdp_websocket_frame_event(data)


def test_decode_cdp_websocket_frame_event_invalid_response() -> None:
    """Test decoding CDPWebSocketFrameEvent with invalid response raises error."""
    data: JSONObject = {
        "requestId": "123.45",
        "timestamp": 12345.678,
        "response": "not an object",
    }
    with pytest.raises(JSONTypeError, match="'response' must be an object"):
        decode_cdp_websocket_frame_event(data)


# =============================================================================
# CDP WebSocket Created Event Tests
# =============================================================================


def test_encode_cdp_websocket_created_event() -> None:
    """Test encoding CDPWebSocketCreatedEvent to JSON."""
    event = CDPWebSocketCreatedEvent(
        requestId="123.45",
        url="wss://tankpit.com/game",
    )
    result = encode_cdp_websocket_created_event(event)
    assert result["requestId"] == "123.45"
    assert result["url"] == "wss://tankpit.com/game"


def test_decode_cdp_websocket_created_event() -> None:
    """Test decoding CDPWebSocketCreatedEvent from JSON."""
    data: JSONObject = {
        "requestId": "123.45",
        "url": "wss://tankpit.com/game",
    }
    result = decode_cdp_websocket_created_event(data)
    assert result["requestId"] == "123.45"
    assert result["url"] == "wss://tankpit.com/game"
