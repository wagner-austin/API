"""Tests for tankpit_bot.types module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.types import (
    BotConfig,
    CapturedMessage,
    CaptureSession,
    CDPWebSocketCreatedEvent,
    CDPWebSocketFrame,
    CDPWebSocketFrameEvent,
    KeyInput,
    MouseInput,
    ProbeInput,
    ProbeResult,
    ProbeSession,
    SnifferConfig,
    WebSocketInfo,
    decode_bot_config,
    decode_capture_session,
    decode_captured_message,
    decode_cdp_websocket_created_event,
    decode_cdp_websocket_frame,
    decode_cdp_websocket_frame_event,
    decode_key_input,
    decode_mouse_input,
    decode_probe_input,
    decode_probe_result,
    decode_probe_session,
    decode_sniffer_config,
    decode_websocket_info,
    encode_bot_config,
    encode_capture_session,
    encode_captured_message,
    encode_cdp_websocket_created_event,
    encode_cdp_websocket_frame,
    encode_cdp_websocket_frame_event,
    encode_key_input,
    encode_mouse_input,
    encode_probe_input,
    encode_probe_result,
    encode_probe_session,
    encode_sniffer_config,
    encode_websocket_info,
)

# =============================================================================
# CapturedMessage Tests
# =============================================================================


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


# =============================================================================
# CaptureSession Tests
# =============================================================================


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
    )
    result = encode_capture_session(session)
    assert result["session_id"] == "abc-123"
    assert result["start_timestamp_ms"] == 0
    assert result["end_timestamp_ms"] == 5000
    assert result["base_url"] == "https://example.com"
    messages_list = result["messages"]
    assert type(messages_list) is list
    assert len(messages_list) == 1


def test_encode_capture_session_with_none_end() -> None:
    """Test encoding CaptureSession with None end timestamp."""
    session = CaptureSession(
        session_id="abc-123",
        start_timestamp_ms=0,
        end_timestamp_ms=None,
        base_url="https://example.com",
        messages=[],
    )
    result = encode_capture_session(session)
    assert result["end_timestamp_ms"] is None


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
    }
    result = decode_capture_session(data)
    assert result["session_id"] == "abc-123"
    assert len(result["messages"]) == 1
    assert result["messages"][0]["direction"] == "sent"


def test_decode_capture_session_invalid_message() -> None:
    """Test decoding CaptureSession with invalid message raises error."""
    data: JSONObject = {
        "session_id": "abc-123",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 5000,
        "base_url": "https://example.com",
        "messages": ["not an object"],
    }
    with pytest.raises(JSONTypeError, match=r"messages\[0\] must be an object"):
        decode_capture_session(data)


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


# =============================================================================
# KeyInput Tests
# =============================================================================


def test_encode_key_input() -> None:
    """Test encoding KeyInput to JSON."""
    inp = KeyInput(key="w")
    result = encode_key_input(inp)
    assert result["key"] == "w"


def test_decode_key_input() -> None:
    """Test decoding KeyInput from JSON."""
    data: JSONObject = {"key": "ArrowUp"}
    result = decode_key_input(data)
    assert result["key"] == "ArrowUp"


# =============================================================================
# MouseInput Tests
# =============================================================================


def test_encode_mouse_input() -> None:
    """Test encoding MouseInput to JSON."""
    inp = MouseInput(x=100, y=200, button="left")
    result = encode_mouse_input(inp)
    assert result["x"] == 100
    assert result["y"] == 200
    assert result["button"] == "left"


def test_decode_mouse_input() -> None:
    """Test decoding MouseInput from JSON."""
    data: JSONObject = {"x": 100, "y": 200, "button": "right"}
    result = decode_mouse_input(data)
    assert result["x"] == 100
    assert result["y"] == 200
    assert result["button"] == "right"


def test_decode_mouse_input_middle_button() -> None:
    """Test decoding MouseInput with middle button."""
    data: JSONObject = {"x": 50, "y": 50, "button": "middle"}
    result = decode_mouse_input(data)
    assert result["button"] == "middle"


def test_decode_mouse_input_invalid_button() -> None:
    """Test decoding MouseInput with invalid button raises error."""
    data: JSONObject = {"x": 100, "y": 200, "button": "invalid"}
    with pytest.raises(JSONTypeError, match="must be 'left', 'right', or 'middle'"):
        decode_mouse_input(data)


# =============================================================================
# ProbeInput Tests
# =============================================================================


def test_encode_probe_input_key() -> None:
    """Test encoding ProbeInput with key input."""
    key_inp = KeyInput(key="space")
    inp = ProbeInput(input_type="key", key_input=key_inp, mouse_input=None)
    result = encode_probe_input(inp)
    assert result["input_type"] == "key"
    key_result = result["key_input"]
    assert type(key_result) is dict
    assert key_result["key"] == "space"
    assert result["mouse_input"] is None


def test_encode_probe_input_mouse() -> None:
    """Test encoding ProbeInput with mouse input."""
    mouse_inp = MouseInput(x=300, y=400, button="left")
    inp = ProbeInput(input_type="mouse", key_input=None, mouse_input=mouse_inp)
    result = encode_probe_input(inp)
    assert result["input_type"] == "mouse"
    assert result["key_input"] is None
    mouse_result = result["mouse_input"]
    assert type(mouse_result) is dict
    assert mouse_result["x"] == 300


def test_decode_probe_input_key() -> None:
    """Test decoding ProbeInput with key input."""
    data: JSONObject = {
        "input_type": "key",
        "key_input": {"key": "w"},
        "mouse_input": None,
    }
    result = decode_probe_input(data)
    assert result["input_type"] == "key"
    key_input = result["key_input"]
    assert type(key_input) is dict
    assert key_input["key"] == "w"
    assert result["mouse_input"] is None


def test_decode_probe_input_mouse() -> None:
    """Test decoding ProbeInput with mouse input."""
    data: JSONObject = {
        "input_type": "mouse",
        "key_input": None,
        "mouse_input": {"x": 100, "y": 200, "button": "left"},
    }
    result = decode_probe_input(data)
    assert result["input_type"] == "mouse"
    assert result["key_input"] is None
    mouse_input = result["mouse_input"]
    assert type(mouse_input) is dict
    assert mouse_input["x"] == 100


def test_decode_probe_input_invalid_type() -> None:
    """Test decoding ProbeInput with invalid input type raises error."""
    data: JSONObject = {
        "input_type": "invalid",
        "key_input": None,
        "mouse_input": None,
    }
    with pytest.raises(JSONTypeError, match="must be 'key' or 'mouse'"):
        decode_probe_input(data)


def test_decode_probe_input_invalid_key_input() -> None:
    """Test decoding ProbeInput with invalid key_input type raises error."""
    data: JSONObject = {
        "input_type": "key",
        "key_input": "not an object",
        "mouse_input": None,
    }
    with pytest.raises(JSONTypeError, match="'key_input' must be an object"):
        decode_probe_input(data)


def test_decode_probe_input_invalid_mouse_input() -> None:
    """Test decoding ProbeInput with invalid mouse_input type raises error."""
    data: JSONObject = {
        "input_type": "mouse",
        "key_input": None,
        "mouse_input": "not an object",
    }
    with pytest.raises(JSONTypeError, match="'mouse_input' must be an object"):
        decode_probe_input(data)


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
