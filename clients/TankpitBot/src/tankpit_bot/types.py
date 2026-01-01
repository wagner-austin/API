"""TypedDict definitions for TankpitBot with encode/decode functions.

All TypedDicts use immutable semantics with proper validation on decode.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_int,
    optional_str,
    require_bool,
    require_float,
    require_int,
    require_list,
    require_str,
)

# =============================================================================
# Literal Types
# =============================================================================

MessageDirection = Literal["sent", "received"]
InputType = Literal["key", "mouse"]
MouseButton = Literal["left", "right", "middle"]


# =============================================================================
# Validation Helpers
# =============================================================================


def _require_message_direction(obj: JSONObject, key: str) -> MessageDirection:
    """Extract and validate MessageDirection from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Key to extract.

    Returns:
        Validated MessageDirection literal.

    Raises:
        JSONTypeError: If value is not a valid MessageDirection.
    """
    value = require_str(obj, key)
    if value == "sent":
        return "sent"
    if value == "received":
        return "received"
    raise JSONTypeError(f"Field '{key}' must be 'sent' or 'received', got '{value}'")


def _require_input_type(obj: JSONObject, key: str) -> InputType:
    """Extract and validate InputType from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Key to extract.

    Returns:
        Validated InputType literal.

    Raises:
        JSONTypeError: If value is not a valid InputType.
    """
    value = require_str(obj, key)
    if value == "key":
        return "key"
    if value == "mouse":
        return "mouse"
    raise JSONTypeError(f"Field '{key}' must be 'key' or 'mouse', got '{value}'")


def _require_mouse_button(obj: JSONObject, key: str) -> MouseButton:
    """Extract and validate MouseButton from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Key to extract.

    Returns:
        Validated MouseButton literal.

    Raises:
        JSONTypeError: If value is not a valid MouseButton.
    """
    value = require_str(obj, key)
    if value == "left":
        return "left"
    if value == "right":
        return "right"
    if value == "middle":
        return "middle"
    raise JSONTypeError(f"Field '{key}' must be 'left', 'right', or 'middle', got '{value}'")


# =============================================================================
# Captured WebSocket Message
# =============================================================================


class CapturedMessage(TypedDict):
    """A captured WebSocket message.

    Attributes:
        timestamp_ms: Unix timestamp in milliseconds when message was captured.
        direction: Whether message was sent or received.
        payload: Raw message payload as string.
        ws_url: WebSocket URL this message was captured from.
    """

    timestamp_ms: int
    direction: MessageDirection
    payload: str
    ws_url: str


def encode_captured_message(msg: CapturedMessage) -> JSONObject:
    """Encode CapturedMessage to JSON-serializable dict.

    Args:
        msg: CapturedMessage to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "timestamp_ms": msg["timestamp_ms"],
        "direction": msg["direction"],
        "payload": msg["payload"],
        "ws_url": msg["ws_url"],
    }
    return result


def decode_captured_message(data: JSONObject) -> CapturedMessage:
    """Decode CapturedMessage from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CapturedMessage.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return CapturedMessage(
        timestamp_ms=require_int(data, "timestamp_ms"),
        direction=_require_message_direction(data, "direction"),
        payload=require_str(data, "payload"),
        ws_url=require_str(data, "ws_url"),
    )


# =============================================================================
# Capture Session
# =============================================================================


class CaptureSession(TypedDict):
    """A complete WebSocket capture session.

    Attributes:
        session_id: Unique identifier for this capture session.
        start_timestamp_ms: Unix timestamp when capture started.
        end_timestamp_ms: Unix timestamp when capture ended (None if ongoing).
        base_url: Base URL of the site being captured.
        messages: List of captured messages.
        magic: XOR magic key from tankpit.magic (None if not captured).
    """

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int | None
    base_url: str
    messages: list[CapturedMessage]
    magic: str | None


def encode_capture_session(session: CaptureSession) -> JSONObject:
    """Encode CaptureSession to JSON-serializable dict.

    Args:
        session: CaptureSession to encode.

    Returns:
        JSON-serializable dict representation.
    """
    encoded_messages: list[JSONValue] = [encode_captured_message(m) for m in session["messages"]]
    result: JSONObject = {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "messages": encoded_messages,
        "magic": session["magic"],
    }
    return result


def decode_capture_session(data: JSONObject) -> CaptureSession:
    """Decode CaptureSession from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CaptureSession.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    raw_messages = require_list(data, "messages")
    messages: list[CapturedMessage] = []
    for idx, raw_msg in enumerate(raw_messages):
        if not isinstance(raw_msg, dict):
            raise JSONTypeError(f"messages[{idx}] must be an object")
        messages.append(decode_captured_message(raw_msg))

    return CaptureSession(
        session_id=require_str(data, "session_id"),
        start_timestamp_ms=require_int(data, "start_timestamp_ms"),
        end_timestamp_ms=optional_int(data, "end_timestamp_ms"),
        base_url=require_str(data, "base_url"),
        messages=messages,
        magic=optional_str(data, "magic"),
    )


# =============================================================================
# Sniffer Configuration
# =============================================================================


class SnifferConfig(TypedDict):
    """Configuration for the WebSocket sniffer.

    Attributes:
        target_url: URL to navigate to and capture WebSocket traffic.
        output_path: Path to save captured session data.
        headless: Whether to run browser in headless mode.
        capture_duration_ms: How long to capture traffic in milliseconds.
    """

    target_url: str
    output_path: str
    headless: bool
    capture_duration_ms: int


def encode_sniffer_config(config: SnifferConfig) -> JSONObject:
    """Encode SnifferConfig to JSON-serializable dict.

    Args:
        config: SnifferConfig to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "target_url": config["target_url"],
        "output_path": config["output_path"],
        "headless": config["headless"],
        "capture_duration_ms": config["capture_duration_ms"],
    }
    return result


def decode_sniffer_config(data: JSONObject) -> SnifferConfig:
    """Decode SnifferConfig from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated SnifferConfig.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    headless_val = data.get("headless")
    if headless_val is None:
        raise JSONTypeError("Missing required field 'headless'")
    if not isinstance(headless_val, bool):
        actual_type = type(headless_val).__name__
        raise JSONTypeError(f"Field 'headless' must be a boolean, got {actual_type}")

    return SnifferConfig(
        target_url=require_str(data, "target_url"),
        output_path=require_str(data, "output_path"),
        headless=headless_val,
        capture_duration_ms=require_int(data, "capture_duration_ms"),
    )


# =============================================================================
# WebSocket Connection Info
# =============================================================================


class WebSocketInfo(TypedDict):
    """Information about a discovered WebSocket connection.

    Attributes:
        url: Full WebSocket URL.
        request_id: CDP request ID for this connection.
        timestamp_ms: When the connection was established.
    """

    url: str
    request_id: str
    timestamp_ms: int


def encode_websocket_info(info: WebSocketInfo) -> JSONObject:
    """Encode WebSocketInfo to JSON-serializable dict.

    Args:
        info: WebSocketInfo to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "url": info["url"],
        "request_id": info["request_id"],
        "timestamp_ms": info["timestamp_ms"],
    }
    return result


def decode_websocket_info(data: JSONObject) -> WebSocketInfo:
    """Decode WebSocketInfo from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated WebSocketInfo.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return WebSocketInfo(
        url=require_str(data, "url"),
        request_id=require_str(data, "request_id"),
        timestamp_ms=require_int(data, "timestamp_ms"),
    )


# =============================================================================
# Bot Configuration
# =============================================================================


class BotConfig(TypedDict):
    """Configuration for the TankpitBot.

    Attributes:
        ws_url: WebSocket URL to connect to.
        username: Tank username for the game.
        game_id: Game/map ID to join.
    """

    ws_url: str
    username: str
    game_id: str


def encode_bot_config(config: BotConfig) -> JSONObject:
    """Encode BotConfig to JSON-serializable dict.

    Args:
        config: BotConfig to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "ws_url": config["ws_url"],
        "username": config["username"],
        "game_id": config["game_id"],
    }
    return result


def decode_bot_config(data: JSONObject) -> BotConfig:
    """Decode BotConfig from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated BotConfig.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return BotConfig(
        ws_url=require_str(data, "ws_url"),
        username=require_str(data, "username"),
        game_id=require_str(data, "game_id"),
    )


# =============================================================================
# CDP WebSocket Frame Types
# =============================================================================


class CDPWebSocketFrame(TypedDict):
    """CDP WebSocketFrame object from Network domain.

    Attributes:
        opcode: WebSocket frame opcode (1=text, 2=binary).
        mask: Whether the frame is masked.
        payloadData: Frame payload. UTF-8 string for opcode 1, base64 for others.
    """

    opcode: int
    mask: bool
    payloadData: str


def encode_cdp_websocket_frame(frame: CDPWebSocketFrame) -> JSONObject:
    """Encode CDPWebSocketFrame to JSON-serializable dict.

    Args:
        frame: CDPWebSocketFrame to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "opcode": frame["opcode"],
        "mask": frame["mask"],
        "payloadData": frame["payloadData"],
    }
    return result


def decode_cdp_websocket_frame(data: JSONObject) -> CDPWebSocketFrame:
    """Decode CDPWebSocketFrame from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CDPWebSocketFrame.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return CDPWebSocketFrame(
        opcode=require_int(data, "opcode"),
        mask=require_bool(data, "mask"),
        payloadData=require_str(data, "payloadData"),
    )


class CDPWebSocketFrameEvent(TypedDict):
    """CDP event params for Network.webSocketFrameReceived/Sent.

    Attributes:
        requestId: Request identifier for the WebSocket connection.
        timestamp: Monotonic timestamp in seconds.
        response: The WebSocket frame data.
    """

    requestId: str
    timestamp: float
    response: CDPWebSocketFrame


def encode_cdp_websocket_frame_event(event: CDPWebSocketFrameEvent) -> JSONObject:
    """Encode CDPWebSocketFrameEvent to JSON-serializable dict.

    Args:
        event: CDPWebSocketFrameEvent to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "requestId": event["requestId"],
        "timestamp": event["timestamp"],
        "response": encode_cdp_websocket_frame(event["response"]),
    }
    return result


def decode_cdp_websocket_frame_event(data: JSONObject) -> CDPWebSocketFrameEvent:
    """Decode CDPWebSocketFrameEvent from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CDPWebSocketFrameEvent.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    response_val = data.get("response")
    if response_val is None:
        raise JSONTypeError("Missing required field 'response'")
    if not isinstance(response_val, dict):
        raise JSONTypeError(
            f"Field 'response' must be an object, got {type(response_val).__name__}"
        )
    return CDPWebSocketFrameEvent(
        requestId=require_str(data, "requestId"),
        timestamp=require_float(data, "timestamp"),
        response=decode_cdp_websocket_frame(response_val),
    )


class CDPWebSocketCreatedEvent(TypedDict):
    """CDP event params for Network.webSocketCreated.

    Attributes:
        requestId: Request identifier for the WebSocket connection.
        url: WebSocket URL.
    """

    requestId: str
    url: str


def encode_cdp_websocket_created_event(event: CDPWebSocketCreatedEvent) -> JSONObject:
    """Encode CDPWebSocketCreatedEvent to JSON-serializable dict.

    Args:
        event: CDPWebSocketCreatedEvent to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "requestId": event["requestId"],
        "url": event["url"],
    }
    return result


def decode_cdp_websocket_created_event(data: JSONObject) -> CDPWebSocketCreatedEvent:
    """Decode CDPWebSocketCreatedEvent from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CDPWebSocketCreatedEvent.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return CDPWebSocketCreatedEvent(
        requestId=require_str(data, "requestId"),
        url=require_str(data, "url"),
    )


# =============================================================================
# Probe Input Types
# =============================================================================


class KeyInput(TypedDict):
    """A keyboard input action.

    Attributes:
        key: Key name (e.g., 'w', 'ArrowUp', 'Space').
    """

    key: str


def encode_key_input(inp: KeyInput) -> JSONObject:
    """Encode KeyInput to JSON-serializable dict.

    Args:
        inp: KeyInput to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {"key": inp["key"]}
    return result


def decode_key_input(data: JSONObject) -> KeyInput:
    """Decode KeyInput from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated KeyInput.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return KeyInput(key=require_str(data, "key"))


class MouseInput(TypedDict):
    """A mouse input action.

    Attributes:
        x: X coordinate in pixels.
        y: Y coordinate in pixels.
        button: Mouse button used.
    """

    x: int
    y: int
    button: MouseButton


def encode_mouse_input(inp: MouseInput) -> JSONObject:
    """Encode MouseInput to JSON-serializable dict.

    Args:
        inp: MouseInput to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "x": inp["x"],
        "y": inp["y"],
        "button": inp["button"],
    }
    return result


def decode_mouse_input(data: JSONObject) -> MouseInput:
    """Decode MouseInput from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated MouseInput.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return MouseInput(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        button=_require_mouse_button(data, "button"),
    )


class ProbeInput(TypedDict):
    """A probe input action (key or mouse).

    Attributes:
        input_type: Type of input ('key' or 'mouse').
        key_input: Key input details (present if input_type is 'key').
        mouse_input: Mouse input details (present if input_type is 'mouse').
    """

    input_type: InputType
    key_input: KeyInput | None
    mouse_input: MouseInput | None


def encode_probe_input(inp: ProbeInput) -> JSONObject:
    """Encode ProbeInput to JSON-serializable dict.

    Args:
        inp: ProbeInput to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "input_type": inp["input_type"],
        "key_input": encode_key_input(inp["key_input"]) if inp["key_input"] else None,
        "mouse_input": encode_mouse_input(inp["mouse_input"]) if inp["mouse_input"] else None,
    }
    return result


def decode_probe_input(data: JSONObject) -> ProbeInput:
    """Decode ProbeInput from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ProbeInput.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    input_type = _require_input_type(data, "input_type")

    key_input: KeyInput | None = None
    key_input_raw = data.get("key_input")
    if key_input_raw is not None:
        if not isinstance(key_input_raw, dict):
            raise JSONTypeError(
                f"Field 'key_input' must be an object, got {type(key_input_raw).__name__}"
            )
        key_input = decode_key_input(key_input_raw)

    mouse_input: MouseInput | None = None
    mouse_input_raw = data.get("mouse_input")
    if mouse_input_raw is not None:
        if not isinstance(mouse_input_raw, dict):
            raise JSONTypeError(
                f"Field 'mouse_input' must be an object, got {type(mouse_input_raw).__name__}"
            )
        mouse_input = decode_mouse_input(mouse_input_raw)

    return ProbeInput(
        input_type=input_type,
        key_input=key_input,
        mouse_input=mouse_input,
    )


# =============================================================================
# Probe Result Types
# =============================================================================


class ProbeResult(TypedDict):
    """Result of a single probe input.

    Attributes:
        input: The input that was injected.
        timestamp_ms: When the input was injected.
        messages_before_count: Number of messages captured before this input.
        messages_after: Messages captured after this input (sent only).
    """

    input: ProbeInput
    timestamp_ms: int
    messages_before_count: int
    messages_after: list[CapturedMessage]


def encode_probe_result(result: ProbeResult) -> JSONObject:
    """Encode ProbeResult to JSON-serializable dict.

    Args:
        result: ProbeResult to encode.

    Returns:
        JSON-serializable dict representation.
    """
    encoded_messages: list[JSONValue] = [
        encode_captured_message(m) for m in result["messages_after"]
    ]
    out: JSONObject = {
        "input": encode_probe_input(result["input"]),
        "timestamp_ms": result["timestamp_ms"],
        "messages_before_count": result["messages_before_count"],
        "messages_after": encoded_messages,
    }
    return out


def decode_probe_result(data: JSONObject) -> ProbeResult:
    """Decode ProbeResult from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ProbeResult.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    input_raw = data.get("input")
    if input_raw is None:
        raise JSONTypeError("Missing required field 'input'")
    if not isinstance(input_raw, dict):
        raise JSONTypeError(f"Field 'input' must be an object, got {type(input_raw).__name__}")

    raw_messages = require_list(data, "messages_after")
    messages: list[CapturedMessage] = []
    for idx, raw_msg in enumerate(raw_messages):
        if not isinstance(raw_msg, dict):
            raise JSONTypeError(f"messages_after[{idx}] must be an object")
        messages.append(decode_captured_message(raw_msg))

    return ProbeResult(
        input=decode_probe_input(input_raw),
        timestamp_ms=require_int(data, "timestamp_ms"),
        messages_before_count=require_int(data, "messages_before_count"),
        messages_after=messages,
    )


class ProbeSession(TypedDict):
    """A complete protocol probe session.

    Attributes:
        session_id: Unique identifier for this probe session.
        start_timestamp_ms: Unix timestamp when probe started.
        end_timestamp_ms: Unix timestamp when probe ended.
        base_url: Base URL of the site being probed.
        results: List of probe results for each input.
    """

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    results: list[ProbeResult]


def encode_probe_session(session: ProbeSession) -> JSONObject:
    """Encode ProbeSession to JSON-serializable dict.

    Args:
        session: ProbeSession to encode.

    Returns:
        JSON-serializable dict representation.
    """
    encoded_results: list[JSONValue] = [encode_probe_result(r) for r in session["results"]]
    result: JSONObject = {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "results": encoded_results,
    }
    return result


def decode_probe_session(data: JSONObject) -> ProbeSession:
    """Decode ProbeSession from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ProbeSession.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    raw_results = require_list(data, "results")
    results: list[ProbeResult] = []
    for idx, raw_result in enumerate(raw_results):
        if not isinstance(raw_result, dict):
            raise JSONTypeError(f"results[{idx}] must be an object")
        results.append(decode_probe_result(raw_result))

    return ProbeSession(
        session_id=require_str(data, "session_id"),
        start_timestamp_ms=require_int(data, "start_timestamp_ms"),
        end_timestamp_ms=require_int(data, "end_timestamp_ms"),
        base_url=require_str(data, "base_url"),
        results=results,
    )


__all__ = [
    "BotConfig",
    "CDPWebSocketCreatedEvent",
    "CDPWebSocketFrame",
    "CDPWebSocketFrameEvent",
    "CaptureSession",
    "CapturedMessage",
    "InputType",
    "KeyInput",
    "MessageDirection",
    "MouseButton",
    "MouseInput",
    "ProbeInput",
    "ProbeResult",
    "ProbeSession",
    "SnifferConfig",
    "WebSocketInfo",
    "decode_bot_config",
    "decode_capture_session",
    "decode_captured_message",
    "decode_cdp_websocket_created_event",
    "decode_cdp_websocket_frame",
    "decode_cdp_websocket_frame_event",
    "decode_key_input",
    "decode_mouse_input",
    "decode_probe_input",
    "decode_probe_result",
    "decode_probe_session",
    "decode_sniffer_config",
    "decode_websocket_info",
    "encode_bot_config",
    "encode_capture_session",
    "encode_captured_message",
    "encode_cdp_websocket_created_event",
    "encode_cdp_websocket_frame",
    "encode_cdp_websocket_frame_event",
    "encode_key_input",
    "encode_mouse_input",
    "encode_probe_input",
    "encode_probe_result",
    "encode_probe_session",
    "encode_sniffer_config",
    "encode_websocket_info",
]
