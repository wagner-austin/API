"""WebSocket message types."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from platform_core.json_utils import JSONObject, optional_str, require_int, require_str

from tankpit_bot.types.literals import (
    MessageDirection,
    SentFrameOrigin,
    require_message_direction,
    require_sent_frame_origin,
)


class CapturedMessage(TypedDict):
    """A captured WebSocket message.

    Attributes:
        timestamp_ms: Unix timestamp in milliseconds when message was captured.
        direction: Whether message was sent or received.
        payload: Raw message payload as string.
        ws_url: WebSocket URL this message was captured from.
        sent_origin: Source attribution for sent frames when available.
        sent_label: Bot-side label for injected sends when available.
        sent_stack: JavaScript send stack when available.
    """

    timestamp_ms: int
    direction: MessageDirection
    payload: str
    ws_url: str
    sent_origin: NotRequired[SentFrameOrigin]
    sent_label: NotRequired[str]
    sent_stack: NotRequired[str]


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
    if "sent_origin" in msg:
        result["sent_origin"] = msg["sent_origin"]
    if "sent_label" in msg:
        result["sent_label"] = msg["sent_label"]
    if "sent_stack" in msg:
        result["sent_stack"] = msg["sent_stack"]
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
    result = CapturedMessage(
        timestamp_ms=require_int(data, "timestamp_ms"),
        direction=require_message_direction(data, "direction"),
        payload=require_str(data, "payload"),
        ws_url=require_str(data, "ws_url"),
    )
    sent_origin = optional_str(data, "sent_origin")
    if sent_origin is not None:
        result["sent_origin"] = require_sent_frame_origin(
            {"sent_origin": sent_origin},
            "sent_origin",
        )
    sent_label = optional_str(data, "sent_label")
    if sent_label is not None:
        result["sent_label"] = sent_label
    sent_stack = optional_str(data, "sent_stack")
    if sent_stack is not None:
        result["sent_stack"] = sent_stack
    return result


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


__all__ = [
    "CapturedMessage",
    "WebSocketInfo",
    "decode_captured_message",
    "decode_websocket_info",
    "encode_captured_message",
    "encode_websocket_info",
]
