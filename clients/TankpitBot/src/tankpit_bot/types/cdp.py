"""CDP WebSocket frame types."""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_bool,
    require_float,
    require_int,
    require_str,
)


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


__all__ = [
    "CDPWebSocketCreatedEvent",
    "CDPWebSocketFrame",
    "CDPWebSocketFrameEvent",
    "decode_cdp_websocket_created_event",
    "decode_cdp_websocket_frame",
    "decode_cdp_websocket_frame_event",
    "encode_cdp_websocket_created_event",
    "encode_cdp_websocket_frame",
    "encode_cdp_websocket_frame_event",
]
