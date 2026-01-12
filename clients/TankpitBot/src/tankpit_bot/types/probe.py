"""Probe input and result types."""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot.types.literals import (
    InputType,
    MouseButton,
    require_input_type,
    require_mouse_button,
)
from tankpit_bot.types.message import (
    CapturedMessage,
    decode_captured_message,
    encode_captured_message,
)


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
        button=require_mouse_button(data, "button"),
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
    input_type = require_input_type(data, "input_type")

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
    "KeyInput",
    "MouseInput",
    "ProbeInput",
    "ProbeResult",
    "ProbeSession",
    "decode_key_input",
    "decode_mouse_input",
    "decode_probe_input",
    "decode_probe_result",
    "decode_probe_session",
    "encode_key_input",
    "encode_mouse_input",
    "encode_probe_input",
    "encode_probe_result",
    "encode_probe_session",
]
