"""Literal types and validation helpers for TypedDict fields."""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import JSONObject, JSONTypeError, require_str

# =============================================================================
# Literal Types
# =============================================================================

MessageDirection = Literal["sent", "received"]
InputType = Literal["key", "mouse"]
MouseButton = Literal["left", "right", "middle"]


# =============================================================================
# Validation Helpers
# =============================================================================


def require_message_direction(obj: JSONObject, key: str) -> MessageDirection:
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


def require_input_type(obj: JSONObject, key: str) -> InputType:
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


def require_mouse_button(obj: JSONObject, key: str) -> MouseButton:
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
# Dict Conversion Helpers
# =============================================================================


def str_dict_to_json(source: dict[str, str]) -> JSONObject:
    """Convert dict[str, str] to JSONObject for type safety.

    Args:
        source: Dict with string keys and values.

    Returns:
        JSONObject with same contents.
    """
    result: JSONObject = {}
    for key, value in source.items():
        result[key] = value
    return result


def int_dict_to_json(source: dict[str, int]) -> JSONObject:
    """Convert dict[str, int] to JSONObject for type safety.

    Args:
        source: Dict with string keys and int values.

    Returns:
        JSONObject with same contents.
    """
    result: JSONObject = {}
    for key, value in source.items():
        result[key] = value
    return result


def mixed_dict_to_json(source: dict[str, int | str]) -> JSONObject:
    """Convert dict[str, int | str] to JSONObject for type safety.

    Args:
        source: Dict with string keys and int or str values.

    Returns:
        JSONObject with same contents.
    """
    result: JSONObject = {}
    for key, value in source.items():
        result[key] = value
    return result


__all__ = [
    "InputType",
    "MessageDirection",
    "MouseButton",
    "int_dict_to_json",
    "mixed_dict_to_json",
    "require_input_type",
    "require_message_direction",
    "require_mouse_button",
    "str_dict_to_json",
]
