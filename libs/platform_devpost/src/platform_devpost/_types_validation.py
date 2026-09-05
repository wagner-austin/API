"""platform_devpost Devpost literal types and shared validation helpers."""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_str,
)

HackathonState = Literal["open", "upcoming", "ended", "submissions"]


# -----------------------------------------------------------------------------
# Internal Validation Helpers
# -----------------------------------------------------------------------------


def _require_dict_value(value: JSONValue, context: str) -> JSONObject:
    """Require value to be a dict.

    Args:
        value: JSON value to check.
        context: Context for error message.

    Returns:
        The value as JSONObject.

    Raises:
        JSONTypeError: If value is not a dict.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{context} must be an object, got {type(value).__name__}")
    return value


def _require_state(obj: JSONObject, key: str) -> HackathonState:
    """Extract and validate HackathonState from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Field key.

    Returns:
        Validated HackathonState.

    Raises:
        JSONTypeError: If field is missing or not a valid state.
    """
    value = require_str(obj, key)
    if value == "open":
        return "open"
    if value == "upcoming":
        return "upcoming"
    if value == "ended":
        return "ended"
    if value == "submissions":
        return "submissions"
    raise JSONTypeError(f"Field '{key}' must be a valid state, got '{value}'")


def _require_state_value(value: JSONValue, context: str) -> HackathonState:
    """Require value to be a valid HackathonState.

    Args:
        value: JSON value to check.
        context: Context for error message.

    Returns:
        Validated HackathonState.

    Raises:
        JSONTypeError: If value is not a valid state.
    """
    if not isinstance(value, str):
        raise JSONTypeError(f"{context} must be a string, got {type(value).__name__}")
    if value == "open":
        return "open"
    if value == "upcoming":
        return "upcoming"
    if value == "ended":
        return "ended"
    if value == "submissions":
        return "submissions"
    raise JSONTypeError(f"{context} must be a valid state, got '{value}'")
