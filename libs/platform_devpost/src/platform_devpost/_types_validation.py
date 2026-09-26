"""platform_devpost Devpost literal types and shared validation helpers."""

from __future__ import annotations

from enum import StrEnum

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
)
from platform_core.members import as_member


class HackathonState(StrEnum):
    """Where a Devpost hackathon is in its calendar, as Devpost's API spells it."""

    OPEN = "open"
    UPCOMING = "upcoming"
    ENDED = "ended"
    SUBMISSIONS = "submissions"


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


def _require_state_value(value: JSONValue, context: str) -> HackathonState:
    """Require value to be a valid HackathonState.

    Args:
        value: JSON value to check.
        context: Context for error message.

    Returns:
        The member whose value is ``value``.

    Raises:
        JSONTypeError: If value is not a string, or names no state.
    """
    if not isinstance(value, str):
        raise JSONTypeError(f"{context} must be a string, got {type(value).__name__}")
    return as_member(value, context, HackathonState)
