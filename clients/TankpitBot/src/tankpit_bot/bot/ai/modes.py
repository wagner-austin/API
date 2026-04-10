"""Top-level AI mode literals and validation helpers.

This module owns the durable HFSM mode vocabulary so it does not get mixed
into the larger planner-state type module.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import JSONObject, require_str

AIMode = Literal[
    "UNSET",
    "HUNT",
    "RECOVER_FUEL",
    "RECOVER_EQUIPMENT",
]

AI_MODE_STATES: tuple[
    Literal[
        "",
        "ACQUIRE",
        "REFRESH",
        "CLOSE",
        "ENGAGE",
        "CONFIRM_KILL",
        "SENSE",
        "SEARCH",
        "APPROACH",
        "PICKUP",
        "DONE",
    ],
    ...,
] = (
    "",
    "ACQUIRE",
    "REFRESH",
    "CLOSE",
    "ENGAGE",
    "CONFIRM_KILL",
    "SENSE",
    "SEARCH",
    "APPROACH",
    "PICKUP",
    "DONE",
)

AIModeState = Literal[
    "",
    "ACQUIRE",
    "REFRESH",
    "CLOSE",
    "ENGAGE",
    "CONFIRM_KILL",
    "SENSE",
    "SEARCH",
    "APPROACH",
    "PICKUP",
    "DONE",
]

AI_MODES: tuple[AIMode, ...] = (
    "UNSET",
    "HUNT",
    "RECOVER_FUEL",
    "RECOVER_EQUIPMENT",
)

HUNT_MODE_STATES: tuple[AIModeState, ...] = (
    "ACQUIRE",
    "REFRESH",
    "CLOSE",
    "ENGAGE",
    "CONFIRM_KILL",
)

RECOVERY_MODE_STATES: tuple[AIModeState, ...] = (
    "SENSE",
    "SEARCH",
    "APPROACH",
    "PICKUP",
    "DONE",
)


def require_ai_mode(data: JSONObject, key: str) -> AIMode:
    """Validate and extract a durable AI mode from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated durable AI mode.

    Raises:
        ValueError: If the value is not a supported durable AI mode.
    """
    raw = require_str(data, key)
    for mode in AI_MODES:
        if raw == mode:
            return mode
    raise ValueError(f"{key} must be one of {AI_MODES}, got {raw!r}")


def require_ai_mode_state(data: JSONObject, key: str) -> AIModeState:
    """Validate and extract a durable AI substate from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated durable AI substate.

    Raises:
        ValueError: If the value is not a supported durable AI substate.
    """
    raw = require_str(data, key)
    for state in AI_MODE_STATES:
        if raw == state:
            return state
    raise ValueError(f"{key} must be one of {AI_MODE_STATES}, got {raw!r}")


def is_valid_ai_mode_state(mode: AIMode, mode_state: AIModeState) -> bool:
    """Return True when the mode/substate pair is valid.

    Args:
        mode: Durable top-level mode.
        mode_state: Substate within that mode.

    Returns:
        True when the mode and substate are a valid pair.
    """
    if mode == "UNSET":
        return mode_state == ""
    if mode == "HUNT":
        return mode_state in HUNT_MODE_STATES
    return mode_state in RECOVERY_MODE_STATES


__all__ = [
    "AI_MODES",
    "AI_MODE_STATES",
    "HUNT_MODE_STATES",
    "RECOVERY_MODE_STATES",
    "AIMode",
    "AIModeState",
    "is_valid_ai_mode_state",
    "require_ai_mode",
    "require_ai_mode_state",
]
