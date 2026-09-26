"""Top-level AI mode vocabulary and the mode/substate pairing rule.

This module owns the durable HFSM mode vocabulary so it does not get mixed
into the larger planner-state type module. Decoders narrow the words with
:func:`platform_core.members.require_member`.
"""

from __future__ import annotations

from enum import StrEnum


class AIMode(StrEnum):
    """The durable HFSM top-level mode."""

    UNSET = "UNSET"
    HUNT = "HUNT"
    COLLECT = "COLLECT"


class AIModeState(StrEnum):
    """The durable substate within an :class:`AIMode`; ``NONE`` belongs to UNSET."""

    NONE = ""
    ACQUIRE = "ACQUIRE"
    REFRESH = "REFRESH"
    CLOSE = "CLOSE"
    SCAN_ON_LANDING = "SCAN_ON_LANDING"
    ENGAGE = "ENGAGE"
    CONFIRM_KILL = "CONFIRM_KILL"
    SENSE = "SENSE"
    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    PICKUP = "PICKUP"
    DONE = "DONE"


HUNT_MODE_STATES: tuple[AIModeState, ...] = (
    AIModeState.ACQUIRE,
    AIModeState.REFRESH,
    AIModeState.CLOSE,
    AIModeState.SCAN_ON_LANDING,
    AIModeState.ENGAGE,
    AIModeState.CONFIRM_KILL,
)

COLLECT_MODE_STATES: tuple[AIModeState, ...] = (
    AIModeState.SENSE,
    AIModeState.SEARCH,
    AIModeState.APPROACH,
    AIModeState.PICKUP,
    AIModeState.DONE,
)


def is_valid_ai_mode_state(mode: AIMode, mode_state: AIModeState) -> bool:
    """Return True when the mode/substate pair is valid.

    Args:
        mode: Durable top-level mode.
        mode_state: Substate within that mode.

    Returns:
        True when the mode and substate are a valid pair.
    """
    if mode is AIMode.UNSET:
        return mode_state is AIModeState.NONE
    if mode is AIMode.HUNT:
        return mode_state in HUNT_MODE_STATES
    return mode_state in COLLECT_MODE_STATES


__all__ = [
    "COLLECT_MODE_STATES",
    "HUNT_MODE_STATES",
    "AIMode",
    "AIModeState",
    "is_valid_ai_mode_state",
]
