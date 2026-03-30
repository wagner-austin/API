"""Bot state machine implementation.

This module provides a type-safe state machine for bot behavior.
States are explicit, transitions are validated, and all state
changes go through a central dispatch mechanism.

Design principles:
- All states are explicit enum values
- Transitions are validated at runtime
- State data is immutable (new state created on change)
- In-flight actions are tracked via a single InFlightActionDict
- Events from game messages trigger state transitions
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    require_int,
    require_str,
)
from typing_extensions import TypedDict


class BotState(Enum):
    """Bot state machine states.

    Each state represents a distinct bot behavior mode.
    Transitions between states are controlled by the state machine.
    """

    # Initial state before game entry
    INITIALIZING = auto()

    # Connected but waiting for game data
    WAITING_FOR_POSITION = auto()

    # Idle, ready to take action
    IDLE = auto()

    # Scanning with radar
    SCANNING = auto()

    # Walking to a target position
    MOVING = auto()

    # Teleport in progress, waiting for server landing confirmation
    TELEPORTING = auto()

    # Moving to pick up a container
    COLLECTING = auto()

    # Engaging in combat
    COMBAT = auto()

    # Low fuel, seeking fuel containers
    LOW_FUEL = auto()

    # Disconnected or error state
    DISCONNECTED = auto()


# Type alias for state names (for TypedDict usage)
StateName = Literal[
    "INITIALIZING",
    "WAITING_FOR_POSITION",
    "IDLE",
    "SCANNING",
    "MOVING",
    "TELEPORTING",
    "COLLECTING",
    "COMBAT",
    "LOW_FUEL",
    "DISCONNECTED",
]


# =============================================================================
# InFlightActionDict — authoritative action lifecycle record
# =============================================================================

ActionKind = Literal[
    "none",
    "move",
    "collect",
    "teleport",
    "scan",
    "shoot",
    "map_open",
]

ACTION_KINDS: tuple[ActionKind, ...] = (
    "none",
    "move",
    "collect",
    "teleport",
    "scan",
    "shoot",
    "map_open",
)

ActionOutcome = Literal[
    "pending",
    "confirmed",
    "timed_out",
    "failed",
]

ACTION_OUTCOMES: tuple[ActionOutcome, ...] = (
    "pending",
    "confirmed",
    "timed_out",
    "failed",
)


class InFlightActionDict(TypedDict):
    """Authoritative record of the current in-flight command.

    This is the single source of truth for what the bot is doing
    right now. Every field that was previously scattered across
    target_x, target_y, scan_pending, and last_action_ms is now
    consolidated here with an explicit lifecycle outcome.

    Attributes:
        kind: What type of action is in flight.
        target_x: Target X coordinate for the action.
        target_y: Target Y coordinate for the action.
        started_ms: Timestamp when the action was dispatched.
        outcome: Current lifecycle state of the action.
    """

    kind: ActionKind
    target_x: int
    target_y: int
    started_ms: int
    outcome: ActionOutcome


def make_no_action() -> InFlightActionDict:
    """Create an action record representing no in-flight action.

    Returns:
        InFlightActionDict with kind="none" and outcome="confirmed".
    """
    return InFlightActionDict(
        kind="none",
        target_x=0,
        target_y=0,
        started_ms=0,
        outcome="confirmed",
    )


def make_in_flight_action(
    kind: ActionKind,
    target_x: int,
    target_y: int,
    started_ms: int,
) -> InFlightActionDict:
    """Create a pending in-flight action record.

    Args:
        kind: Type of action being dispatched.
        target_x: Target X coordinate.
        target_y: Target Y coordinate.
        started_ms: Current timestamp in milliseconds.

    Returns:
        InFlightActionDict with outcome="pending".
    """
    return InFlightActionDict(
        kind=kind,
        target_x=target_x,
        target_y=target_y,
        started_ms=started_ms,
        outcome="pending",
    )


def _require_action_kind(data: JSONObject, key: str) -> ActionKind:
    """Validate and extract an ActionKind from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated ActionKind value.

    Raises:
        ValueError: If value is not a valid ActionKind.
    """
    raw = require_str(data, key)
    for kind in ACTION_KINDS:
        if raw == kind:
            return kind
    raise ValueError(f"{key} must be one of {ACTION_KINDS}, got {raw!r}")


def _require_action_outcome(
    data: JSONObject,
    key: str,
) -> ActionOutcome:
    """Validate and extract an ActionOutcome from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated ActionOutcome value.

    Raises:
        ValueError: If value is not a valid ActionOutcome.
    """
    raw = require_str(data, key)
    for outcome in ACTION_OUTCOMES:
        if raw == outcome:
            return outcome
    raise ValueError(
        f"{key} must be one of {ACTION_OUTCOMES}, got {raw!r}",
    )


def encode_in_flight_action(
    action: InFlightActionDict,
) -> JSONObject:
    """Encode InFlightActionDict to JSON-serializable dict.

    Args:
        action: InFlightActionDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "kind": action["kind"],
        "target_x": action["target_x"],
        "target_y": action["target_y"],
        "started_ms": action["started_ms"],
        "outcome": action["outcome"],
    }


def decode_in_flight_action(data: JSONObject) -> InFlightActionDict:
    """Decode InFlightActionDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated InFlightActionDict.

    Raises:
        ValueError: If kind or outcome is invalid.
        JSONTypeError: If required fields are missing or invalid.
    """
    return InFlightActionDict(
        kind=_require_action_kind(data, "kind"),
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        started_ms=require_int(data, "started_ms"),
        outcome=_require_action_outcome(data, "outcome"),
    )


# =============================================================================
# BotStateDataDict
# =============================================================================


class BotStateDataDict(TypedDict):
    """Immutable state data for the bot state machine.

    Attributes:
        state: Current bot state.
        fuel_threshold: Fuel level that triggers LOW_FUEL state.
        in_flight_action: Authoritative record of the current
            in-flight command, including target, timing, and
            lifecycle outcome.
    """

    state: StateName
    fuel_threshold: int
    in_flight_action: InFlightActionDict


def make_initial_state_data() -> BotStateDataDict:
    """Create initial state data for a new bot.

    Returns:
        BotStateDataDict with INITIALIZING state and no action.
    """
    return BotStateDataDict(
        state="INITIALIZING",
        fuel_threshold=200,
        in_flight_action=make_no_action(),
    )


def transition_to(
    current: BotStateDataDict,
    new_state: StateName,
    *,
    in_flight_action: InFlightActionDict | None = None,
) -> BotStateDataDict:
    """Create new state data with updated state and action.

    This is the ONLY way to change state - ensures immutability.

    Args:
        current: Current state data.
        new_state: New state to transition to.
        in_flight_action: New action record. If None, inherits the
            current action (useful for state changes that don't
            start a new action, like LOW_FUEL transitions).

    Returns:
        New BotStateDataDict with updated values.
    """
    return BotStateDataDict(
        state=new_state,
        fuel_threshold=current["fuel_threshold"],
        in_flight_action=(
            in_flight_action if in_flight_action is not None else current["in_flight_action"]
        ),
    )


def set_fuel_threshold(
    current: BotStateDataDict,
    threshold: int,
) -> BotStateDataDict:
    """Update fuel threshold without changing state.

    Args:
        current: Current state data.
        threshold: New fuel threshold.

    Returns:
        New BotStateDataDict with updated threshold.
    """
    return BotStateDataDict(
        state=current["state"],
        fuel_threshold=threshold,
        in_flight_action=current["in_flight_action"],
    )


# Valid state transitions - maps current state to allowed next states
VALID_TRANSITIONS: dict[StateName, frozenset[StateName]] = {
    "INITIALIZING": frozenset({"WAITING_FOR_POSITION", "DISCONNECTED"}),
    "WAITING_FOR_POSITION": frozenset({"IDLE", "DISCONNECTED"}),
    "IDLE": frozenset(
        {
            "IDLE",
            "SCANNING",
            "MOVING",
            "TELEPORTING",
            "COLLECTING",
            "COMBAT",
            "LOW_FUEL",
            "DISCONNECTED",
        },
    ),
    "SCANNING": frozenset(
        {"IDLE", "MOVING", "TELEPORTING", "COLLECTING", "COMBAT", "LOW_FUEL", "DISCONNECTED"},
    ),
    "MOVING": frozenset(
        {"IDLE", "SCANNING", "TELEPORTING", "COLLECTING", "COMBAT", "LOW_FUEL", "DISCONNECTED"},
    ),
    "TELEPORTING": frozenset(
        {"IDLE", "SCANNING", "MOVING", "COLLECTING", "COMBAT", "LOW_FUEL", "DISCONNECTED"},
    ),
    "COLLECTING": frozenset(
        {"IDLE", "SCANNING", "MOVING", "TELEPORTING", "COMBAT", "LOW_FUEL", "DISCONNECTED"},
    ),
    "COMBAT": frozenset(
        {"IDLE", "SCANNING", "MOVING", "TELEPORTING", "COLLECTING", "LOW_FUEL", "DISCONNECTED"},
    ),
    "LOW_FUEL": frozenset(
        {"IDLE", "SCANNING", "MOVING", "TELEPORTING", "COLLECTING", "COMBAT", "DISCONNECTED"},
    ),
    "DISCONNECTED": frozenset({"INITIALIZING"}),
}


def is_valid_transition(from_state: StateName, to_state: StateName) -> bool:
    """Check if a state transition is valid.

    Args:
        from_state: Current state.
        to_state: Desired next state.

    Returns:
        True if transition is allowed.
    """
    allowed = VALID_TRANSITIONS.get(from_state, frozenset())
    return to_state in allowed


def validate_transition(from_state: StateName, to_state: StateName) -> None:
    """Validate a state transition, raising if invalid.

    Args:
        from_state: Current state.
        to_state: Desired next state.

    Raises:
        ValueError: If transition is not allowed.
    """
    if not is_valid_transition(from_state, to_state):
        allowed = VALID_TRANSITIONS.get(from_state, frozenset())
        raise ValueError(
            f"Invalid transition from {from_state} to {to_state}. Allowed: {sorted(allowed)}"
        )


__all__ = [
    "ACTION_KINDS",
    "ACTION_OUTCOMES",
    "VALID_TRANSITIONS",
    "ActionKind",
    "ActionOutcome",
    "BotState",
    "BotStateDataDict",
    "InFlightActionDict",
    "StateName",
    "decode_in_flight_action",
    "encode_in_flight_action",
    "is_valid_transition",
    "make_in_flight_action",
    "make_initial_state_data",
    "make_no_action",
    "set_fuel_threshold",
    "transition_to",
    "validate_transition",
]
