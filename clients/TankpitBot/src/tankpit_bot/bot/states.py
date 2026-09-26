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

from enum import StrEnum

from platform_core.json_utils import JSONObject, require_int
from platform_core.members import require_member
from typing_extensions import TypedDict


class BotState(StrEnum):
    """Bot state machine states, each carrying its own name as its word.

    Each state represents a distinct bot behavior mode.
    Transitions between states are controlled by the state machine.
    """

    # Initial state before game entry
    INITIALIZING = "INITIALIZING"

    # Connected but waiting for game data
    WAITING_FOR_POSITION = "WAITING_FOR_POSITION"

    # Idle, ready to take action
    IDLE = "IDLE"

    # Scanning with radar
    SCANNING = "SCANNING"

    # Walking to a target position
    MOVING = "MOVING"

    # Teleport in progress, waiting for server landing confirmation
    TELEPORTING = "TELEPORTING"

    # Moving to pick up a container
    COLLECTING = "COLLECTING"

    # Engaging in combat
    COMBAT = "COMBAT"

    # Low fuel, seeking fuel containers
    LOW_FUEL = "LOW_FUEL"

    # Disconnected or error state
    DISCONNECTED = "DISCONNECTED"


# =============================================================================
# InFlightActionDict — authoritative action lifecycle record
# =============================================================================


class ActionKind(StrEnum):
    """What type of command is in flight; ``NONE`` is the idle sentinel.

    Wider than :class:`tankpit_bot.ledger.events.ActionKind`, which
    records only what the bot DID and so has no ``NONE``.
    """

    NONE = "none"
    MOVE = "move"
    COLLECT = "collect"
    TELEPORT = "teleport"
    SCAN = "scan"
    SHOOT = "shoot"
    MAP_OPEN = "map_open"
    SCOPE = "scope"


class ActionOutcome(StrEnum):
    """Lifecycle state of the in-flight command."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


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
        InFlightActionDict with kind NONE and outcome CONFIRMED.
    """
    return InFlightActionDict(
        kind=ActionKind.NONE,
        target_x=0,
        target_y=0,
        started_ms=0,
        outcome=ActionOutcome.CONFIRMED,
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
        InFlightActionDict with outcome PENDING.
    """
    return InFlightActionDict(
        kind=kind,
        target_x=target_x,
        target_y=target_y,
        started_ms=started_ms,
        outcome=ActionOutcome.PENDING,
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
        "kind": action["kind"].value,
        "target_x": action["target_x"],
        "target_y": action["target_y"],
        "started_ms": action["started_ms"],
        "outcome": action["outcome"].value,
    }


def decode_in_flight_action(data: JSONObject) -> InFlightActionDict:
    """Decode InFlightActionDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated InFlightActionDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid, or kind
            or outcome is a word outside its vocabulary.
    """
    return InFlightActionDict(
        kind=require_member(data, "kind", ActionKind),
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        started_ms=require_int(data, "started_ms"),
        outcome=require_member(data, "outcome", ActionOutcome),
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

    state: BotState
    fuel_threshold: int
    in_flight_action: InFlightActionDict


def make_initial_state_data() -> BotStateDataDict:
    """Create initial state data for a new bot.

    Returns:
        BotStateDataDict with INITIALIZING state and no action.
    """
    return BotStateDataDict(
        state=BotState.INITIALIZING,
        fuel_threshold=200,
        in_flight_action=make_no_action(),
    )


def transition_to(
    current: BotStateDataDict,
    new_state: BotState,
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
VALID_TRANSITIONS: dict[BotState, frozenset[BotState]] = {
    BotState.INITIALIZING: frozenset({BotState.WAITING_FOR_POSITION, BotState.DISCONNECTED}),
    BotState.WAITING_FOR_POSITION: frozenset({BotState.IDLE, BotState.DISCONNECTED}),
    BotState.IDLE: frozenset(
        {
            BotState.IDLE,
            BotState.SCANNING,
            BotState.MOVING,
            BotState.TELEPORTING,
            BotState.COLLECTING,
            BotState.COMBAT,
            BotState.LOW_FUEL,
            BotState.DISCONNECTED,
        },
    ),
    BotState.SCANNING: frozenset(
        {
            BotState.IDLE,
            BotState.MOVING,
            BotState.TELEPORTING,
            BotState.COLLECTING,
            BotState.COMBAT,
            BotState.LOW_FUEL,
            BotState.DISCONNECTED,
        },
    ),
    BotState.MOVING: frozenset(
        {
            BotState.IDLE,
            BotState.SCANNING,
            BotState.TELEPORTING,
            BotState.COLLECTING,
            BotState.COMBAT,
            BotState.LOW_FUEL,
            BotState.DISCONNECTED,
        },
    ),
    BotState.TELEPORTING: frozenset(
        {
            BotState.IDLE,
            BotState.SCANNING,
            BotState.MOVING,
            BotState.COLLECTING,
            BotState.COMBAT,
            BotState.LOW_FUEL,
            BotState.DISCONNECTED,
        },
    ),
    BotState.COLLECTING: frozenset(
        {
            BotState.IDLE,
            BotState.SCANNING,
            BotState.MOVING,
            BotState.TELEPORTING,
            BotState.COMBAT,
            BotState.LOW_FUEL,
            BotState.DISCONNECTED,
        },
    ),
    BotState.COMBAT: frozenset(
        {
            BotState.IDLE,
            BotState.SCANNING,
            BotState.MOVING,
            BotState.TELEPORTING,
            BotState.COLLECTING,
            BotState.LOW_FUEL,
            BotState.DISCONNECTED,
        },
    ),
    BotState.LOW_FUEL: frozenset(
        {
            BotState.IDLE,
            BotState.SCANNING,
            BotState.MOVING,
            BotState.TELEPORTING,
            BotState.COLLECTING,
            BotState.COMBAT,
            BotState.DISCONNECTED,
        },
    ),
    BotState.DISCONNECTED: frozenset({BotState.INITIALIZING}),
}


def is_valid_transition(from_state: BotState, to_state: BotState) -> bool:
    """Check if a state transition is valid.

    Args:
        from_state: Current state.
        to_state: Desired next state.

    Returns:
        True if transition is allowed.
    """
    return to_state in VALID_TRANSITIONS[from_state]


def validate_transition(from_state: BotState, to_state: BotState) -> None:
    """Validate a state transition, raising if invalid.

    Args:
        from_state: Current state.
        to_state: Desired next state.

    Raises:
        ValueError: If transition is not allowed.
    """
    if not is_valid_transition(from_state, to_state):
        allowed = VALID_TRANSITIONS[from_state]
        raise ValueError(
            f"Invalid transition from {from_state.value} to {to_state.value}."
            f" Allowed: {sorted(state.value for state in allowed)}"
        )


__all__ = [
    "VALID_TRANSITIONS",
    "ActionKind",
    "ActionOutcome",
    "BotState",
    "BotStateDataDict",
    "InFlightActionDict",
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
