"""Bot state machine implementation.

This module provides a type-safe state machine for bot behavior.
States are explicit, transitions are validated, and all state
changes go through a central dispatch mechanism.

Design principles:
- All states are explicit enum values
- Transitions are validated at runtime
- State data is immutable (new state created on change)
- Events from game messages trigger state transitions
- Actions are dispatched based on current state
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Literal

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

    # Moving to a target position
    MOVING = auto()

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
    "COLLECTING",
    "COMBAT",
    "LOW_FUEL",
    "DISCONNECTED",
]


class BotStateDataDict(TypedDict):
    """Immutable state data for the bot state machine.

    Attributes:
        state: Current bot state.
        target_x: Target X coordinate (for MOVING/COLLECTING states).
        target_y: Target Y coordinate (for MOVING/COLLECTING states).
        fuel_threshold: Fuel level that triggers LOW_FUEL state.
        scan_pending: Whether a radar scan result is pending.
        last_action_ms: Timestamp of last action taken.
    """

    state: StateName
    target_x: int
    target_y: int
    fuel_threshold: int
    scan_pending: bool
    last_action_ms: int


def make_initial_state_data() -> BotStateDataDict:
    """Create initial state data for a new bot.

    Returns:
        BotStateDataDict with INITIALIZING state and default values.
    """
    return BotStateDataDict(
        state="INITIALIZING",
        target_x=0,
        target_y=0,
        fuel_threshold=200,
        scan_pending=False,
        last_action_ms=0,
    )


def transition_to(
    current: BotStateDataDict,
    new_state: StateName,
    *,
    target_x: int | None = None,
    target_y: int | None = None,
    scan_pending: bool | None = None,
    last_action_ms: int | None = None,
) -> BotStateDataDict:
    """Create new state data with updated state and optional fields.

    This is the ONLY way to change state - ensures immutability.

    Args:
        current: Current state data.
        new_state: New state to transition to.
        target_x: New target X coordinate (optional).
        target_y: New target Y coordinate (optional).
        scan_pending: New scan pending flag (optional).
        last_action_ms: New last action timestamp (optional).

    Returns:
        New BotStateDataDict with updated values.
    """
    return BotStateDataDict(
        state=new_state,
        target_x=target_x if target_x is not None else current["target_x"],
        target_y=target_y if target_y is not None else current["target_y"],
        fuel_threshold=current["fuel_threshold"],
        scan_pending=scan_pending if scan_pending is not None else current["scan_pending"],
        last_action_ms=last_action_ms if last_action_ms is not None else current["last_action_ms"],
    )


def set_target(
    current: BotStateDataDict,
    target_x: int,
    target_y: int,
) -> BotStateDataDict:
    """Update target coordinates without changing state.

    Args:
        current: Current state data.
        target_x: New target X coordinate.
        target_y: New target Y coordinate.

    Returns:
        New BotStateDataDict with updated target.
    """
    return BotStateDataDict(
        state=current["state"],
        target_x=target_x,
        target_y=target_y,
        fuel_threshold=current["fuel_threshold"],
        scan_pending=current["scan_pending"],
        last_action_ms=current["last_action_ms"],
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
        target_x=current["target_x"],
        target_y=current["target_y"],
        fuel_threshold=threshold,
        scan_pending=current["scan_pending"],
        last_action_ms=current["last_action_ms"],
    )


# Valid state transitions - maps current state to allowed next states
VALID_TRANSITIONS: dict[StateName, frozenset[StateName]] = {
    "INITIALIZING": frozenset({"WAITING_FOR_POSITION", "DISCONNECTED"}),
    "WAITING_FOR_POSITION": frozenset({"IDLE", "DISCONNECTED"}),
    "IDLE": frozenset({"SCANNING", "MOVING", "COLLECTING", "COMBAT", "LOW_FUEL", "DISCONNECTED"}),
    "SCANNING": frozenset({"IDLE", "MOVING", "COLLECTING", "COMBAT", "LOW_FUEL", "DISCONNECTED"}),
    "MOVING": frozenset({"IDLE", "SCANNING", "COLLECTING", "COMBAT", "LOW_FUEL", "DISCONNECTED"}),
    "COLLECTING": frozenset({"IDLE", "SCANNING", "MOVING", "COMBAT", "LOW_FUEL", "DISCONNECTED"}),
    "COMBAT": frozenset({"IDLE", "SCANNING", "MOVING", "COLLECTING", "LOW_FUEL", "DISCONNECTED"}),
    "LOW_FUEL": frozenset({"IDLE", "SCANNING", "MOVING", "COLLECTING", "DISCONNECTED"}),
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
    "VALID_TRANSITIONS",
    "BotState",
    "BotStateDataDict",
    "StateName",
    "is_valid_transition",
    "make_initial_state_data",
    "set_fuel_threshold",
    "set_target",
    "transition_to",
    "validate_transition",
]
