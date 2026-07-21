"""Move (walk) outcome emitters.

Five resolutions: target reached, server movement rejection ("you
can't go there"), 0x52 command rejection, stall timeout, and the
executor's hostile-mine discard -- previously a silent ``emit_ai``
line (rejection-loop audit instance #1's move-path sibling).
"""

from __future__ import annotations

from tankpit_bot.ledger.outcome._emit import emit_action_outcome
from tankpit_bot.ledger.ring import ActionOutcomeRecordDict


def emit_move_position_reached(
    *, duration_ms: int, target_x: int, target_y: int, landed_x: int, landed_y: int
) -> ActionOutcomeRecordDict:
    """Record a walk that reached its exact target tile.

    Args:
        duration_ms: Dispatch-to-arrival wall-clock ms.
        target_x: Requested X.
        target_y: Requested Y.
        landed_x: Arrival X (equals target for a reached walk).
        landed_y: Arrival Y.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="move",
        outcome="position_reached",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        landed_x=landed_x,
        landed_y=landed_y,
    )


def emit_move_movement_rejected(
    *, duration_ms: int, target_x: int, target_y: int
) -> ActionOutcomeRecordDict:
    """Record a walk the server rejected via the movement-failed path.

    Args:
        duration_ms: Dispatch-to-rejection wall-clock ms.
        target_x: Requested X.
        target_y: Requested Y.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="move",
        outcome="movement_rejected",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
    )


def emit_move_command_rejected(
    *, duration_ms: int, target_x: int, target_y: int, error_code: int
) -> ActionOutcomeRecordDict:
    """Record a walk the server refused with a 0x52 Supervisor error.

    Args:
        duration_ms: Dispatch-to-rejection wall-clock ms.
        target_x: Requested X.
        target_y: Requested Y.
        error_code: The 0x52 error code.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="move",
        outcome="command_rejected",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        error_code=error_code,
    )


def emit_move_stall_timeout(
    *, duration_ms: int, target_x: int, target_y: int, timeout_ms: int
) -> ActionOutcomeRecordDict:
    """Record a walk that stalled past its timeout.

    Args:
        duration_ms: Dispatch-to-stall wall-clock ms.
        target_x: Requested X.
        target_y: Requested Y.
        timeout_ms: The stall threshold that fired.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="move",
        outcome="stall_timeout",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        timeout_ms=timeout_ms,
    )


__all__ = [
    "emit_move_command_rejected",
    "emit_move_movement_rejected",
    "emit_move_position_reached",
    "emit_move_stall_timeout",
]
