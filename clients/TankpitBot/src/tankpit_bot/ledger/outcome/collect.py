"""Collect (container pickup) outcome emitters.

Seven resolutions: target tile reached, container consumed under the
bot, server movement rejection, 0x52 command rejection (tank full /
empty container / inventory full), stall timeout, and the executor's
two pickup-validation discards (container gone, fuel/equipment kind
mismatch) -- previously silent ``emit_ai`` lines (rejection-loop
audit instance #3).
"""

from __future__ import annotations

from tankpit_bot.ledger.outcome._emit import emit_action_outcome
from tankpit_bot.ledger.ring import ActionOutcomeRecordDict


def emit_collect_position_reached(
    *, duration_ms: int, target_x: int, target_y: int, landed_x: int, landed_y: int
) -> ActionOutcomeRecordDict:
    """Record a collection that reached the container's tile.

    Args:
        duration_ms: Dispatch-to-arrival wall-clock ms.
        target_x: Container X.
        target_y: Container Y.
        landed_x: Arrival X.
        landed_y: Arrival Y.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="collect",
        outcome="position_reached",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        landed_x=landed_x,
        landed_y=landed_y,
    )


def emit_collect_container_consumed(
    *, duration_ms: int, target_x: int, target_y: int, landed_x: int, landed_y: int
) -> ActionOutcomeRecordDict:
    """Record a collection resolved by the container vanishing.

    The container left the registry (picked up by us on an adjacent
    approach, or consumed by someone else) before the bot's tile
    matched the target.

    Args:
        duration_ms: Dispatch-to-consumption wall-clock ms.
        target_x: Container X.
        target_y: Container Y.
        landed_x: Bot X when the container vanished.
        landed_y: Bot Y likewise.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="collect",
        outcome="container_consumed",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        landed_x=landed_x,
        landed_y=landed_y,
    )


def emit_collect_movement_rejected(
    *, duration_ms: int, target_x: int, target_y: int
) -> ActionOutcomeRecordDict:
    """Record a collection walk the server rejected.

    Args:
        duration_ms: Dispatch-to-rejection wall-clock ms.
        target_x: Container X.
        target_y: Container Y.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="collect",
        outcome="movement_rejected",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
    )


def emit_collect_command_rejected(
    *, duration_ms: int, target_x: int, target_y: int, error_code: int
) -> ActionOutcomeRecordDict:
    """Record a collection the server genuinely refused with a 0x52 error.

    Since 2026-07-19 this covers only the true refusals (code 0
    geometry, code 1 can't-go); codes 4/5/7 resolve as their typed
    outcomes (``pickup_empty`` / ``clamped_transfer`` /
    ``inventory_full``).

    Args:
        duration_ms: Dispatch-to-rejection wall-clock ms.
        target_x: Container X.
        target_y: Container Y.
        error_code: The 0x52 error code.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="collect",
        outcome="command_rejected",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        error_code=error_code,
    )


def emit_collect_pickup_empty(
    *, duration_ms: int, target_x: int, target_y: int
) -> ActionOutcomeRecordDict:
    """Record a pickup that found the container drained (0x52 code 4).

    Environmental miss -- someone consumed the container between the
    planner's scan and the pickup. The belief at the target is removed
    by the same handler that emits this.

    Args:
        duration_ms: Dispatch-to-resolution wall-clock ms.
        target_x: Container X.
        target_y: Container Y.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="collect",
        outcome="pickup_empty",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
    )


def emit_collect_clamped_transfer(
    *, duration_ms: int, target_x: int, target_y: int
) -> ActionOutcomeRecordDict:
    """Record a fuel pickup clamped at the cap (0x52 code 5) -- a success.

    The server transferred ``min(volume, headroom)`` and kept the
    remainder in the container (the 0x43 partial-pickup message
    updates the container belief; the wire's absolute fuel carries the
    gain). The code 5 is the completion signal for a clamped
    transfer, not a refusal.

    Args:
        duration_ms: Dispatch-to-resolution wall-clock ms.
        target_x: Container X.
        target_y: Container Y.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="collect",
        outcome="clamped_transfer",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
    )


def emit_collect_inventory_full(
    *, duration_ms: int, target_x: int, target_y: int
) -> ActionOutcomeRecordDict:
    """Record an equipment pickup refused because ALL slots are full (code 7).

    The rejection is an authoritative absolute inventory statement;
    the handler reconciles every slot belief to capacity and keeps
    the container (fill-what's-empty mechanic, user 2026-07-18).

    Args:
        duration_ms: Dispatch-to-resolution wall-clock ms.
        target_x: Container X.
        target_y: Container Y.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="collect",
        outcome="inventory_full",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
    )


def emit_collect_stall_timeout(
    *, duration_ms: int, target_x: int, target_y: int, timeout_ms: int
) -> ActionOutcomeRecordDict:
    """Record a collection that stalled past its timeout.

    Args:
        duration_ms: Dispatch-to-stall wall-clock ms.
        target_x: Container X.
        target_y: Container Y.
        timeout_ms: The stall threshold that fired.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="collect",
        outcome="stall_timeout",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        timeout_ms=timeout_ms,
    )


def emit_collect_discarded_no_container(
    *, target_x: int, target_y: int, pickup_kind: str
) -> ActionOutcomeRecordDict:
    """Record an executor discard: the target container no longer exists.

    Args:
        target_x: Requested container X.
        target_y: Requested container Y.
        pickup_kind: Which pickup was attempted (``fuel``/``equipment``).

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="collect",
        outcome="discarded_no_container",
        duration_ms=0,
        target_x=target_x,
        target_y=target_y,
        pickup_kind=pickup_kind,
    )


def emit_collect_discarded_kind_mismatch(
    *, target_x: int, target_y: int, pickup_kind: str
) -> ActionOutcomeRecordDict:
    """Record an executor discard: tracked container is the other kind.

    Args:
        target_x: Requested container X.
        target_y: Requested container Y.
        pickup_kind: Which pickup was attempted (``fuel``/``equipment``);
            the tracked container is the opposite kind.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="collect",
        outcome="discarded_kind_mismatch",
        duration_ms=0,
        target_x=target_x,
        target_y=target_y,
        pickup_kind=pickup_kind,
    )


__all__ = [
    "emit_collect_clamped_transfer",
    "emit_collect_command_rejected",
    "emit_collect_container_consumed",
    "emit_collect_discarded_kind_mismatch",
    "emit_collect_discarded_no_container",
    "emit_collect_inventory_full",
    "emit_collect_movement_rejected",
    "emit_collect_pickup_empty",
    "emit_collect_stall_timeout",
]
