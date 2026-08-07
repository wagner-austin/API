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
from tankpit_bot.ledger.records import ActionOutcomeRecordDict
from tankpit_bot.ledger.service import LedgerService


def emit_collect_position_reached(
    ledger: LedgerService,
    *,
    duration_ms: int,
    target_x: int,
    target_y: int,
    landed_x: int,
    landed_y: int,
) -> ActionOutcomeRecordDict:
    """Record a collection that reached the container's tile.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-arrival wall-clock ms.
        target_x: Container X.
        target_y: Container Y.
        landed_x: Arrival X.
        landed_y: Arrival Y.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="collect",
        outcome="position_reached",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        landed_x=landed_x,
        landed_y=landed_y,
    )


def emit_collect_container_consumed(
    ledger: LedgerService,
    *,
    duration_ms: int,
    target_x: int,
    target_y: int,
    landed_x: int,
    landed_y: int,
) -> ActionOutcomeRecordDict:
    """Record a collection resolved by the container vanishing.

    The container left the registry (picked up by us on an adjacent
    approach, or consumed by someone else) before the bot's tile
    matched the target.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-consumption wall-clock ms.
        target_x: Container X.
        target_y: Container Y.
        landed_x: Bot X when the container vanished.
        landed_y: Bot Y likewise.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="collect",
        outcome="container_consumed",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        landed_x=landed_x,
        landed_y=landed_y,
    )


def emit_collect_movement_rejected(
    ledger: LedgerService, *, duration_ms: int, target_x: int, target_y: int
) -> ActionOutcomeRecordDict:
    """Record a collection walk the server rejected.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-rejection wall-clock ms.
        target_x: Container X.
        target_y: Container Y.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="collect",
        outcome="movement_rejected",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
    )


def emit_collect_command_rejected(
    ledger: LedgerService, *, duration_ms: int, target_x: int, target_y: int, error_code: int
) -> ActionOutcomeRecordDict:
    """Record a collection the server genuinely refused with a 0x52 error.

    Since 2026-07-19 this covers only the true refusals (code 0
    geometry, code 1 can't-go); codes 4/5/7 resolve as their typed
    outcomes (``pickup_empty`` / ``clamped_transfer`` /
    ``inventory_full``).

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-rejection wall-clock ms.
        target_x: Container X.
        target_y: Container Y.
        error_code: The 0x52 error code.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="collect",
        outcome="command_rejected",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        error_code=error_code,
    )


def emit_collect_pickup_empty(
    ledger: LedgerService, *, duration_ms: int, target_x: int, target_y: int
) -> ActionOutcomeRecordDict:
    """Record a pickup that found the container drained (0x52 code 4).

    Environmental miss -- someone consumed the container between the
    planner's scan and the pickup. The belief at the target is removed
    by the same handler that emits this.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-resolution wall-clock ms.
        target_x: Container X.
        target_y: Container Y.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="collect",
        outcome="pickup_empty",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
    )


def emit_collect_clamped_transfer(
    ledger: LedgerService, *, duration_ms: int, target_x: int, target_y: int
) -> ActionOutcomeRecordDict:
    """Record a fuel pickup clamped at the cap (0x52 code 5) -- a success.

    The server transferred ``min(volume, headroom)`` and kept the
    remainder in the container (the 0x43 partial-pickup message
    updates the container belief; the wire's absolute fuel carries the
    gain). The code 5 is the completion signal for a clamped
    transfer, not a refusal.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-resolution wall-clock ms.
        target_x: Container X.
        target_y: Container Y.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="collect",
        outcome="clamped_transfer",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
    )


def emit_collect_inventory_full(
    ledger: LedgerService, *, duration_ms: int, target_x: int, target_y: int
) -> ActionOutcomeRecordDict:
    """Record an equipment pickup refused because ALL slots are full (code 7).

    The rejection is an authoritative absolute inventory statement;
    the handler reconciles every slot belief to capacity and keeps
    the container (fill-what's-empty mechanic, user 2026-07-18).

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-resolution wall-clock ms.
        target_x: Container X.
        target_y: Container Y.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="collect",
        outcome="inventory_full",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
    )


def emit_collect_stall_timeout(
    ledger: LedgerService, *, duration_ms: int, target_x: int, target_y: int, timeout_ms: int
) -> ActionOutcomeRecordDict:
    """Record a collection that stalled past its timeout.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-stall wall-clock ms.
        target_x: Container X.
        target_y: Container Y.
        timeout_ms: The stall threshold that fired.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="collect",
        outcome="stall_timeout",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        timeout_ms=timeout_ms,
    )


__all__ = [
    "emit_collect_clamped_transfer",
    "emit_collect_command_rejected",
    "emit_collect_container_consumed",
    "emit_collect_inventory_full",
    "emit_collect_movement_rejected",
    "emit_collect_pickup_empty",
    "emit_collect_stall_timeout",
]
