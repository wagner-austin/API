"""Scan (radar) outcome emitters.

Three resolutions: the radar sweep completed, the action stalled out,
or the server rejected the command with a 0x52 Supervisor error.
"""

from __future__ import annotations

from tankpit_bot.ledger.outcome._emit import emit_action_outcome
from tankpit_bot.ledger.records import ActionOutcomeRecordDict
from tankpit_bot.ledger.service import LedgerService


def emit_scan_radar_complete(
    ledger: LedgerService, *, duration_ms: int, target_x: int, target_y: int
) -> ActionOutcomeRecordDict:
    """Record a completed radar sweep.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-completion wall-clock ms.
        target_x: Scan anchor X (the viewport tile the scan covered).
        target_y: Scan anchor Y.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="scan",
        outcome="radar_complete",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
    )


def emit_scan_stall_timeout(
    ledger: LedgerService, *, duration_ms: int, target_x: int, target_y: int, timeout_ms: int
) -> ActionOutcomeRecordDict:
    """Record a scan that stalled past its timeout.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-stall wall-clock ms.
        target_x: Scan anchor X.
        target_y: Scan anchor Y.
        timeout_ms: The stall threshold that fired.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="scan",
        outcome="stall_timeout",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        timeout_ms=timeout_ms,
    )


def emit_scan_command_rejected(
    ledger: LedgerService, *, duration_ms: int, target_x: int, target_y: int, error_code: int
) -> ActionOutcomeRecordDict:
    """Record a scan the server refused with a 0x52 Supervisor error.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-rejection wall-clock ms.
        target_x: Scan anchor X.
        target_y: Scan anchor Y.
        error_code: The 0x52 error code.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="scan",
        outcome="command_rejected",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        error_code=error_code,
    )


__all__ = [
    "emit_scan_command_rejected",
    "emit_scan_radar_complete",
    "emit_scan_stall_timeout",
]
