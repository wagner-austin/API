"""Map-open outcome emitters.

Three resolutions: MAP_DATA processed, stall timeout, 0x52 rejection.
Map opens have no target tile -- the payloads carry none (no
sentinels).
"""

from __future__ import annotations

from tankpit_bot.ledger.outcome._emit import emit_action_outcome
from tankpit_bot.ledger.records import ActionOutcomeRecordDict
from tankpit_bot.ledger.service import LedgerService


def emit_map_open_data_processed(
    ledger: LedgerService, *, duration_ms: int
) -> ActionOutcomeRecordDict:
    """Record a map open resolved by MAP_DATA arriving and decoding.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-processing wall-clock ms.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="map_open",
        outcome="map_data_processed",
        duration_ms=duration_ms,
    )


def emit_map_open_stall_timeout(
    ledger: LedgerService, *, duration_ms: int, timeout_ms: int
) -> ActionOutcomeRecordDict:
    """Record a map open that stalled past its timeout.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-stall wall-clock ms.
        timeout_ms: The stall threshold that fired.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="map_open",
        outcome="stall_timeout",
        duration_ms=duration_ms,
        timeout_ms=timeout_ms,
    )


def emit_map_open_command_rejected(
    ledger: LedgerService, *, duration_ms: int, error_code: int
) -> ActionOutcomeRecordDict:
    """Record a map open the server refused with a 0x52 error.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-rejection wall-clock ms.
        error_code: The 0x52 error code.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="map_open",
        outcome="command_rejected",
        duration_ms=duration_ms,
        error_code=error_code,
    )


__all__ = [
    "emit_map_open_command_rejected",
    "emit_map_open_data_processed",
    "emit_map_open_stall_timeout",
]
