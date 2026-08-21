"""Scope (viewport pan) outcome emitters.

Two resolutions: the answering 0x5A confirmed the shifted window, or
the pan stalled out. Promoted from fire-and-forget 2026-08-20
([[viewport-shift-protocol]] scope-pending radar drop): an untracked
pan let the next tick dispatch radar or map_open into the window the
server silently drops commands in. The pan's direction rides the
``scope_shift_sent`` diagnostic and the paired decision, not the
outcome — mirroring map_open's targetless emitters.
"""

from __future__ import annotations

from tankpit_bot.ledger.outcome._emit import emit_action_outcome
from tankpit_bot.ledger.records import ActionOutcomeRecordDict
from tankpit_bot.ledger.service import LedgerService


def emit_scope_confirmed(ledger: LedgerService, *, duration_ms: int) -> ActionOutcomeRecordDict:
    """Record a pan whose 0x5A confirmation landed.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-confirmation wall-clock ms (median is
            one server tick — 759 archived pans, 2026-08-20 sweep).

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="scope",
        outcome="confirmed",
        duration_ms=duration_ms,
    )


def emit_scope_stall_timeout(
    ledger: LedgerService, *, duration_ms: int, timeout_ms: int
) -> ActionOutcomeRecordDict:
    """Record a pan that stalled past its timeout.

    Args:
        ledger: Session ledger receiving the outcome.
        duration_ms: Dispatch-to-stall wall-clock ms.
        timeout_ms: The stall threshold that fired.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        ledger,
        action_kind="scope",
        outcome="stall_timeout",
        duration_ms=duration_ms,
        timeout_ms=timeout_ms,
    )


__all__ = [
    "emit_scope_confirmed",
    "emit_scope_stall_timeout",
]
