"""First-class mode-transition events (blind spot #11).

Every AI mode flip (``COLLECT`` <-> ``HUNT`` <-> ``UNSET``) becomes a
structured ``mode_transition`` event with its own event id, the
decision that drove it (``caused_by``), and the decision's typed
reason -- replacing the free-text ``emit_ai`` narration that made
transition history unqueryable.
"""

from __future__ import annotations

from tankpit_bot.ledger.records import ModeTransitionRecordDict
from tankpit_bot.ledger.service import LedgerService
from tankpit_bot.runtime_logging import emit_diagnostic


def emit_mode_transition(
    ledger: LedgerService,
    *,
    from_mode: str,
    to_mode: str,
    reason_kind: str,
    caused_by: int,
) -> ModeTransitionRecordDict:
    """Record one AI mode flip as a first-class event.

    Args:
        ledger: Session ledger receiving the transition.
        from_mode: Mode before the flip.
        to_mode: Mode after the flip.
        reason_kind: Typed reason of the driving decision.
        caused_by: Driving decision's event id (0 when none recorded).

    Returns:
        The recorded transition.
    """
    record = ModeTransitionRecordDict(
        event_id=ledger.next_event_id(),
        from_mode=from_mode,
        to_mode=to_mode,
        reason_kind=reason_kind,
        caused_by=caused_by,
    )
    ledger.transitions.append(record)
    emit_diagnostic(
        diagnostic_kind="mode_transition",
        event_id=record["event_id"],
        from_mode=from_mode,
        to_mode=to_mode,
        reason_kind=reason_kind,
        caused_by=caused_by,
    )
    return record


def mode_transitions(ledger: LedgerService) -> list[ModeTransitionRecordDict]:
    """Return every mode transition recorded this session, in order.

    Args:
        ledger: Session ledger owning the transition log.

    Returns:
        Transition records, oldest first.
    """
    return list(ledger.transitions)


__all__ = [
    "emit_mode_transition",
    "mode_transitions",
]
