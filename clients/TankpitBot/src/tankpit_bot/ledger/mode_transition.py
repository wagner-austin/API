"""First-class mode-transition events (blind spot #11).

Every AI mode flip (``COLLECT`` <-> ``HUNT`` <-> ``UNSET``) becomes a
structured ``mode_transition`` event with its own event id, the
decision that drove it (``caused_by``), and the decision's typed
reason -- replacing the free-text ``emit_ai`` narration that made
transition history unqueryable.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.ledger.events import next_event_id
from tankpit_bot.runtime_logging import emit_diagnostic


class ModeTransitionRecordDict(TypedDict):
    """One recorded mode transition.

    Attributes:
        event_id: Process-wide monotonic event id.
        from_mode: Mode before the flip.
        to_mode: Mode after the flip.
        reason_kind: Typed reason of the decision that flipped it.
        caused_by: Event id of the recorded decision driving the flip;
            ``0`` for flips driven by non-dispatching decisions (the
            manual-hold override records no decision).
    """

    event_id: int
    from_mode: str
    to_mode: str
    reason_kind: str
    caused_by: int


_transitions: list[ModeTransitionRecordDict] = []


def emit_mode_transition(
    *,
    from_mode: str,
    to_mode: str,
    reason_kind: str,
    caused_by: int,
) -> ModeTransitionRecordDict:
    """Record one AI mode flip as a first-class event.

    Args:
        from_mode: Mode before the flip.
        to_mode: Mode after the flip.
        reason_kind: Typed reason of the driving decision.
        caused_by: Driving decision's event id (0 when none recorded).

    Returns:
        The recorded transition.
    """
    record = ModeTransitionRecordDict(
        event_id=next_event_id(),
        from_mode=from_mode,
        to_mode=to_mode,
        reason_kind=reason_kind,
        caused_by=caused_by,
    )
    _transitions.append(record)
    emit_diagnostic(
        diagnostic_kind="mode_transition",
        event_id=record["event_id"],
        from_mode=from_mode,
        to_mode=to_mode,
        reason_kind=reason_kind,
        caused_by=caused_by,
    )
    return record


def mode_transitions() -> list[ModeTransitionRecordDict]:
    """Return every mode transition recorded this session, in order.

    Returns:
        Transition records, oldest first.
    """
    return list(_transitions)


def reset_mode_transitions() -> None:
    """Clear the transition log. Called from test-isolation fixtures."""
    _transitions.clear()


__all__ = [
    "ModeTransitionRecordDict",
    "emit_mode_transition",
    "mode_transitions",
    "reset_mode_transitions",
]
