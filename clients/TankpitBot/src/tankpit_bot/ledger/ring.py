"""Bounded per-kind ring of action-outcome records.

Every emitted outcome is appended to its kind's ring (bounded at
:data:`RING_CAPACITY`), giving planners an in-session queryable view
of recent attempts -- ``recent_outcomes("shoot", 5)`` replaces
implicit state inference from scattered counters.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.ledger.events import ACTION_KINDS, ActionKind
from tankpit_bot.ledger.outcomes import ActionOutcome

RING_CAPACITY = 128


class ActionOutcomeRecordDict(TypedDict):
    """One recorded action outcome.

    Attributes:
        event_id: Process-wide monotonic event id.
        attempt_id: Per-kind monotonic attempt counter.
        action_kind: Which action kind this outcome belongs to.
        outcome: Outcome label from the kind's outcome union.
        duration_ms: Wall-clock ms from dispatch to resolution.
            ``-1`` when the resolution gate fired with no recorded
            dispatch time (the historical wire-complete convention --
            reviewers spot the case by the negative value).
        caused_by: Event id of the recorded decision this outcome
            resolves; ``0`` when no decision was registered (direct
            emitter tests, or a resolution for a dispatch made before
            ledger tracking was reset).
        detail: Outcome-specific scalar payload -- exactly the typed
            fields the kind's emit helper attached (target coords for
            targeted kinds, error codes on rejections, landing coords
            on teleports). Strictness lives at the emit-helper
            signatures; this is the flattened query view.
    """

    event_id: int
    attempt_id: int
    action_kind: ActionKind
    outcome: ActionOutcome
    duration_ms: int
    caused_by: int
    detail: dict[str, str | int | float | bool]


_rings: dict[ActionKind, list[ActionOutcomeRecordDict]] = {kind: [] for kind in ACTION_KINDS}


def append_outcome_record(record: ActionOutcomeRecordDict) -> None:
    """Append a record to its kind's ring, evicting the oldest at capacity.

    Args:
        record: Outcome record to append.
    """
    ring = _rings[record["action_kind"]]
    ring.append(record)
    if len(ring) > RING_CAPACITY:
        del ring[0]


def recent_outcomes(kind: ActionKind, count: int) -> list[ActionOutcomeRecordDict]:
    """Return the most recent outcome records for a kind, oldest first.

    Args:
        kind: Action kind to query.
        count: Maximum number of records to return.

    Returns:
        Up to ``count`` most recent records, oldest first.
    """
    ring = _rings[kind]
    if count <= 0:
        return []
    return list(ring[-count:])


def outcome_counts(kind: ActionKind) -> dict[str, int]:
    """Return per-outcome counts over the kind's retained ring.

    Args:
        kind: Action kind to summarize.

    Returns:
        Mapping of outcome label to occurrence count in the ring.
    """
    counts: dict[str, int] = {}
    for record in _rings[kind]:
        counts[record["outcome"]] = counts.get(record["outcome"], 0) + 1
    return counts


def reset_outcome_rings() -> None:
    """Clear every ring. Called from test-isolation fixtures."""
    for kind in ACTION_KINDS:
        _rings[kind].clear()


__all__ = [
    "RING_CAPACITY",
    "ActionOutcomeRecordDict",
    "append_outcome_record",
    "outcome_counts",
    "recent_outcomes",
    "reset_outcome_rings",
]
