"""Bounded per-kind ring of action-outcome records.

Every emitted outcome is appended to its kind's ring (bounded at
:data:`tankpit_bot.ledger.records.RING_CAPACITY`), giving planners an
in-session queryable view of recent attempts --
``recent_outcomes(ledger, "shoot", 5)`` replaces implicit state
inference from scattered counters.
"""

from __future__ import annotations

from tankpit_bot.ledger.events import ActionKind
from tankpit_bot.ledger.records import RING_CAPACITY, ActionOutcomeRecordDict
from tankpit_bot.ledger.service import LedgerService


def append_outcome_record(ledger: LedgerService, record: ActionOutcomeRecordDict) -> None:
    """Append a record to its kind's ring, evicting the oldest at capacity.

    Args:
        ledger: Session ledger owning the rings.
        record: Outcome record to append.
    """
    ring = ledger.rings[record["action_kind"]]
    ring.append(record)
    if len(ring) > RING_CAPACITY:
        del ring[0]


def recent_outcomes(
    ledger: LedgerService,
    kind: ActionKind,
    count: int,
) -> list[ActionOutcomeRecordDict]:
    """Return the most recent outcome records for a kind, oldest first.

    Args:
        ledger: Session ledger owning the rings.
        kind: Action kind to query.
        count: Maximum number of records to return.

    Returns:
        Up to ``count`` most recent records, oldest first.
    """
    ring = ledger.rings[kind]
    if count <= 0:
        return []
    return list(ring[-count:])


def outcome_counts(ledger: LedgerService, kind: ActionKind) -> dict[str, int]:
    """Return per-outcome counts over the kind's retained ring.

    Args:
        ledger: Session ledger owning the rings.
        kind: Action kind to summarize.

    Returns:
        Mapping of outcome label to occurrence count in the ring.
    """
    counts: dict[str, int] = {}
    for record in ledger.rings[kind]:
        counts[record["outcome"]] = counts.get(record["outcome"], 0) + 1
    return counts


__all__ = [
    "append_outcome_record",
    "outcome_counts",
    "recent_outcomes",
]
