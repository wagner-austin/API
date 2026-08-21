"""Ledger service — owns the session's bookkeeping as instance attributes.

Replaces the six module-level globals that made the ledger a
process-wide singleton: the event counter, the per-kind outcome rings,
the decision store, the mode-transition log, the outcome-pairing
trackers, and the pending teleport dispatch.

One ``LedgerService`` per session. Two sessions in one process now keep
two independent ledgers, which the module globals made impossible
([[session-state-deglobalisation]] step 6). The session reaches it
through :attr:`WorldService.ledger`, the same way it already reaches the
fuel, damage, and ammo books.
"""

from __future__ import annotations

from tankpit_bot.ledger.events import ACTION_KINDS, ActionKind
from tankpit_bot.ledger.records import (
    ActionOutcomeRecordDict,
    DecisionRecordDict,
    ModeTransitionRecordDict,
    PendingTeleportDispatchDict,
)


class LedgerService:
    """Per-session ledger state.

    Attributes:
        event_counter: Backing counter for :meth:`next_event_id`.
        rings: Bounded per-kind outcome history.
        decisions: Recorded planner decisions by event id.
        transitions: Mode flips in occurrence order.
        attempt_counters: Per-kind monotonic attempt ids.
        pending_decisions: Unresolved decision event id per action kind.
        resolved_decision_ids: Decision ids an outcome has consumed.
        pending_teleport: Dispatch context of the in-flight teleport.
        zero_dispatch_streaks: Consecutive zero-duration ``superseded``
            closes per kind — the live livelock detector's counter
            (see ``LIVENESS_STALL_STREAK``). Reset by any non-superseded
            outcome of the kind.
    """

    def __init__(self) -> None:
        """Start an empty ledger for one session."""
        self.event_counter: int = 0
        self.rings: dict[ActionKind, list[ActionOutcomeRecordDict]] = {
            kind: [] for kind in ACTION_KINDS
        }
        self.decisions: dict[int, DecisionRecordDict] = {}
        self.transitions: list[ModeTransitionRecordDict] = []
        self.attempt_counters: dict[ActionKind, int] = dict.fromkeys(ACTION_KINDS, 0)
        self.pending_decisions: dict[ActionKind, int] = {}
        self.resolved_decision_ids: set[int] = set()
        self.pending_teleport: PendingTeleportDispatchDict | None = None
        self.zero_dispatch_streaks: dict[ActionKind, int] = dict.fromkeys(ACTION_KINDS, 0)

    def next_event_id(self) -> int:
        """Return the next session-wide monotonic event id.

        Returns:
            Strictly increasing integer, starting at 1.
        """
        self.event_counter += 1
        return self.event_counter

    def next_attempt_id(self, action_kind: ActionKind) -> int:
        """Return the next attempt id for a kind (strictly monotonic).

        Args:
            action_kind: Kind whose counter advances.

        Returns:
            Strictly increasing integer per kind, starting at 1.
        """
        self.attempt_counters[action_kind] += 1
        return self.attempt_counters[action_kind]


__all__ = [
    "LedgerService",
]
