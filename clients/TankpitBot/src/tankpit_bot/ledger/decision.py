"""Decision records: what the planner committed to, correlated to outcomes.

Phase 2 correlation layer. The executor records every dispatchable
planner decision here before validation; the outcome fabric
(:mod:`tankpit_bot.ledger.outcome._emit`) automatically pairs each
recorded decision with the next outcome of its action kind, so every
outcome carries ``caused_by`` -- the decision event id it resolves.

The pairing rule mirrors the bot's own invariant: at most one
in-flight action per kind. When a new decision of a kind arrives while
the prior one is still unresolved (a mid-action re-dispatch), the
prior decision is closed with an explicit ``superseded`` outcome
rather than silently dropped -- every recorded decision therefore gets
exactly one outcome, except the still-pending ones at session end
(exposed via :func:`pending_decision_ids` for the shutdown sweep).
"""

from __future__ import annotations

from tankpit_bot.contracts.base import LedgerInvariantError
from tankpit_bot.contracts.enforcement import enforce_contract, require
from tankpit_bot.ledger.events import ActionKind
from tankpit_bot.ledger.outcome._emit import (
    pending_decision_ids,
    register_pending_decision,
    resolved_decision_ids,
)
from tankpit_bot.ledger.records import DecisionRecordDict
from tankpit_bot.ledger.service import LedgerService


class DecisionRecordContract:
    """Structural invariants on a recorded decision."""

    @property
    def name(self) -> str:
        """Name of the contract."""
        return "decision_record"

    def check(
        self,
        ledger: LedgerService,
        *,
        action_kind: ActionKind,
        cmd_type: str,
        mode: str,
        score: int,
        reason_kind: str,
        reason_context: dict[str, str | int],
        target_x: int,
        target_y: int,
        target_id: int,
    ) -> None:
        """Validate a decision before it enters the ledger.

        The contract's signature mirrors the guarded function's, which
        is what makes enforcement type-preserving -- so ``ledger`` is
        declared and unread: the invariants are over the record, not
        over which ledger receives it.

        Args:
            ledger: Session ledger the decision is bound for; unread.
            action_kind: Ledger action kind the command maps to.
            cmd_type: The wire command type dispatched.
            mode: Behavior mode label at decision time.
            score: Behavior priority score.
            reason_kind: Typed decision reason label.
            reason_context: Reason-specific scalar payload.
            target_x: Behavior target X.
            target_y: Behavior target Y.
            target_id: Combat target tank id.

        Raises:
            LedgerInvariantError: If the score is out of its 0-1000
                band or the reason kind is empty.
        """
        _ = ledger
        require(
            0 <= score <= 1000,
            LedgerInvariantError,
            score=repr(score),
            action_kind=action_kind,
        )
        require(
            reason_kind != "",
            LedgerInvariantError,
            cmd_type=cmd_type,
            mode=mode,
        )


@enforce_contract(DecisionRecordContract())
def record_decision(
    ledger: LedgerService,
    *,
    action_kind: ActionKind,
    cmd_type: str,
    mode: str,
    score: int,
    reason_kind: str,
    reason_context: dict[str, str | int],
    target_x: int,
    target_y: int,
    target_id: int,
) -> int:
    """Record a dispatchable planner decision; return its event id.

    Registers the decision as the pending causal parent for its action
    kind -- the next outcome of that kind carries this id in
    ``caused_by``. A prior unresolved decision of the same kind is
    closed with a ``superseded`` outcome first.

    Args:
        ledger: Session ledger receiving the decision.
        action_kind: Ledger action kind the command maps to.
        cmd_type: The wire command type dispatched.
        mode: Behavior mode label at decision time.
        score: Behavior priority score (0-1000).
        reason_kind: Typed decision reason label.
        reason_context: Reason-specific scalar payload.
        target_x: Behavior target X.
        target_y: Behavior target Y.
        target_id: Combat target tank id (0 when untargeted).

    Returns:
        The recorded decision's event id.
    """
    record = DecisionRecordDict(
        event_id=ledger.next_event_id(),
        action_kind=action_kind,
        cmd_type=cmd_type,
        mode=mode,
        score=score,
        reason_kind=reason_kind,
        reason_context=dict(reason_context),
        target_x=target_x,
        target_y=target_y,
        target_id=target_id,
    )
    ledger.decisions[record["event_id"]] = record
    register_pending_decision(ledger, action_kind, record["event_id"])
    return record["event_id"]


def decision_record(ledger: LedgerService, event_id: int) -> DecisionRecordDict | None:
    """Return the recorded decision for an event id, if any.

    Args:
        ledger: Session ledger holding the decision store.
        event_id: Decision event id (e.g. an outcome's ``caused_by``).

    Returns:
        The decision record, or None for 0 / unknown ids.
    """
    return ledger.decisions.get(event_id)


def latest_decision_event_id(ledger: LedgerService) -> int:
    """Return the most recently recorded decision's event id.

    Args:
        ledger: Session ledger holding the decision store.

    Returns:
        The last recorded id, or 0 when no decision has been recorded.
    """
    if not ledger.decisions:
        return 0
    return max(ledger.decisions)


def verify_outcome_invariant(ledger: LedgerService) -> dict[str, int]:
    """Session-end sweep: every recorded decision resolved or pending.

    The pairing machinery makes an orphan structurally impossible --
    an outcome always consumes its kind's pending decision, and a
    superseding decision closes its predecessor -- so a violation here
    means a code path bypassed the fabric. Fail hard per the
    architecture's foundational principle.

    Args:
        ledger: Session ledger to sweep.

    Returns:
        The still-pending decision ids per action kind (the wire never
        answered before shutdown) for the session summary.

    Raises:
        LedgerInvariantError: If any recorded decision is neither
            resolved by an outcome nor pending.
    """
    pending = pending_decision_ids(ledger)
    allowed = set(pending.values()) | resolved_decision_ids(ledger)
    orphans = sorted(event_id for event_id in ledger.decisions if event_id not in allowed)
    require(
        not orphans,
        LedgerInvariantError,
        orphan_decision_ids=", ".join(str(event_id) for event_id in orphans),
    )
    return {str(kind): event_id for kind, event_id in pending.items()}


__all__ = [
    "DecisionRecordContract",
    "decision_record",
    "latest_decision_event_id",
    "record_decision",
    "verify_outcome_invariant",
]
