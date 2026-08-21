"""Shared action-outcome emission: attempt ids + pairing + diagnostic + ring.

Single low-level path every per-kind emit helper routes through: one
``action_outcome`` diagnostic event per attempt resolution, one ring
append, one per-kind monotonic attempt id. The per-kind modules
(:mod:`scan`, :mod:`move`, ...) own the typed outcome vocabulary and
the strict per-outcome argument signatures; this module owns the
plumbing.

Decision correlation (Phase 2): the executor registers each recorded
decision as the pending causal parent for its action kind (the bot has
at most one in-flight action per kind); the next outcome of that kind
consumes it into ``caused_by``. Registering a new decision while the
prior one is unresolved closes the prior with an explicit
``superseded`` outcome, so every recorded decision resolves to exactly
one outcome -- the still-pending set at session end is exposed via
:func:`pending_decision_ids` for the shutdown sweep.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.ledger.events import ActionKind
from tankpit_bot.ledger.outcomes import LIVENESS_STALL_STREAK, ActionOutcome
from tankpit_bot.ledger.records import ActionOutcomeRecordDict
from tankpit_bot.ledger.ring import append_outcome_record
from tankpit_bot.ledger.service import LedgerService
from tankpit_bot.runtime_logging import emit_diagnostic

log = get_logger(__name__)


def register_pending_decision(
    ledger: LedgerService,
    action_kind: ActionKind,
    event_id: int,
) -> None:
    """Register a decision as the causal parent of the kind's next outcome.

    A prior unresolved decision of the same kind is closed with a
    ``superseded`` outcome first (mid-action re-dispatch: the bot
    replaced its own plan before the wire resolved it).

    Args:
        ledger: Session ledger holding the pairing state.
        action_kind: Kind the decision's command maps to.
        event_id: The recorded decision's event id.
    """
    prior = ledger.pending_decisions.get(action_kind)
    if prior is not None:
        dispatched = prior in ledger.dispatched_decision_ids
        emit_action_outcome(
            ledger,
            action_kind=action_kind,
            outcome="superseded",
            duration_ms=0,
            superseded_by=event_id,
            dispatched=dispatched,
        )
        # The live livelock detector counts only UNDISPATCHED
        # supersedes: a decision replaced before anything reached the
        # wire, whose streak is the planner/veto feedback gap
        # ([[fleet-coordination]] gatherer livelock — 93 in a row
        # while the session looked busy). A superseded close of a
        # DISPATCHED decision instead resets the streak — the planner
        # is reaching the wire, its outcome just went unclassified
        # (the 2026-08-21 false positive: 12 dispatched-and-echoed
        # clearance shots read as a livelock). Fires once at the
        # crossing; any genuine resolution also re-arms it below.
        if dispatched:
            ledger.zero_dispatch_streaks[action_kind] = 0
        else:
            ledger.zero_dispatch_streaks[action_kind] += 1
            streak = ledger.zero_dispatch_streaks[action_kind]
            if streak == LIVENESS_STALL_STREAK:
                log.warning(
                    "LIVENESS STALL: %s replanned %d consecutive times with zero dispatches",
                    action_kind,
                    streak,
                )
                emit_diagnostic(
                    diagnostic_kind="liveness_stall",
                    action_kind=action_kind,
                    streak=streak,
                )
    ledger.pending_decisions[action_kind] = event_id


def mark_decision_dispatched(ledger: LedgerService, event_id: int) -> None:
    """Record that a decision's command actually reached the wire.

    Called by the executor after a successful dispatch. A decision so
    marked can close ``superseded`` without feeding the zero-dispatch
    streak — the planner's output demonstrably left the process, so a
    replan on top of it is re-aiming, not a livelock. Marking by event
    id (not kind) keeps the mark valid across
    :func:`transfer_pending_decision` (a deferred teleport's map_open
    dispatch marks the ORIGINAL teleport decision).

    Args:
        ledger: Session ledger holding the pairing state.
        event_id: The dispatched decision's event id.
    """
    ledger.dispatched_decision_ids.add(event_id)


def transfer_pending_decision(
    ledger: LedgerService,
    from_kind: ActionKind,
    to_kind: ActionKind,
) -> None:
    """Move a pending decision to the kind its tick actually produced.

    Used when a dispatch path substitutes a different wire action for
    the decided one -- e.g. a teleport deferring to open the map first:
    the decision's real product this tick is the map open, so its
    outcome arrives on the ``map_open`` kind. Transferring keeps the
    exactly-one-outcome invariant without a spurious ``superseded``
    (nothing was re-planned; the same decision is still executing).

    No-op when ``from_kind`` has no pending decision. An existing
    pending decision on ``to_kind`` is closed as superseded through the
    normal registration path.

    Args:
        ledger: Session ledger holding the pairing state.
        from_kind: Kind the decision was recorded under.
        to_kind: Kind whose next outcome will resolve the decision.
    """
    moved = ledger.pending_decisions.pop(from_kind, None)
    if moved is None:
        return
    register_pending_decision(ledger, to_kind, moved)


def pending_decision_ids(ledger: LedgerService) -> dict[ActionKind, int]:
    """Return the still-unresolved decision id per action kind.

    Args:
        ledger: Session ledger holding the pairing state.

    Returns:
        Mapping of action kind to its pending decision event id. Empty
        when every recorded decision has resolved to an outcome. Read
        by the session-end sweep -- these are the only decisions
        legitimately allowed to lack an outcome (the wire never
        answered before shutdown).
    """
    return dict(ledger.pending_decisions)


def resolved_decision_ids(ledger: LedgerService) -> set[int]:
    """Return every decision id an outcome has resolved this session.

    Args:
        ledger: Session ledger holding the pairing state.

    Returns:
        Set of decision event ids consumed into ``caused_by``.
    """
    return set(ledger.resolved_decision_ids)


def emit_action_outcome(
    ledger: LedgerService,
    *,
    action_kind: ActionKind,
    outcome: ActionOutcome,
    duration_ms: int,
    **detail: str | int | float | bool,
) -> ActionOutcomeRecordDict:
    """Record one resolved action attempt: diagnostic event + ring entry.

    Consumes the kind's pending decision (if any) into ``caused_by``
    -- the outcome resolves that decision.

    Args:
        ledger: Session ledger receiving the outcome.
        action_kind: Kind of action that resolved.
        outcome: Outcome label from the kind's outcome union.
        duration_ms: Wall-clock ms from dispatch to resolution; ``-1``
            when no dispatch time was recorded.
        **detail: Outcome-specific scalar payload, exactly as the
            kind's emit helper declared it.

    Returns:
        The recorded outcome, as appended to the kind's ring.
    """
    caused_by = ledger.pending_decisions.pop(action_kind, 0)
    if caused_by != 0:
        ledger.resolved_decision_ids.add(caused_by)
        ledger.dispatched_decision_ids.discard(caused_by)
    if outcome != "superseded":
        # Any genuine resolution of the kind proves the planner's
        # output is reaching the wire again — the stall counter and
        # its one-shot diagnostic both re-arm.
        ledger.zero_dispatch_streaks[action_kind] = 0
    record = ActionOutcomeRecordDict(
        event_id=ledger.next_event_id(),
        attempt_id=ledger.next_attempt_id(action_kind),
        action_kind=action_kind,
        outcome=outcome,
        duration_ms=duration_ms,
        caused_by=caused_by,
        detail=dict(detail),
    )
    append_outcome_record(ledger, record)
    emit_diagnostic(
        diagnostic_kind="action_outcome",
        action_kind=action_kind,
        outcome=outcome,
        event_id=record["event_id"],
        attempt_id=record["attempt_id"],
        duration_ms=duration_ms,
        caused_by=caused_by,
        **detail,
    )
    return record


__all__ = [
    "emit_action_outcome",
    "log",
    "mark_decision_dispatched",
    "pending_decision_ids",
    "register_pending_decision",
    "resolved_decision_ids",
    "transfer_pending_decision",
]
