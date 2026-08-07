"""Record shapes the ledger stores — no state, no behaviour.

These live apart from the modules that build them so
:mod:`tankpit_bot.ledger.service` can type its attributes without
importing those modules, which import ``LedgerService`` in turn. Keeping
the shapes here is what lets the whole cluster take the service as a
parameter without closing an import cycle
([[session-state-deglobalisation]] step 6).
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.ledger.events import ActionKind
from tankpit_bot.ledger.outcomes import ActionOutcome

RING_CAPACITY = 128
"""Per-kind retention bound on the outcome rings."""


class ActionOutcomeRecordDict(TypedDict):
    """One recorded action outcome.

    Attributes:
        event_id: Session-wide monotonic event id.
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
            ledger tracking began).
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


class DecisionRecordDict(TypedDict):
    """One recorded planner decision.

    Attributes:
        event_id: Session-wide monotonic event id.
        action_kind: Ledger action kind the command maps to.
        cmd_type: The wire command type dispatched.
        mode: Behavior mode label at decision time.
        score: Behavior priority score (0-1000).
        reason_kind: Typed decision reason label.
        reason_context: Reason-specific scalar payload.
        target_x: Behavior target X.
        target_y: Behavior target Y.
        target_id: Combat target tank id (0 when untargeted).
    """

    event_id: int
    action_kind: ActionKind
    cmd_type: str
    mode: str
    score: int
    reason_kind: str
    reason_context: dict[str, str | int]
    target_x: int
    target_y: int
    target_id: int


class ModeTransitionRecordDict(TypedDict):
    """One recorded mode transition.

    Attributes:
        event_id: Session-wide monotonic event id.
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


class PendingTeleportDispatchDict(TypedDict):
    """Dispatch context held until a completion gate resolves the attempt.

    Attributes:
        target_x: Requested landing X coordinate.
        target_y: Requested landing Y coordinate.
        started_ms: Wall-clock dispatch time.
        message_index: Length of the captured-message list at dispatch;
            everything after this index is the attempt's wire window.
        sent_window: Compact live-client context at dispatch time.
    """

    target_x: int
    target_y: int
    started_ms: int
    message_index: int
    sent_window: str


__all__ = [
    "RING_CAPACITY",
    "ActionOutcomeRecordDict",
    "DecisionRecordDict",
    "ModeTransitionRecordDict",
    "PendingTeleportDispatchDict",
]
