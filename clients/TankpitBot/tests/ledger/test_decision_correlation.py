"""Tests for the Decision↔Outcome correlation layer."""

from __future__ import annotations

import pytest

from tankpit_bot.contracts.base import LedgerInvariantError
from tankpit_bot.ledger.decision import (
    DecisionRecordContract,
    decision_record,
    latest_decision_event_id,
    record_decision,
    verify_outcome_invariant,
)
from tankpit_bot.ledger.mode_transition import (
    emit_mode_transition,
    mode_transitions,
)
from tankpit_bot.ledger.outcome._emit import pending_decision_ids
from tankpit_bot.ledger.outcome.move import emit_move_position_reached
from tankpit_bot.ledger.outcome.shoot import (
    emit_shoot_miss,
)
from tankpit_bot.ledger.ring import recent_outcomes
from tankpit_bot.ledger.service import LedgerService


@pytest.fixture()
def ledger() -> LedgerService:
    """Return a ledger this test alone owns.

    Returns:
        A fresh, empty ledger.
    """
    return LedgerService()


def _record_move_decision(ledger: LedgerService, score: int = 800) -> int:
    """Record a canonical move decision for pairing tests.

    Args:
        ledger: Session ledger receiving the decision.
        score: Behavior score to stamp.

    Returns:
        The recorded decision's event id.
    """
    return record_decision(
        ledger,
        action_kind="move",
        cmd_type="move",
        mode="COLLECT",
        score=score,
        reason_kind="search_collect_local",
        reason_context={},
        target_x=10,
        target_y=20,
        target_id=0,
    )


def test_outcome_consumes_pending_decision_into_caused_by(ledger: LedgerService) -> None:
    """The next outcome of a kind resolves the recorded decision."""
    decision_id = _record_move_decision(ledger)
    assert pending_decision_ids(ledger) == {"move": decision_id}
    outcome = emit_move_position_reached(
        ledger, duration_ms=500, target_x=10, target_y=20, landed_x=10, landed_y=20
    )
    assert outcome["caused_by"] == decision_id
    assert pending_decision_ids(ledger) == {}
    stored = decision_record(ledger, decision_id)
    assert stored == {
        "event_id": decision_id,
        "action_kind": "move",
        "cmd_type": "move",
        "mode": "COLLECT",
        "score": 800,
        "reason_kind": "search_collect_local",
        "reason_context": {},
        "target_x": 10,
        "target_y": 20,
        "target_id": 0,
    }


def test_superseding_decision_closes_the_prior_one(ledger: LedgerService) -> None:
    """A re-dispatch closes the unresolved prior decision explicitly."""
    first_id = _record_move_decision(ledger)
    second_id = _record_move_decision(ledger, score=900)
    records = recent_outcomes(ledger, "move", 5)
    assert len(records) == 1
    assert records[0]["outcome"] == "superseded"
    assert records[0]["caused_by"] == first_id
    assert records[0]["detail"] == {"superseded_by": second_id}
    assert pending_decision_ids(ledger) == {"move": second_id}


def test_outcome_without_recorded_decision_is_unattributed(ledger: LedgerService) -> None:
    """Emitters fired with no pending decision record caused_by=0."""
    outcome = emit_shoot_miss(ledger, duration_ms=100, target_id=5, target_name="x")
    assert outcome["caused_by"] == 0


def test_verify_outcome_invariant_passes_with_pending_and_resolved(ledger: LedgerService) -> None:
    """The sweep accepts resolved decisions plus the pending tail."""
    first = _record_move_decision(ledger)
    emit_move_position_reached(
        ledger, duration_ms=1, target_x=10, target_y=20, landed_x=10, landed_y=20
    )
    tail = _record_move_decision(ledger)
    unresolved = verify_outcome_invariant(ledger)
    assert unresolved == {"move": tail}
    assert first != tail


def test_verify_outcome_invariant_raises_on_bypassed_fabric(ledger: LedgerService) -> None:
    """A decision resolved outside the fabric is a hard violation."""
    decision_id = _record_move_decision(ledger)
    # Simulate a bypass: something cleared the pending pairing without
    # emitting an outcome for it.
    ledger.pending_decisions.clear()
    with pytest.raises(LedgerInvariantError) as exc:
        verify_outcome_invariant(ledger)
    assert exc.value.details == {"orphan_decision_ids": str(decision_id)}


def test_decision_record_contract_rejects_bad_score_and_empty_reason(ledger: LedgerService) -> None:
    """The record contract enforces score bounds and a present reason."""
    contract = DecisionRecordContract()
    assert contract.name == "decision_record"
    with pytest.raises(LedgerInvariantError):
        record_decision(
            ledger,
            action_kind="move",
            cmd_type="move",
            mode="COLLECT",
            score=1001,
            reason_kind="search_collect_local",
            reason_context={},
            target_x=10,
            target_y=20,
            target_id=0,
        )
    with pytest.raises(LedgerInvariantError):
        record_decision(
            ledger,
            action_kind="move",
            cmd_type="move",
            mode="COLLECT",
            score=500,
            reason_kind="",
            reason_context={},
            target_x=10,
            target_y=20,
            target_id=0,
        )


def test_latest_decision_event_id_tracks_recording(ledger: LedgerService) -> None:
    """The latest id is 0 before any record and advances after."""
    assert latest_decision_event_id(ledger) == 0
    first = _record_move_decision(ledger)
    assert latest_decision_event_id(ledger) == first


def test_mode_transitions_are_first_class_events(ledger: LedgerService) -> None:
    """Mode flips record event id, reason, and causal decision."""
    decision_id = _record_move_decision(ledger)
    record = emit_mode_transition(
        ledger,
        from_mode="HUNT",
        to_mode="COLLECT",
        reason_kind="fuel_collect",
        caused_by=decision_id,
    )
    assert record["from_mode"] == "HUNT"
    assert record["to_mode"] == "COLLECT"
    assert record["caused_by"] == decision_id
    assert record["event_id"] > decision_id
    assert mode_transitions(ledger) == [record]
