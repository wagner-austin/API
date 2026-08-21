"""Tests for the unified action-outcome fabric: ids, ring, emitters."""

from __future__ import annotations

import pytest

from tankpit_bot.ledger.events import ACTION_KINDS
from tankpit_bot.ledger.outcome.map_open import (
    emit_map_open_command_rejected,
    emit_map_open_data_processed,
    emit_map_open_stall_timeout,
)
from tankpit_bot.ledger.outcome.move import (
    emit_move_command_rejected,
    emit_move_movement_rejected,
    emit_move_position_reached,
    emit_move_stall_timeout,
)
from tankpit_bot.ledger.outcome.scan import (
    emit_scan_command_rejected,
    emit_scan_radar_complete,
    emit_scan_stall_timeout,
)
from tankpit_bot.ledger.outcome.teleport import (
    emit_teleport_landed,
    emit_teleport_stall_timeout,
    record_teleport_dispatch,
)
from tankpit_bot.ledger.records import RING_CAPACITY
from tankpit_bot.ledger.ring import outcome_counts, recent_outcomes
from tankpit_bot.ledger.service import LedgerService
from tests.conftest import FakeFileSystem


@pytest.fixture()
def ledger() -> LedgerService:
    """Return a ledger this test alone owns.

    Replaces the four reset calls this file used to make around every
    test: a service nobody else holds cannot leak
    ([[session-state-deglobalisation]] step 6).

    Returns:
        A fresh, empty ledger.
    """
    return LedgerService()


def test_event_ids_are_strictly_monotonic(ledger: LedgerService) -> None:
    """next_event_id increments by exactly one per call."""
    assert ledger.next_event_id() == 1
    assert ledger.next_event_id() == 2
    assert ledger.next_event_id() == 3


def test_attempt_ids_are_per_kind_monotonic(ledger: LedgerService) -> None:
    """Each kind's attempt counter advances independently."""
    first_scan = emit_scan_radar_complete(ledger, duration_ms=10, target_x=1, target_y=2)
    first_move = emit_move_movement_rejected(ledger, duration_ms=20, target_x=3, target_y=4)
    second_scan = emit_scan_stall_timeout(
        ledger, duration_ms=30, target_x=1, target_y=2, timeout_ms=100
    )
    assert first_scan["attempt_id"] == 1
    assert first_move["attempt_id"] == 1
    assert second_scan["attempt_id"] == 2
    assert first_scan["event_id"] == 1
    assert first_move["event_id"] == 2
    assert second_scan["event_id"] == 3


def test_transfer_from_a_kind_with_nothing_pending_is_a_no_op(
    ledger: LedgerService,
) -> None:
    """Transferring from an empty kind leaves the destination untouched.

    The docstring's contract is "no-op when ``from_kind`` has no pending
    decision", and the destination is the part worth pinning: reaching
    ``register_pending_decision`` with ``None`` would close the
    destination's real pending decision as ``superseded`` -- a
    fabricated outcome for a decision nothing re-planned -- and then
    store ``None`` in its place, so the outcome that eventually arrives
    would pair against nothing.
    """
    from tankpit_bot.ledger.outcome._emit import (
        pending_decision_ids,
        register_pending_decision,
        transfer_pending_decision,
    )

    register_pending_decision(ledger, "map_open", 42)

    transfer_pending_decision(ledger, "teleport", "map_open")

    assert pending_decision_ids(ledger) == {"map_open": 42}
    assert outcome_counts(ledger, "map_open").get("superseded", 0) == 0


def test_transfer_moves_a_real_pending_decision(ledger: LedgerService) -> None:
    """Control: a genuine transfer does move the decision to the new kind."""
    from tankpit_bot.ledger.outcome._emit import (
        pending_decision_ids,
        register_pending_decision,
        transfer_pending_decision,
    )

    register_pending_decision(ledger, "teleport", 7)

    transfer_pending_decision(ledger, "teleport", "map_open")

    assert pending_decision_ids(ledger) == {"map_open": 7}


def test_ring_records_and_counts_outcomes(ledger: LedgerService) -> None:
    """Emitted outcomes land in their kind's ring and are countable."""
    emit_scan_radar_complete(ledger, duration_ms=10, target_x=1, target_y=2)
    emit_scan_radar_complete(ledger, duration_ms=12, target_x=1, target_y=2)
    emit_scan_command_rejected(ledger, duration_ms=5, target_x=1, target_y=2, error_code=0)
    records = recent_outcomes(ledger, "scan", 10)
    assert [r["outcome"] for r in records] == [
        "radar_complete",
        "radar_complete",
        "command_rejected",
    ]
    assert outcome_counts(ledger, "scan") == {"radar_complete": 2, "command_rejected": 1}
    assert recent_outcomes(ledger, "scan", 0) == []
    assert recent_outcomes(ledger, "move", 5) == []


def test_ring_evicts_oldest_at_capacity(ledger: LedgerService) -> None:
    """The ring stays bounded and keeps the newest records."""
    for index in range(RING_CAPACITY + 10):
        emit_map_open_data_processed(ledger, duration_ms=index)
    records = recent_outcomes(ledger, "map_open", RING_CAPACITY + 10)
    assert len(records) == RING_CAPACITY
    assert records[0]["duration_ms"] == 10
    assert records[-1]["duration_ms"] == RING_CAPACITY + 9


def test_move_outcomes_carry_typed_payloads(ledger: LedgerService) -> None:
    """Move emitters attach exactly their declared detail fields."""
    reached = emit_move_position_reached(
        ledger, duration_ms=100, target_x=5, target_y=6, landed_x=5, landed_y=6
    )
    assert reached["detail"] == {"target_x": 5, "target_y": 6, "landed_x": 5, "landed_y": 6}
    rejected = emit_move_command_rejected(
        ledger, duration_ms=50, target_x=5, target_y=6, error_code=3
    )
    assert rejected["detail"]["error_code"] == 3
    stalled = emit_move_stall_timeout(
        ledger, duration_ms=10000, target_x=5, target_y=6, timeout_ms=10000
    )
    assert stalled["detail"]["timeout_ms"] == 10000


def test_map_open_outcomes_have_no_target_fields(ledger: LedgerService) -> None:
    """Map-open payloads carry no target coordinates (no sentinels)."""
    processed = emit_map_open_data_processed(ledger, duration_ms=900)
    assert processed["detail"] == {}
    stalled = emit_map_open_stall_timeout(ledger, duration_ms=8000, timeout_ms=8000)
    assert stalled["detail"] == {"timeout_ms": 8000}
    rejected = emit_map_open_command_rejected(ledger, duration_ms=100, error_code=0)
    assert rejected["detail"] == {"error_code": 0}


def test_teleport_landed_classifies_exact_vs_inexact(ledger: LedgerService) -> None:
    """Landing on the requested tile is exact; displacement is inexact."""
    exact = emit_teleport_landed(
        ledger, duration_ms=300, target_x=10, target_y=20, landed_x=10, landed_y=20, messages=[]
    )
    assert exact["outcome"] == "landed_exact"
    inexact = emit_teleport_landed(
        ledger, duration_ms=300, target_x=10, target_y=20, landed_x=11, landed_y=20, messages=[]
    )
    assert inexact["outcome"] == "landed_inexact"
    assert exact["detail"]["sent_window"] == "(none)"


def test_teleport_dispatch_context_flows_into_the_outcome(ledger: LedgerService) -> None:
    """A recorded dispatch enriches the landing with wire windows."""
    record_teleport_dispatch(
        ledger, target_x=10, target_y=20, message_index=1, sent_window="fuel=800 pos=(9,20)"
    )
    landed = emit_teleport_landed(
        ledger,
        duration_ms=300,
        target_x=10,
        target_y=20,
        landed_x=10,
        landed_y=20,
        messages=[
            {"direction": "sent", "payload": "74AABB", "timestamp_ms": 1, "ws_url": "ws://game"},
            {
                "direction": "received",
                "payload": "3D0102030405",
                "timestamp_ms": 2,
                "ws_url": "ws://game",
            },
        ],
    )
    assert landed["detail"]["sent_window"] == "fuel=800 pos=(9,20)"
    assert landed["detail"]["received_window"] == "received:12:3D0102030405"
    follow_up = emit_teleport_stall_timeout(
        ledger, duration_ms=9000, target_x=1, target_y=2, timeout_ms=9000, messages=[]
    )
    assert follow_up["detail"]["sent_window"] == "(none)"


def test_action_kinds_cover_the_seven_ledgered_actions(ledger: LedgerService) -> None:
    """The ledger records exactly the seven real action kinds.

    ``scope`` joined 2026-08-20 when the viewport pan was promoted
    from fire-and-forget ([[viewport-shift-protocol]] scope-pending
    radar drop) — a new kind here is a deliberate act, never drift.
    """
    assert ACTION_KINDS == ("scan", "move", "teleport", "collect", "map_open", "shoot", "scope")


def test_zero_dispatch_streak_counts_supersedes_and_rearms(ledger: LedgerService) -> None:
    """The livelock counter tracks zero-dispatch replans per kind.

    Each ``register_pending_decision`` that closes a prior decision as
    zero-duration ``superseded`` advances the kind's streak; any
    genuine (non-superseded) resolution of the kind resets it. The
    counter is the live half of the liveness instrument
    ([[fleet-coordination]] gatherer livelock, 2026-08-20).
    """
    from tankpit_bot.ledger.outcome._emit import register_pending_decision

    register_pending_decision(ledger, "collect", ledger.next_event_id())
    assert ledger.zero_dispatch_streaks["collect"] == 0

    register_pending_decision(ledger, "collect", ledger.next_event_id())
    register_pending_decision(ledger, "collect", ledger.next_event_id())
    assert ledger.zero_dispatch_streaks["collect"] == 2
    # Another kind's streak is independent.
    assert ledger.zero_dispatch_streaks["scan"] == 0

    # A genuine resolution of the kind re-arms the counter.
    emit_move_position_reached(
        ledger, duration_ms=10, target_x=1, target_y=2, landed_x=1, landed_y=2
    )
    assert ledger.zero_dispatch_streaks["collect"] == 2
    register_pending_decision(ledger, "move", ledger.next_event_id())
    emit_move_position_reached(
        ledger, duration_ms=10, target_x=1, target_y=2, landed_x=1, landed_y=2
    )
    assert ledger.zero_dispatch_streaks["move"] == 0


def test_dispatched_supersede_resets_the_streak_and_records_the_mark(
    ledger: LedgerService,
) -> None:
    """A superseded close of a DISPATCHED decision is a re-aim, not a stall.

    The 2026-08-21 false positive: 12 clearance shots each reached the
    wire and echoed, yet every decision closed ``superseded`` and the
    counter read them as zero dispatches. With the executor's dispatch
    mark, the supersede resets the streak, carries ``dispatched=True``
    in its record, and consumes the mark.
    """
    from tankpit_bot.ledger.outcome._emit import (
        mark_decision_dispatched,
        register_pending_decision,
    )

    first = ledger.next_event_id()
    register_pending_decision(ledger, "shoot", first)
    mark_decision_dispatched(ledger, first)
    # Build a streak first so the reset is observable.
    ledger.zero_dispatch_streaks["shoot"] = 5

    register_pending_decision(ledger, "shoot", ledger.next_event_id())

    assert ledger.zero_dispatch_streaks["shoot"] == 0
    superseded = recent_outcomes(ledger, "shoot", 1)[0]
    assert superseded["outcome"] == "superseded"
    assert superseded["detail"]["dispatched"] is True
    assert first not in ledger.dispatched_decision_ids

    # An UNDISPATCHED supersede still counts, from the reset baseline.
    register_pending_decision(ledger, "shoot", ledger.next_event_id())
    assert ledger.zero_dispatch_streaks["shoot"] == 1
    undispatched = recent_outcomes(ledger, "shoot", 1)[0]
    assert undispatched["detail"]["dispatched"] is False


def test_dispatch_mark_survives_the_pending_transfer(ledger: LedgerService) -> None:
    """A deferred teleport's map_open dispatch credits the ORIGINAL decision.

    The mark is keyed by event id, so moving the pending decision from
    ``teleport`` to ``map_open`` (the executor's map-open deferral)
    keeps the dispatched fact attached to it.
    """
    from tankpit_bot.ledger.outcome._emit import (
        mark_decision_dispatched,
        register_pending_decision,
        transfer_pending_decision,
    )

    decision_id = ledger.next_event_id()
    register_pending_decision(ledger, "teleport", decision_id)
    transfer_pending_decision(ledger, "teleport", "map_open")
    mark_decision_dispatched(ledger, decision_id)

    register_pending_decision(ledger, "map_open", ledger.next_event_id())

    superseded = recent_outcomes(ledger, "map_open", 1)[0]
    assert superseded["detail"]["dispatched"] is True
    assert ledger.zero_dispatch_streaks["map_open"] == 0


def test_genuine_resolution_discards_the_dispatch_mark(ledger: LedgerService) -> None:
    """Resolving a marked decision consumes its mark — the set stays bounded."""
    from tankpit_bot.ledger.outcome._emit import (
        mark_decision_dispatched,
        register_pending_decision,
    )

    decision_id = ledger.next_event_id()
    register_pending_decision(ledger, "scan", decision_id)
    mark_decision_dispatched(ledger, decision_id)

    emit_scan_radar_complete(ledger, duration_ms=10, target_x=1, target_y=2)

    assert ledger.dispatched_decision_ids == set()
    assert ledger.zero_dispatch_streaks["scan"] == 0


def test_liveness_stall_diagnostic_fires_once_at_the_crossing(
    fake_fs: FakeFileSystem,
) -> None:
    """Crossing ``LIVENESS_STALL_STREAK`` emits exactly one diagnostic.

    Once at the crossing, silent while the streak continues, re-armed
    by a genuine resolution — so a wedged session announces itself in
    the live log without spamming every subsequent tick.
    """
    from pathlib import Path

    from platform_core.json_utils import load_json_str, narrow_json_to_dict

    from tankpit_bot.ledger.outcome._emit import register_pending_decision
    from tankpit_bot.ledger.outcomes import LIVENESS_STALL_STREAK
    from tankpit_bot.runtime_logging import configure_probe_runtime_logging

    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    ledger = LedgerService()

    def stall_events() -> int:
        raw = fake_fs.read_text(Path(artifacts["latest_events_path"]))
        return sum(
            1
            for line in raw.splitlines()
            if line
            and narrow_json_to_dict(load_json_str(line)).get("diagnostic_kind") == "liveness_stall"
        )

    for _ in range(LIVENESS_STALL_STREAK + 3):
        register_pending_decision(ledger, "collect", ledger.next_event_id())
    assert ledger.zero_dispatch_streaks["collect"] == LIVENESS_STALL_STREAK + 2
    assert stall_events() == 1

    # Genuine resolution re-arms; a second wedge announces again.
    emit_scan_radar_complete(ledger, duration_ms=10, target_x=1, target_y=2)
    register_pending_decision(ledger, "scan", ledger.next_event_id())
    for _ in range(LIVENESS_STALL_STREAK):
        register_pending_decision(ledger, "scan", ledger.next_event_id())
    assert stall_events() == 2
