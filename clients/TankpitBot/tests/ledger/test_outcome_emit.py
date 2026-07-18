"""Tests for the unified action-outcome fabric: ids, ring, emitters."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from tankpit_bot.ledger.events import ACTION_KINDS, next_event_id, reset_event_ids
from tankpit_bot.ledger.outcome._emit import reset_action_outcome_tracking
from tankpit_bot.ledger.outcome.collect import (
    emit_collect_command_rejected,
    emit_collect_container_consumed,
    emit_collect_discarded_kind_mismatch,
    emit_collect_discarded_no_container,
    emit_collect_movement_rejected,
    emit_collect_position_reached,
    emit_collect_stall_timeout,
)
from tankpit_bot.ledger.outcome.map_open import (
    emit_map_open_command_rejected,
    emit_map_open_data_processed,
    emit_map_open_stall_timeout,
)
from tankpit_bot.ledger.outcome.move import (
    emit_move_command_rejected,
    emit_move_discarded_hostile_mine,
    emit_move_movement_rejected,
    emit_move_position_reached,
    emit_move_stall_timeout,
)
from tankpit_bot.ledger.outcome.scan import (
    emit_scan_command_rejected,
    emit_scan_radar_complete,
    emit_scan_stall_timeout,
)
from tankpit_bot.ledger.outcome.shoot import (
    emit_shoot_command_rejected,
    emit_shoot_discarded_target_not_tracked,
    emit_shoot_hit,
    emit_shoot_miss,
)
from tankpit_bot.ledger.outcome.teleport import (
    emit_teleport_command_rejected,
    emit_teleport_discarded_combat_target_stale,
    emit_teleport_discarded_hostile_mine,
    emit_teleport_discarded_resource_target_invalid,
    emit_teleport_discarded_resource_target_stale,
    emit_teleport_landed,
    emit_teleport_stall_timeout,
    record_teleport_dispatch,
    reset_teleport_dispatch_tracking,
)
from tankpit_bot.ledger.ring import (
    RING_CAPACITY,
    outcome_counts,
    recent_outcomes,
    reset_outcome_rings,
)


@pytest.fixture(autouse=True)
def _reset_ledger() -> Generator[None, None, None]:
    """Reset ledger counters and rings around every test."""
    reset_event_ids()
    reset_action_outcome_tracking()
    reset_outcome_rings()
    reset_teleport_dispatch_tracking()
    yield
    reset_event_ids()
    reset_action_outcome_tracking()
    reset_outcome_rings()
    reset_teleport_dispatch_tracking()


def test_event_ids_are_strictly_monotonic() -> None:
    """next_event_id increments by exactly one per call."""
    assert next_event_id() == 1
    assert next_event_id() == 2
    assert next_event_id() == 3


def test_attempt_ids_are_per_kind_monotonic() -> None:
    """Each kind's attempt counter advances independently."""
    first_scan = emit_scan_radar_complete(duration_ms=10, target_x=1, target_y=2)
    first_move = emit_move_movement_rejected(duration_ms=20, target_x=3, target_y=4)
    second_scan = emit_scan_stall_timeout(duration_ms=30, target_x=1, target_y=2, timeout_ms=100)
    assert first_scan["attempt_id"] == 1
    assert first_move["attempt_id"] == 1
    assert second_scan["attempt_id"] == 2
    assert first_scan["event_id"] == 1
    assert first_move["event_id"] == 2
    assert second_scan["event_id"] == 3


def test_ring_records_and_counts_outcomes() -> None:
    """Emitted outcomes land in their kind's ring and are countable."""
    emit_scan_radar_complete(duration_ms=10, target_x=1, target_y=2)
    emit_scan_radar_complete(duration_ms=12, target_x=1, target_y=2)
    emit_scan_command_rejected(duration_ms=5, target_x=1, target_y=2, error_code=0)
    records = recent_outcomes("scan", 10)
    assert [r["outcome"] for r in records] == [
        "radar_complete",
        "radar_complete",
        "command_rejected",
    ]
    assert outcome_counts("scan") == {"radar_complete": 2, "command_rejected": 1}
    assert recent_outcomes("scan", 0) == []
    assert recent_outcomes("move", 5) == []


def test_ring_evicts_oldest_at_capacity() -> None:
    """The ring stays bounded and keeps the newest records."""
    for index in range(RING_CAPACITY + 10):
        emit_map_open_data_processed(duration_ms=index)
    records = recent_outcomes("map_open", RING_CAPACITY + 10)
    assert len(records) == RING_CAPACITY
    assert records[0]["duration_ms"] == 10
    assert records[-1]["duration_ms"] == RING_CAPACITY + 9


def test_move_outcomes_carry_typed_payloads() -> None:
    """Move emitters attach exactly their declared detail fields."""
    reached = emit_move_position_reached(
        duration_ms=100, target_x=5, target_y=6, landed_x=5, landed_y=6
    )
    assert reached["detail"] == {"target_x": 5, "target_y": 6, "landed_x": 5, "landed_y": 6}
    rejected = emit_move_command_rejected(duration_ms=50, target_x=5, target_y=6, error_code=3)
    assert rejected["detail"]["error_code"] == 3
    stalled = emit_move_stall_timeout(duration_ms=10000, target_x=5, target_y=6, timeout_ms=10000)
    assert stalled["detail"]["timeout_ms"] == 10000
    discarded = emit_move_discarded_hostile_mine(target_x=5, target_y=6)
    assert discarded["outcome"] == "discarded_hostile_mine"
    assert discarded["duration_ms"] == 0


def test_collect_outcomes_cover_all_seven_labels() -> None:
    """Every collect resolution is recordable."""
    emit_collect_position_reached(duration_ms=1, target_x=1, target_y=1, landed_x=1, landed_y=1)
    emit_collect_container_consumed(duration_ms=2, target_x=1, target_y=1, landed_x=0, landed_y=1)
    emit_collect_movement_rejected(duration_ms=3, target_x=1, target_y=1)
    emit_collect_command_rejected(duration_ms=4, target_x=1, target_y=1, error_code=5)
    emit_collect_stall_timeout(duration_ms=5, target_x=1, target_y=1, timeout_ms=10000)
    emit_collect_discarded_no_container(target_x=1, target_y=1, pickup_kind="fuel")
    emit_collect_discarded_kind_mismatch(target_x=1, target_y=1, pickup_kind="equipment")
    assert sorted(outcome_counts("collect")) == [
        "command_rejected",
        "container_consumed",
        "discarded_kind_mismatch",
        "discarded_no_container",
        "movement_rejected",
        "position_reached",
        "stall_timeout",
    ]


def test_map_open_outcomes_have_no_target_fields() -> None:
    """Map-open payloads carry no target coordinates (no sentinels)."""
    processed = emit_map_open_data_processed(duration_ms=900)
    assert processed["detail"] == {}
    stalled = emit_map_open_stall_timeout(duration_ms=8000, timeout_ms=8000)
    assert stalled["detail"] == {"timeout_ms": 8000}
    rejected = emit_map_open_command_rejected(duration_ms=100, error_code=0)
    assert rejected["detail"] == {"error_code": 0}


def test_shoot_outcomes_carry_target_identity() -> None:
    """Shoot payloads carry target identity, not fabricated coords."""
    hit = emit_shoot_hit(
        duration_ms=2000,
        target_id=530,
        target_name="orange-3",
        victim_id=-1,
        on_intended_target=True,
        hit_signal="ammo_delta",
    )
    assert hit["detail"]["hit_signal"] == "ammo_delta"
    assert hit["detail"]["victim_id"] == -1
    miss = emit_shoot_miss(duration_ms=2100, target_id=530, target_name="orange-3")
    assert miss["outcome"] == "miss"
    rejected = emit_shoot_command_rejected(
        duration_ms=500, target_id=530, target_name="orange-3", error_code=0
    )
    assert rejected["detail"]["error_code"] == 0
    discarded = emit_shoot_discarded_target_not_tracked(target_x=10, target_y=20, target_id=530)
    assert discarded["outcome"] == "discarded_target_not_tracked"


def test_teleport_landed_classifies_exact_vs_inexact() -> None:
    """Landing on the requested tile is exact; displacement is inexact."""
    exact = emit_teleport_landed(
        duration_ms=300, target_x=10, target_y=20, landed_x=10, landed_y=20, messages=[]
    )
    assert exact["outcome"] == "landed_exact"
    inexact = emit_teleport_landed(
        duration_ms=300, target_x=10, target_y=20, landed_x=11, landed_y=20, messages=[]
    )
    assert inexact["outcome"] == "landed_inexact"
    assert exact["detail"]["sent_window"] == "(none)"


def test_teleport_dispatch_context_flows_into_the_outcome() -> None:
    """A recorded dispatch enriches the landing with wire windows."""
    record_teleport_dispatch(
        target_x=10, target_y=20, message_index=1, sent_window="fuel=800 pos=(9,20)"
    )
    landed = emit_teleport_landed(
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
        duration_ms=9000, target_x=1, target_y=2, timeout_ms=9000, messages=[]
    )
    assert follow_up["detail"]["sent_window"] == "(none)"


def test_teleport_discards_cover_all_four_classes() -> None:
    """Every executor teleport-discard class is recordable."""
    emit_teleport_discarded_hostile_mine(target_x=1, target_y=2)
    emit_teleport_discarded_combat_target_stale(target_x=1, target_y=2, target_id=530)
    emit_teleport_discarded_resource_target_stale(target_x=1, target_y=2, resource_kind="fuel")
    emit_teleport_discarded_resource_target_invalid(target_x=1, target_y=2, source="world_state")
    emit_teleport_command_rejected(
        duration_ms=100, target_x=1, target_y=2, error_code=0, messages=[]
    )
    assert sorted(outcome_counts("teleport")) == [
        "command_rejected",
        "discarded_combat_target_stale",
        "discarded_hostile_mine",
        "discarded_resource_target_invalid",
        "discarded_resource_target_stale",
    ]


def test_action_kinds_cover_the_six_ledgered_actions() -> None:
    """The ledger records exactly the six real action kinds."""
    assert ACTION_KINDS == ("scan", "move", "teleport", "collect", "map_open", "shoot")
