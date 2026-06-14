"""End-to-end tests for bot-side teleport attempt diagnostics.

Every test drives the REAL pipeline:
:func:`tankpit_bot.runtime_logging.configure_bot_runtime_logging` ->
:func:`tankpit_bot.diagnostics.teleport_attempts.record_teleport_dispatch`
/ :func:`emit_teleport_attempt_outcome` -> JSONL via
:class:`tests.conftest.FakeFileSystem` -> real
:func:`tankpit_bot.diagnostics.issue_report.build_issue_report`.
Nothing is mocked: the emitted records classify through the same
issue-report path as action-lab teleport attempts.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import FakeFileSystem

from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.diagnostics.issue_report import build_issue_report
from tankpit_bot.diagnostics.teleport_attempts import (
    emit_teleport_attempt_outcome,
    record_teleport_dispatch,
    reset_teleport_attempt_tracking,
)
from tankpit_bot.runtime_logging import configure_bot_runtime_logging
from tankpit_bot.types.message import CapturedMessage


def _make_snapshot() -> PageClientSnapshotDict:
    """Return a healthy live-client snapshot for dispatch context."""
    return PageClientSnapshotDict(
        timestamp_ms=1000,
        client_present=True,
        map_visible=True,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=50,
        last_page_client_send_age_ms=100,
        last_bot_send_age_ms=16,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        map_fields={},
        world_collections={},
    )


def _message(direction: str, payload: str) -> CapturedMessage:
    """Return a captured message for window assembly."""
    sent: CapturedMessage = CapturedMessage(
        timestamp_ms=1000,
        direction="sent",
        payload=payload,
        ws_url="wss://tankpit.com/ws/",
    )
    received: CapturedMessage = CapturedMessage(
        timestamp_ms=1000,
        direction="received",
        payload=payload,
        ws_url="wss://tankpit.com/ws/",
    )
    return sent if direction == "sent" else received


def test_landed_attempt_classifies_as_success_in_issue_report(
    fake_fs: FakeFileSystem,
) -> None:
    """A dispatch + landed outcome appears as a successful teleport attempt."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    messages = [_message("sent", "QUFB"), _message("received", "TUFQX0RBVEE=")]
    record_teleport_dispatch(
        target_x=161,
        target_y=109,
        message_index=0,
        sent_window=(
            "map_visible=True pending_actions=0 ws_ready_state=1"
            " heartbeat_age_ms=50 page_send_age_ms=100 bot_send_age_ms=16"
        ),
    )

    emitted = emit_teleport_attempt_outcome(status="landed_exact", messages=messages)

    assert emitted is True
    report = build_issue_report(Path(artifacts["latest_events_path"]))
    assert report["teleport_success_count"] == 1
    assert report["teleport_failure_count"] == 0
    attempt = report["teleport_attempts"][0]
    assert attempt["target_x"] == 161
    assert attempt["target_y"] == 109
    assert attempt["teleport_cycle_id"] == 1
    assert attempt["status"] == "landed_exact"
    assert attempt["sent_window"] == (
        "map_visible=True pending_actions=0 ws_ready_state=1 "
        "heartbeat_age_ms=50 page_send_age_ms=100 bot_send_age_ms=16"
    )
    assert attempt["received_window"] == "sent:4:QUFB | received:12:TUFQX0RBVEE="


def test_stall_attempt_classifies_as_failure_in_issue_report(
    fake_fs: FakeFileSystem,
) -> None:
    """A dispatch + stall outcome appears as a failed teleport attempt."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    record_teleport_dispatch(
        target_x=226,
        target_y=106,
        message_index=0,
        sent_window=(
            "map_visible=True pending_actions=0 ws_ready_state=1"
            " heartbeat_age_ms=50 page_send_age_ms=100 bot_send_age_ms=16"
        ),
    )

    emitted = emit_teleport_attempt_outcome(status="stall_timeout", messages=[])

    assert emitted is True
    report = build_issue_report(Path(artifacts["latest_events_path"]))
    assert report["teleport_success_count"] == 0
    assert report["teleport_failure_count"] == 1
    assert report["teleport_attempts"][0]["received_window"] == "(none)"


def test_outcome_without_pending_dispatch_emits_nothing(
    fake_fs: FakeFileSystem,
) -> None:
    """A completion gate firing with no recorded dispatch is a no-op."""
    artifacts = configure_bot_runtime_logging("20260610-120000")

    emitted = emit_teleport_attempt_outcome(status="landed_exact", messages=[])

    assert emitted is False
    report = build_issue_report(Path(artifacts["latest_events_path"]))
    assert report["teleport_attempts"] == []


def test_window_keeps_only_the_last_twelve_messages(fake_fs: FakeFileSystem) -> None:
    """Long exchanges truncate to the latest twelve window entries."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    messages = [_message("received", f"payload-{index:02d}") for index in range(20)]
    record_teleport_dispatch(
        target_x=161,
        target_y=109,
        message_index=0,
        sent_window=(
            "map_visible=True pending_actions=0 ws_ready_state=1"
            " heartbeat_age_ms=50 page_send_age_ms=100 bot_send_age_ms=16"
        ),
    )

    assert emit_teleport_attempt_outcome(status="landed_exact", messages=messages)

    report = build_issue_report(Path(artifacts["latest_events_path"]))
    window = report["teleport_attempts"][0]["received_window"]
    assert window.count("|") == 11
    assert "payload-19" in window
    assert "payload-07" not in window


def test_window_starts_at_the_dispatch_message_index(fake_fs: FakeFileSystem) -> None:
    """Messages exchanged before the dispatch never enter the window."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    messages = [
        _message("received", "before-dispatch"),
        _message("sent", "teleport-bytes"),
        _message("received", "after-dispatch"),
    ]
    record_teleport_dispatch(
        target_x=161,
        target_y=109,
        message_index=1,
        sent_window=(
            "map_visible=True pending_actions=0 ws_ready_state=1"
            " heartbeat_age_ms=50 page_send_age_ms=100 bot_send_age_ms=16"
        ),
    )

    assert emit_teleport_attempt_outcome(status="landed_inexact", messages=messages)

    report = build_issue_report(Path(artifacts["latest_events_path"]))
    window = report["teleport_attempts"][0]["received_window"]
    assert "before-dispatch" not in window
    assert "teleport-bytes" in window
    assert "after-dispatch" in window


def test_cycle_counter_increments_per_dispatch(fake_fs: FakeFileSystem) -> None:
    """Each dispatch gets a fresh monotonic cycle id."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    for _ in range(2):
        record_teleport_dispatch(
            target_x=161,
            target_y=109,
            message_index=0,
            sent_window=(
                "map_visible=True pending_actions=0 ws_ready_state=1"
                " heartbeat_age_ms=50 page_send_age_ms=100 bot_send_age_ms=16"
            ),
        )
        assert emit_teleport_attempt_outcome(status="landed_exact", messages=[])

    report = build_issue_report(Path(artifacts["latest_events_path"]))
    assert [a["teleport_cycle_id"] for a in report["teleport_attempts"]] == [1, 2]


def test_reset_drops_pending_dispatch(fake_fs: FakeFileSystem) -> None:
    """``reset_teleport_attempt_tracking`` clears the pending attempt."""
    configure_bot_runtime_logging("20260610-120000")
    record_teleport_dispatch(
        target_x=161,
        target_y=109,
        message_index=0,
        sent_window=(
            "map_visible=True pending_actions=0 ws_ready_state=1"
            " heartbeat_age_ms=50 page_send_age_ms=100 bot_send_age_ms=16"
        ),
    )

    reset_teleport_attempt_tracking()

    assert emit_teleport_attempt_outcome(status="landed_exact", messages=[]) is False
