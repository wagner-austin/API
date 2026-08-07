"""Tests for the teleport probe's report formatting.

Attempt-window entries, page snapshots, map-data index lookup, and the
run summary.
"""

from __future__ import annotations

import base64

import pytest
from tests.action_lab._replay_cdp import StubSnapshotCDPSession
from tests.action_lab._teleport_harness import (
    _make_attempt,
    _make_world,
    _SequencedProvider,
)

from tankpit_bot.action_lab.teleport_helpers import (
    TeleportProbeError,
    _find_map_data_message_index,
    _format_attempt_window_entries,
    _format_page_snapshots,
    _start_teleport_page_snapshots,
    format_teleport_probe_summary,
)
from tankpit_bot.action_lab.types import (
    TeleportProbeSessionDict,
)
from tankpit_bot.types import (
    CapturedMessage,
)


def test_format_attempt_window_entries_filters_direction_and_reports_more() -> None:
    provider = _SequencedProvider([_make_world(900, 100, 100, 900)])
    provider._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload="$1|0",
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1001,
            direction="sent",
            payload="%AUTH",
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1002,
            direction="sent",
            payload="%MOVE",
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1003,
            direction="sent",
            payload="%RADAR",
            ws_url="wss://tankpit.com/ws/",
        ),
    ]

    summary = _format_attempt_window_entries(
        provider,
        message_start_index=0,
        direction="sent",
        limit=2,
    )

    assert "1:" in summary
    assert "2:" in summary
    assert "...+1 more" in summary


def test_format_attempt_window_entries_returns_exact_window_without_more_suffix() -> None:
    provider = _SequencedProvider([_make_world(900, 100, 100, 900)])
    provider._messages = [
        CapturedMessage(
            timestamp_ms=1001,
            direction="sent",
            payload="%AUTH",
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1002,
            direction="sent",
            payload="%MOVE",
            ws_url="wss://tankpit.com/ws/",
        ),
    ]

    summary = _format_attempt_window_entries(
        provider,
        message_start_index=0,
        direction="sent",
        limit=2,
    )

    assert "0:" in summary
    assert "1:" in summary
    assert "...+" not in summary


def test_format_attempt_window_entries_for_received_messages_omits_sent_metadata() -> None:
    provider = _SequencedProvider([_make_world(900, 100, 100, 900)])
    provider._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload="bad",
            ws_url="wss://tankpit.com/ws/",
        )
    ]

    summary = _format_attempt_window_entries(
        provider,
        message_start_index=0,
        direction="received",
        limit=6,
    )

    assert "origin=" not in summary


def test_format_attempt_window_entries_includes_sent_origin_metadata() -> None:
    provider = _SequencedProvider([_make_world(900, 100, 100, 900)])
    provider._messages = [
        CapturedMessage(
            timestamp_ms=1001,
            direction="sent",
            payload="%AUTH",
            ws_url="wss://tankpit.com/ws/",
            sent_origin="bot_injected",
            sent_label="teleport(129,106)",
        ),
        CapturedMessage(
            timestamp_ms=1002,
            direction="sent",
            payload="%MOVE",
            ws_url="wss://tankpit.com/ws/",
            sent_origin="page_client",
        ),
    ]

    summary = _format_attempt_window_entries(
        provider,
        message_start_index=0,
        direction="sent",
        limit=6,
    )

    assert "origin=bot_injected label=teleport(129,106)" in summary
    assert "origin=page_client" in summary


def test_format_page_snapshots_returns_none_for_empty_list() -> None:
    assert _format_page_snapshots([]) == "none"


def test_find_map_data_message_index_skips_earlier_and_sent_messages() -> None:
    provider = _SequencedProvider([_make_world(900, 100, 100, 900)])
    map_data_payload = base64.b64encode(bytes([0, 0, 0x2E]) + bytes(600)).decode("ascii")
    provider._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload=map_data_payload,
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1001,
            direction="sent",
            payload=map_data_payload,
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1002,
            direction="received",
            payload=map_data_payload,
            ws_url="wss://tankpit.com/ws/",
        ),
    ]

    result = _find_map_data_message_index(
        provider,
        message_start_index=1,
        scan_start_index=0,
    )

    assert result == 2


def test_find_map_data_message_index_skips_non_map_data_received_messages() -> None:
    provider = _SequencedProvider([_make_world(900, 100, 100, 900)])
    map_data_payload = base64.b64encode(bytes([0, 0, 0x2E]) + bytes(600)).decode("ascii")
    provider._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload="bad",
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1001,
            direction="received",
            payload=map_data_payload,
            ws_url="wss://tankpit.com/ws/",
        ),
    ]

    result = _find_map_data_message_index(
        provider,
        message_start_index=0,
        scan_start_index=0,
    )

    assert result == 1


def test_start_teleport_page_snapshots_rejects_missing_cdp() -> None:
    with pytest.raises(TeleportProbeError, match="cdp session is unavailable"):
        _start_teleport_page_snapshots(
            cdp=None,
            capture_before_map_open=True,
            unavailable_error=TeleportProbeError,
            unavailable_message="cdp session is unavailable",
        )


def test_start_teleport_page_snapshots_can_skip_initial_capture() -> None:
    snapshots, capture_page_snapshot = _start_teleport_page_snapshots(
        cdp=StubSnapshotCDPSession(),
        capture_before_map_open=False,
        unavailable_error=TeleportProbeError,
        unavailable_message="cdp session is unavailable",
    )

    assert snapshots == []
    snapshot = capture_page_snapshot("timeout")
    assert snapshot["phase"] == "timeout"


def test_format_teleport_probe_summary_counts_statuses() -> None:
    session = TeleportProbeSessionDict(
        session_id="summary",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        teleport_strategy="sync_before_teleport",
        max_targets=4,
        capture_session_path="teleport_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 100,
            "intel_ready_timestamp_ms": 150,
            "initial_sync_started_ms": 200,
            "initial_world_timestamp_ms": 250,
            "command_ready_timestamp_ms": 300,
            "first_attempt_started_ms": 325,
            "game_ready_to_intel_ready_ms": 50,
            "intel_ready_to_initial_world_ms": 100,
            "initial_world_to_command_ready_ms": 50,
            "command_ready_to_first_attempt_ms": 25,
        },
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=500,
        targets=[],
        attempts=[
            _make_attempt("landed_exact"),
            _make_attempt("landed_offset"),
            _make_attempt("map_sync_timeout"),
            _make_attempt("teleport_timeout"),
        ],
    )
    assert format_teleport_probe_summary(session) == (
        "Teleport probe complete: strategy=sync_before_teleport attempts=4 exact=1 "
        "offset=1 map_sync_timeout=1 teleport_timeout=1 "
        "session_to_initial_sync_ms=199 initial_sync_to_command_ready_ms=100"
    )
