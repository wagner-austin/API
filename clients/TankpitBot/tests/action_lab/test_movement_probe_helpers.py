"""Tests for the movement probe's standalone helpers.

Target building, probe construction, terrain lookup, and the run
summary.
"""

from __future__ import annotations

from tests.action_lab._movement_probe_harness import (
    _make_attempt,
    _TerrainMapStub,
)

from tankpit_bot.action_lab.movement_probe import (
    MovementProbe,
    _build_probe_targets,
    _create_movement_probe,
    _find_first_sent_label_timestamp,
    _get_probe_terrain_map,
    _require_positive,
    format_movement_probe_summary,
)
from tankpit_bot.action_lab.movement_probe_types import (
    MovementProbeSessionDict,
)
from tankpit_bot.types import (
    CapturedMessage,
)


def test_find_first_sent_label_timestamp_returns_first_matching_bot_send() -> None:
    messages = [
        CapturedMessage(
            timestamp_ms=10,
            direction="sent",
            payload="a",
            ws_url="wss://x",
            sent_origin="page_client",
        ),
        CapturedMessage(timestamp_ms=20, direction="received", payload="b", ws_url="wss://x"),
        CapturedMessage(
            timestamp_ms=30,
            direction="sent",
            payload="c",
            ws_url="wss://x",
            sent_origin="bot_injected",
            sent_label="map_open",
        ),
        CapturedMessage(
            timestamp_ms=40,
            direction="sent",
            payload="d",
            ws_url="wss://x",
            sent_origin="bot_injected",
            sent_label="map_open",
        ),
    ]
    assert _find_first_sent_label_timestamp(messages, start_index=0, label="map_open") == 30
    assert _find_first_sent_label_timestamp(messages, start_index=3, label="map_open") == 40
    assert _find_first_sent_label_timestamp(messages, start_index=0, label="move") is None


def test_require_positive_returns_value() -> None:
    assert _require_positive(5, "max_targets") == 5


def test_create_movement_probe_returns_concrete_probe() -> None:
    probe = _create_movement_probe(
        "https://tankpit.com/play",
        headless=True,
        prefer_account=False,
    )
    assert type(probe) is MovementProbe
    assert probe._target_url == "https://tankpit.com/play"
    assert probe._headless is True


def test_get_probe_terrain_map_defaults_to_none_without_loaded_map() -> None:
    assert _get_probe_terrain_map() is None


def test_build_probe_targets_uses_real_target_builder() -> None:
    targets = _build_probe_targets(
        100,
        104,
        _TerrainMapStub(),
        max_targets=2,
    )
    assert len(targets) == 2
    assert targets[0]["label"].startswith("fuel_ground_")


def test_format_movement_probe_summary_counts_statuses() -> None:
    session = MovementProbeSessionDict(
        session_id="movement-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        max_targets=2,
        capture_session_path="movement_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 300,
            "intel_ready_timestamp_ms": 350,
            "initial_sync_started_ms": 400,
            "initial_world_timestamp_ms": 450,
            "command_ready_timestamp_ms": 460,
            "first_attempt_started_ms": 500,
            "game_ready_to_intel_ready_ms": 50,
            "intel_ready_to_initial_world_ms": 100,
            "initial_world_to_command_ready_ms": 10,
            "command_ready_to_first_attempt_ms": 40,
        },
        move_timeout_ms=5000,
        settle_delay_ms=500,
        queue_map_open_during_move=True,
        map_open_delay_ms=150,
        targets=[],
        attempts=[_make_attempt("arrived_exact"), _make_attempt("move_timeout")],
    )
    summary = format_movement_probe_summary(session)
    assert "arrived_exact=1" in summary
    assert "move_timeout=1" in summary
