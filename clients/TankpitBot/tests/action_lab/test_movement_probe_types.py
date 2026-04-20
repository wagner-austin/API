"""Tests for movement probe session types."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.action_lab.movement_probe_types import (
    MovementProbeAttemptResultDict,
    MovementProbeSessionDict,
    decode_movement_probe_attempt_result,
    decode_movement_probe_session,
    encode_movement_probe_attempt_result,
    encode_movement_probe_session,
)
from tankpit_bot.action_lab.types import TeleportTargetDict


def _attempt() -> MovementProbeAttemptResultDict:
    """Build a sample movement attempt payload."""
    return MovementProbeAttemptResultDict(
        target=TeleportTargetDict(label="move_1", x=120, y=121),
        status="arrived_exact",
        move_started_ms=1000,
        map_open_requested_ms=1200,
        map_open_message_timestamp_ms=1210,
        completion_timestamp_ms=1800,
        move_elapsed_ms=800,
        fuel_before=900,
        fuel_after=890,
        world_timestamp_before=990,
        world_timestamp_after=1790,
        settled_x=120,
        settled_y=121,
        message_start_index=10,
        message_end_index=18,
    )


def _session() -> MovementProbeSessionDict:
    """Build a sample movement session payload."""
    return MovementProbeSessionDict(
        session_id="movement-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        max_targets=3,
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
        targets=[
            TeleportTargetDict(label="move_1", x=120, y=121),
            TeleportTargetDict(label="move_2", x=124, y=125),
        ],
        attempts=[_attempt()],
    )


def test_encode_decode_attempt_round_trips() -> None:
    """Movement attempts round-trip through encode/decode."""
    attempt = _attempt()
    encoded = encode_movement_probe_attempt_result(attempt)
    decoded = decode_movement_probe_attempt_result(encoded)
    assert decoded == attempt


def test_encode_decode_timeout_attempt_with_optional_nulls_round_trips() -> None:
    """Timeout attempts preserve nullable fields through encode/decode."""
    attempt = _attempt()
    attempt["status"] = "move_timeout"
    attempt["map_open_requested_ms"] = None
    attempt["map_open_message_timestamp_ms"] = None
    attempt["fuel_after"] = None
    attempt["settled_x"] = None
    attempt["settled_y"] = None
    encoded = encode_movement_probe_attempt_result(attempt)
    decoded = decode_movement_probe_attempt_result(encoded)
    assert decoded == attempt


def test_decode_attempt_rejects_invalid_status() -> None:
    """Movement attempt decoding rejects unsupported statuses."""
    encoded = encode_movement_probe_attempt_result(_attempt())
    encoded["status"] = "bad"
    with pytest.raises(JSONTypeError, match="invalid movement probe status"):
        decode_movement_probe_attempt_result(encoded)


def test_decode_attempt_rejects_non_object_target() -> None:
    """Movement attempt decoding requires an object target payload."""
    encoded = encode_movement_probe_attempt_result(_attempt())
    encoded["target"] = "bad"
    with pytest.raises(JSONTypeError, match="Field 'target' must be an object"):
        decode_movement_probe_attempt_result(encoded)


def test_decode_attempt_rejects_boolean_optional_int() -> None:
    """Movement attempt decoding rejects boolean values in optional int fields."""
    encoded = encode_movement_probe_attempt_result(_attempt())
    encoded["fuel_after"] = True
    with pytest.raises(JSONTypeError, match="integer or null"):
        decode_movement_probe_attempt_result(encoded)


def test_encode_decode_session_round_trips() -> None:
    """Movement sessions round-trip through encode/decode."""
    session = _session()
    encoded = encode_movement_probe_session(session)
    decoded = decode_movement_probe_session(encoded)
    assert decoded == session


def test_decode_session_rejects_non_bool_queue_flag() -> None:
    """Movement sessions require a boolean queue-map-open flag."""
    encoded = encode_movement_probe_session(_session())
    encoded["queue_map_open_during_move"] = "true"
    with pytest.raises(JSONTypeError, match="must be a boolean"):
        decode_movement_probe_session(encoded)


def test_decode_session_rejects_non_object_target_item() -> None:
    """Movement session decoding requires object targets."""
    encoded = encode_movement_probe_session(_session())
    encoded["targets"] = ["bad"]
    with pytest.raises(JSONTypeError, match="Field 'targets' must contain only objects"):
        decode_movement_probe_session(encoded)


def test_decode_session_rejects_non_object_attempt_item() -> None:
    """Movement session decoding requires object attempts."""
    encoded = encode_movement_probe_session(_session())
    encoded["attempts"] = ["bad"]
    with pytest.raises(JSONTypeError, match="Field 'attempts' must contain only objects"):
        decode_movement_probe_session(encoded)


def test_decode_session_rejects_non_object_startup_timing() -> None:
    """Movement session decoding requires an object startup timing payload."""
    encoded = encode_movement_probe_session(_session())
    encoded["startup_timing"] = "bad"
    with pytest.raises(JSONTypeError, match="Field 'startup_timing' must be an object"):
        decode_movement_probe_session(encoded)
