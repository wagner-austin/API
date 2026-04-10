"""Tests for action-lab teleport TypedDict codecs."""

from __future__ import annotations

from typing import Literal

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportProbeSessionDict,
    TeleportStartupTimingDict,
    TeleportTargetDict,
    decode_teleport_attempt_result,
    decode_teleport_probe_session,
    decode_teleport_startup_timing,
    decode_teleport_target,
    encode_teleport_attempt_result,
    encode_teleport_probe_session,
    encode_teleport_startup_timing,
    encode_teleport_target,
)


def _sample_target() -> TeleportTargetDict:
    """Build a sample teleport target."""
    return TeleportTargetDict(label="box_r0_c0", x=150, y=171)


def _sample_attempt(
    status: Literal["landed_exact", "landed_offset", "map_sync_timeout", "teleport_timeout"] = (
        "landed_exact"
    ),
) -> TeleportAttemptResultDict:
    """Build a sample teleport attempt result."""
    return TeleportAttemptResultDict(
        target=_sample_target(),
        status=status,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        completion_timestamp_ms=1500,
        map_sync_elapsed_ms=200,
        teleport_elapsed_ms=200,
        fuel_before=900,
        fuel_after=840,
        world_timestamp_before=900,
        world_timestamp_after=1450,
        landed_signal_received=True,
        landed_x=150,
        landed_y=171,
        message_start_index=10,
        message_end_index=14,
    )


def _sample_session() -> TeleportProbeSessionDict:
    """Build a sample teleport probe session."""
    return TeleportProbeSessionDict(
        session_id="teleport-session",
        start_timestamp_ms=100,
        end_timestamp_ms=1000,
        base_url="https://tankpit.com/play",
        spawn_x=158,
        spawn_y=132,
        teleport_strategy="sync_before_teleport",
        max_targets=3,
        capture_session_path="teleport_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing=TeleportStartupTimingDict(
            game_ready_timestamp_ms=300,
            intel_ready_timestamp_ms=350,
            initial_sync_started_ms=400,
            initial_world_timestamp_ms=450,
            command_ready_timestamp_ms=460,
            first_attempt_started_ms=500,
            game_ready_to_intel_ready_ms=50,
            intel_ready_to_initial_world_ms=100,
            initial_world_to_command_ready_ms=10,
            command_ready_to_first_attempt_ms=40,
        ),
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=500,
        targets=[_sample_target()],
        attempts=[_sample_attempt()],
    )


def _sample_startup_timing() -> TeleportStartupTimingDict:
    """Build a sample startup timing payload."""
    return _sample_session()["startup_timing"]


def test_target_round_trip() -> None:
    """Teleport targets encode and decode cleanly."""
    encoded = encode_teleport_target(_sample_target())
    decoded = decode_teleport_target(encoded)
    assert decoded == _sample_target()


def test_attempt_round_trip() -> None:
    """Teleport attempt results encode and decode cleanly."""
    encoded = encode_teleport_attempt_result(_sample_attempt())
    decoded = decode_teleport_attempt_result(encoded)
    assert decoded == _sample_attempt()


def test_attempt_round_trip_with_null_fields() -> None:
    """Optional attempt fields preserve null values through codecs."""
    attempt = TeleportAttemptResultDict(
        target=_sample_target(),
        status="map_sync_timeout",
        map_open_started_ms=1000,
        map_sync_timestamp_ms=None,
        teleport_started_ms=None,
        completion_timestamp_ms=4000,
        map_sync_elapsed_ms=None,
        teleport_elapsed_ms=None,
        fuel_before=900,
        fuel_after=900,
        world_timestamp_before=950,
        world_timestamp_after=950,
        landed_signal_received=False,
        landed_x=158,
        landed_y=132,
        message_start_index=10,
        message_end_index=11,
    )
    decoded = decode_teleport_attempt_result(encode_teleport_attempt_result(attempt))
    assert decoded == attempt


def test_decode_attempt_rejects_invalid_status() -> None:
    """Attempt decode rejects unsupported status strings."""
    encoded = encode_teleport_attempt_result(_sample_attempt())
    encoded["status"] = "bad"
    with pytest.raises(JSONTypeError, match="invalid teleport attempt status"):
        decode_teleport_attempt_result(encoded)


def test_decode_attempt_accepts_landed_offset_status() -> None:
    """Attempt decode accepts landed_offset status values."""
    encoded = encode_teleport_attempt_result(_sample_attempt("landed_offset"))
    decoded = decode_teleport_attempt_result(encoded)
    assert decoded["status"] == "landed_offset"


def test_decode_attempt_accepts_teleport_timeout_status() -> None:
    """Attempt decode accepts teleport_timeout status values."""
    encoded = encode_teleport_attempt_result(_sample_attempt("teleport_timeout"))
    decoded = decode_teleport_attempt_result(encoded)
    assert decoded["status"] == "teleport_timeout"


def test_decode_attempt_rejects_invalid_optional_int() -> None:
    """Attempt decode rejects malformed optional integer fields."""
    encoded = encode_teleport_attempt_result(_sample_attempt())
    encoded["teleport_elapsed_ms"] = "nope"
    with pytest.raises(JSONTypeError, match="teleport_elapsed_ms"):
        decode_teleport_attempt_result(encoded)


def test_decode_attempt_rejects_non_object_target() -> None:
    """Attempt decode rejects non-object target payloads."""
    encoded = encode_teleport_attempt_result(_sample_attempt())
    encoded["target"] = "bad"
    with pytest.raises(JSONTypeError, match="Field 'target' must be an object"):
        decode_teleport_attempt_result(encoded)


def test_decode_attempt_rejects_non_boolean_landed_signal() -> None:
    """Attempt decode rejects non-boolean landed signal values."""
    encoded = encode_teleport_attempt_result(_sample_attempt())
    encoded["landed_signal_received"] = "bad"
    with pytest.raises(JSONTypeError, match="landed_signal_received"):
        decode_teleport_attempt_result(encoded)


def test_session_round_trip() -> None:
    """Teleport sessions encode and decode cleanly."""
    encoded = encode_teleport_probe_session(_sample_session())
    decoded = decode_teleport_probe_session(encoded)
    assert decoded == _sample_session()


def test_startup_timing_round_trip() -> None:
    """Startup timing payloads encode and decode cleanly."""
    encoded = encode_teleport_startup_timing(_sample_startup_timing())
    decoded = decode_teleport_startup_timing(encoded)
    assert decoded == _sample_startup_timing()


def test_session_decode_rejects_non_object_target() -> None:
    """Session decode rejects non-object target entries."""
    encoded = encode_teleport_probe_session(_sample_session())
    encoded["targets"] = ["bad"]
    with pytest.raises(JSONTypeError, match="targets"):
        decode_teleport_probe_session(encoded)


def test_session_decode_rejects_non_object_attempt() -> None:
    """Session decode rejects non-object attempt entries."""
    encoded = encode_teleport_probe_session(_sample_session())
    encoded["attempts"] = ["bad"]
    with pytest.raises(JSONTypeError, match="attempts"):
        decode_teleport_probe_session(encoded)


def test_session_decode_rejects_invalid_strategy() -> None:
    """Session decode rejects unsupported teleport strategies."""
    encoded = encode_teleport_probe_session(_sample_session())
    encoded["teleport_strategy"] = "bad"
    with pytest.raises(JSONTypeError, match="invalid teleport strategy"):
        decode_teleport_probe_session(encoded)


def test_session_decode_accepts_immediate_strategy() -> None:
    """Session decode accepts the immediate teleport strategy."""
    encoded = encode_teleport_probe_session(_sample_session())
    encoded["teleport_strategy"] = "immediate_after_map_open"
    decoded = decode_teleport_probe_session(encoded)
    assert decoded["teleport_strategy"] == "immediate_after_map_open"


def test_session_decode_rejects_non_object_startup_timing() -> None:
    """Session decode rejects non-object startup timing payloads."""
    encoded = encode_teleport_probe_session(_sample_session())
    encoded["startup_timing"] = "bad"
    with pytest.raises(JSONTypeError, match="startup_timing"):
        decode_teleport_probe_session(encoded)
