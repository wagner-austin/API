"""Codec tests for the weave probe session types."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.action_lab.types import TeleportStartupTimingDict
from tankpit_bot.action_lab.weave_probe_types import (
    WeaveBeatDict,
    WeaveBurstDict,
    WeaveProbeSessionDict,
    decode_weave_beat,
    decode_weave_burst,
    decode_weave_probe_session,
    encode_weave_beat,
    encode_weave_burst,
    encode_weave_probe_session,
)


def _beat(beat_number: int = 1, *, moved: bool = False) -> WeaveBeatDict:
    """Build one beat."""
    return WeaveBeatDict(
        beat_number=beat_number,
        dispatched_ms=1000 + beat_number,
        target_x=101,
        target_y=100,
        moved=moved,
        move_x=100 if moved else -1,
        move_y=101 if moved else -1,
    )


def _burst(*, killed: bool = False) -> WeaveBurstDict:
    """Build one complete burst record."""
    return WeaveBurstDict(
        target_id=900,
        target_name="orange-1",
        beats=[_beat(1), _beat(2, moved=True)],
        shots_dispatched=2,
        moves_dispatched=1,
        dual_before=40,
        dual_after=38,
        homing_before=20,
        homing_after=20,
        fuel_before=1000,
        fuel_after=970,
        served_hits=2,
        target_killed=killed,
    )


def _startup_timing() -> TeleportStartupTimingDict:
    """Build the envelope startup timing block."""
    return TeleportStartupTimingDict(
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
    )


def _session() -> WeaveProbeSessionDict:
    """Build one complete session record."""
    return WeaveProbeSessionDict(
        session_id="weave-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        beats_per_burst=8,
        capture_session_path="weave_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing=_startup_timing(),
        bursts=[_burst(), _burst(killed=True)],
    )


def test_beat_round_trips_through_the_codec() -> None:
    """Encode then decode reproduces the beat exactly."""
    beat = _beat(2, moved=True)
    assert decode_weave_beat(encode_weave_beat(beat)) == beat


def test_burst_round_trips_through_the_codec() -> None:
    """Encode then decode reproduces the burst exactly."""
    burst = _burst(killed=True)
    assert decode_weave_burst(encode_weave_burst(burst)) == burst


def test_session_round_trips_through_the_codec() -> None:
    """Encode then decode reproduces the session exactly."""
    session = _session()
    assert decode_weave_probe_session(encode_weave_probe_session(session)) == session


def test_burst_beats_must_contain_objects() -> None:
    """A non-object beat entry is rejected loudly."""
    encoded = encode_weave_burst(_burst())
    encoded["beats"] = [1]
    with pytest.raises(JSONTypeError, match="beats must contain objects"):
        decode_weave_burst(encoded)


def test_session_bursts_must_contain_objects() -> None:
    """A non-object burst entry is rejected loudly."""
    encoded = encode_weave_probe_session(_session())
    encoded["bursts"] = ["not-a-burst"]
    with pytest.raises(JSONTypeError, match="bursts must contain objects"):
        decode_weave_probe_session(encoded)


def test_session_startup_timing_must_be_an_object() -> None:
    """A non-object startup timing block is rejected loudly."""
    encoded = encode_weave_probe_session(_session())
    encoded["startup_timing"] = "later"
    with pytest.raises(JSONTypeError, match="must be an object"):
        decode_weave_probe_session(encoded)
