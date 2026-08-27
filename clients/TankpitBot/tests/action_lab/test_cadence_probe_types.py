"""Codec tests for the fire-cadence probe session types."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.action_lab.cadence_probe_types import (
    CadenceBurstDict,
    CadenceProbeSessionDict,
    CadenceShotDict,
    decode_cadence_burst,
    decode_cadence_probe_session,
    decode_cadence_shot,
    encode_cadence_burst,
    encode_cadence_probe_session,
    encode_cadence_shot,
)
from tankpit_bot.action_lab.types import TeleportStartupTimingDict


def _shot(shot_number: int = 1) -> CadenceShotDict:
    """Build one dispatched shot."""
    return CadenceShotDict(
        shot_number=shot_number,
        dispatched_ms=1000 + shot_number,
        target_x=101,
        target_y=100,
    )


def _burst(*, killed: bool = False) -> CadenceBurstDict:
    """Build one complete burst record."""
    return CadenceBurstDict(
        spacing_ms=500,
        target_id=900,
        target_name="orange-1",
        shots=[_shot(1), _shot(2)],
        dispatched=2,
        dual_before=40,
        dual_after=38,
        homing_before=20,
        homing_after=20,
        fuel_before=1000,
        fuel_after=980,
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


def _session() -> CadenceProbeSessionDict:
    """Build one complete session record."""
    return CadenceProbeSessionDict(
        session_id="cadence-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        shots_per_burst=6,
        capture_session_path="cadence_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing=_startup_timing(),
        bursts=[_burst(), _burst(killed=True)],
    )


def test_shot_round_trips_through_the_codec() -> None:
    """Encode then decode reproduces the shot exactly."""
    shot = _shot()
    assert decode_cadence_shot(encode_cadence_shot(shot)) == shot


def test_burst_round_trips_through_the_codec() -> None:
    """Encode then decode reproduces the burst exactly."""
    burst = _burst(killed=True)
    assert decode_cadence_burst(encode_cadence_burst(burst)) == burst


def test_session_round_trips_through_the_codec() -> None:
    """Encode then decode reproduces the session exactly."""
    session = _session()
    assert decode_cadence_probe_session(encode_cadence_probe_session(session)) == session


def test_burst_shots_must_contain_objects() -> None:
    """A non-object shot entry is rejected loudly."""
    encoded = encode_cadence_burst(_burst())
    encoded["shots"] = [1]
    with pytest.raises(JSONTypeError, match="shots must contain objects"):
        decode_cadence_burst(encoded)


def test_burst_target_killed_must_be_a_boolean() -> None:
    """A non-boolean kill flag is rejected loudly."""
    encoded = encode_cadence_burst(_burst())
    encoded["target_killed"] = 1
    with pytest.raises(JSONTypeError, match="must be a boolean"):
        decode_cadence_burst(encoded)


def test_session_bursts_must_contain_objects() -> None:
    """A non-object burst entry is rejected loudly."""
    encoded = encode_cadence_probe_session(_session())
    encoded["bursts"] = ["not-a-burst"]
    with pytest.raises(JSONTypeError, match="bursts must contain objects"):
        decode_cadence_probe_session(encoded)


def test_session_startup_timing_must_be_an_object() -> None:
    """A non-object startup timing block is rejected loudly."""
    encoded = encode_cadence_probe_session(_session())
    encoded["startup_timing"] = "later"
    with pytest.raises(JSONTypeError, match="must be an object"):
        decode_cadence_probe_session(encoded)
