"""Tests for combat probe type encoding/decoding."""

from __future__ import annotations

from typing import Literal

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.action_lab.combat_probe_types import (
    CombatEngagementDict,
    CombatProbeSessionDict,
    CombatShotResultDict,
    decode_combat_engagement,
    decode_combat_probe_session,
    decode_combat_shot_result,
    encode_combat_engagement,
    encode_combat_probe_session,
    encode_combat_shot_result,
)


def _make_shot(
    *,
    shot_number: int = 1,
    result: Literal["hit", "miss", "timeout"] = "hit",
    distance: int = 1,
    weapon_byte: int | None = 1,
) -> CombatShotResultDict:
    return CombatShotResultDict(
        shot_number=shot_number,
        self_x=10,
        self_y=20,
        target_x=11,
        target_y=20,
        distance=distance,
        result=result,
        weapon_byte=weapon_byte,
        target_name="purple-1",
        target_id=500,
        timestamp_ms=100000,
    )


def _make_engagement(
    shots: list[CombatShotResultDict] | None = None,
) -> CombatEngagementDict:
    return CombatEngagementDict(
        target_id=500,
        target_name="purple-1",
        initial_target_x=11,
        initial_target_y=20,
        initial_distance=1,
        landed_x=10,
        landed_y=20,
        shots=shots or [_make_shot()],
        total_hits=1,
        total_misses=0,
        total_timeouts=0,
        kill_confirmed=True,
        target_fled=False,
        final_target_x=11,
        final_target_y=20,
        final_distance=1,
    )


def _make_session(
    engagements: list[CombatEngagementDict] | None = None,
) -> CombatProbeSessionDict:
    return CombatProbeSessionDict(
        session_id="test-session-1",
        start_timestamp_ms=90000,
        end_timestamp_ms=120000,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        max_engagements=3,
        max_shots_per_engagement=20,
        capture_session_path="",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 91000,
            "intel_ready_timestamp_ms": 92000,
            "initial_sync_started_ms": 93000,
            "initial_world_timestamp_ms": 94000,
            "command_ready_timestamp_ms": 95000,
            "first_attempt_started_ms": 96000,
            "game_ready_to_intel_ready_ms": 1000,
            "intel_ready_to_initial_world_ms": 2000,
            "initial_world_to_command_ready_ms": 1000,
            "command_ready_to_first_attempt_ms": 1000,
        },
        engagements=[_make_engagement()] if engagements is None else engagements,
    )


class TestCombatShotResultCodec:
    """Round-trip encode/decode for CombatShotResultDict."""

    def test_round_trip_hit(self) -> None:
        shot = _make_shot(result="hit", weapon_byte=1)
        encoded = encode_combat_shot_result(shot)
        decoded = decode_combat_shot_result(encoded)
        assert decoded == shot

    def test_round_trip_miss(self) -> None:
        shot = _make_shot(result="miss", weapon_byte=None)
        encoded = encode_combat_shot_result(shot)
        decoded = decode_combat_shot_result(encoded)
        assert decoded == shot

    def test_round_trip_timeout(self) -> None:
        shot = _make_shot(result="timeout", weapon_byte=None)
        encoded = encode_combat_shot_result(shot)
        decoded = decode_combat_shot_result(encoded)
        assert decoded == shot

    def test_invalid_result_raises(self) -> None:
        encoded = encode_combat_shot_result(_make_shot())
        encoded["result"] = "invalid"
        with pytest.raises(JSONTypeError, match="invalid shot result"):
            decode_combat_shot_result(encoded)

    def test_optional_int_null(self) -> None:
        shot = _make_shot(weapon_byte=None)
        encoded = encode_combat_shot_result(shot)
        assert encoded["weapon_byte"] is None
        decoded = decode_combat_shot_result(encoded)
        assert decoded["weapon_byte"] is None

    def test_optional_int_invalid_type(self) -> None:
        encoded = encode_combat_shot_result(_make_shot())
        encoded["weapon_byte"] = "not_an_int"
        with pytest.raises(JSONTypeError, match="must be an integer or null"):
            decode_combat_shot_result(encoded)

    def test_optional_int_bool_rejected(self) -> None:
        encoded = encode_combat_shot_result(_make_shot())
        encoded["weapon_byte"] = True
        with pytest.raises(JSONTypeError, match="must be an integer or null"):
            decode_combat_shot_result(encoded)


class TestCombatEngagementCodec:
    """Round-trip encode/decode for CombatEngagementDict."""

    def test_round_trip(self) -> None:
        eng = _make_engagement()
        encoded = encode_combat_engagement(eng)
        decoded = decode_combat_engagement(encoded)
        assert decoded == eng

    def test_multiple_shots(self) -> None:
        shots = [
            _make_shot(shot_number=1, result="hit", distance=1),
            _make_shot(shot_number=2, result="miss", distance=3, weapon_byte=None),
            _make_shot(shot_number=3, result="hit", distance=2),
        ]
        eng = _make_engagement(shots=shots)
        encoded = encode_combat_engagement(eng)
        decoded = decode_combat_engagement(encoded)
        assert decoded == eng
        assert len(decoded["shots"]) == 3

    def test_bool_field_invalid_raises(self) -> None:
        encoded = encode_combat_engagement(_make_engagement())
        encoded["kill_confirmed"] = "yes"
        with pytest.raises(JSONTypeError, match="must be a boolean"):
            decode_combat_engagement(encoded)

    def test_shots_non_object_raises(self) -> None:
        encoded = encode_combat_engagement(_make_engagement())
        encoded["shots"] = ["not_an_object"]
        with pytest.raises(JSONTypeError, match="shots must contain objects"):
            decode_combat_engagement(encoded)


class TestCombatProbeSessionCodec:
    """Round-trip encode/decode for CombatProbeSessionDict."""

    def test_round_trip(self) -> None:
        session = _make_session()
        encoded = encode_combat_probe_session(session)
        decoded = decode_combat_probe_session(encoded)
        assert decoded == session

    def test_empty_engagements(self) -> None:
        session = _make_session(engagements=[])
        encoded = encode_combat_probe_session(session)
        decoded = decode_combat_probe_session(encoded)
        assert decoded["engagements"] == []

    def test_invalid_startup_timing_raises(self) -> None:
        encoded = encode_combat_probe_session(_make_session())
        encoded["startup_timing"] = "not_an_object"
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_combat_probe_session(encoded)

    def test_engagements_non_object_raises(self) -> None:
        encoded = encode_combat_probe_session(_make_session())
        encoded["engagements"] = ["not_an_object"]
        with pytest.raises(JSONTypeError, match="engagements must contain objects"):
            decode_combat_probe_session(encoded)
