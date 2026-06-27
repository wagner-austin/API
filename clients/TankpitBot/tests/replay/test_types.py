"""Tests for replay TypedDicts encode/decode round-trip and validation."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.bot.ai.types import EnemyThreatDict, make_enemy_threat
from tankpit_bot.replay.types import (
    ReplaySessionResultDict,
    ReplayTickTraceDict,
    decode_replay_session_result,
    decode_replay_tick_trace,
    encode_replay_session_result,
    encode_replay_tick_trace,
)


def _sample_threat() -> EnemyThreatDict:
    """Return a sample enemy threat for testing."""
    return make_enemy_threat(
        tank_id=42,
        x=110,
        y=125,
        distance=15,
        damage_state=1,
        rank=4,
        team=1,
        name="Artax",
        is_bot=False,
        timestamp_ms=900,
    )


def _sample_trace() -> ReplayTickTraceDict:
    """Return a sample tick trace for testing."""
    return ReplayTickTraceDict(
        tick_index=0,
        timestamp_ms=1000,
        self_x=100,
        self_y=120,
        fuel=500,
        behavior_mode="HUNT",
        behavior_score=900,
        behavior_reason="find_enemies",
        ai_mode="HUNT",
        ai_mode_state="ACQUIRE",
        command_type="map_open",
        target_x=0,
        target_y=0,
        combat_target_id=-1,
        resource_target_kind="",
        visible_threats=[_sample_threat()],
        container_count=5,
    )


def _sample_result() -> ReplaySessionResultDict:
    """Return a sample session result for testing."""
    return ReplaySessionResultDict(
        session_id="test-session-001",
        total_ticks=1,
        total_messages=10,
        traces=[_sample_trace()],
    )


class TestReplayTickTrace:
    """Tests for ReplayTickTraceDict encode/decode."""

    def test_encode_returns_json_object(self) -> None:
        """encode_replay_tick_trace returns a dict with all fields."""
        trace = _sample_trace()
        encoded = encode_replay_tick_trace(trace)
        assert encoded["tick_index"] == 0
        assert encoded["timestamp_ms"] == 1000
        assert encoded["self_x"] == 100
        assert encoded["self_y"] == 120
        assert encoded["fuel"] == 500
        assert encoded["behavior_mode"] == "HUNT"
        assert encoded["behavior_score"] == 900
        assert encoded["behavior_reason"] == "find_enemies"
        assert encoded["ai_mode"] == "HUNT"
        assert encoded["ai_mode_state"] == "ACQUIRE"
        assert encoded["command_type"] == "map_open"
        assert encoded["target_x"] == 0
        assert encoded["target_y"] == 0
        assert encoded["combat_target_id"] == -1
        assert encoded["resource_target_kind"] == ""
        assert encoded["container_count"] == 5

    def test_round_trip(self) -> None:
        """encode then decode produces identical trace."""
        original = _sample_trace()
        encoded = encode_replay_tick_trace(original)
        decoded = decode_replay_tick_trace(encoded)
        assert decoded == original

    def test_decode_missing_field_raises(self) -> None:
        """decode_replay_tick_trace raises on missing required field."""
        encoded = encode_replay_tick_trace(_sample_trace())
        del encoded["fuel"]
        with pytest.raises(JSONTypeError):
            decode_replay_tick_trace(encoded)

    def test_round_trip_with_nonzero_targets(self) -> None:
        """Round-trip preserves non-zero target coordinates."""
        trace = ReplayTickTraceDict(
            tick_index=3,
            timestamp_ms=5000,
            self_x=50,
            self_y=60,
            fuel=800,
            behavior_mode="COLLECT",
            behavior_score=900,
            behavior_reason="fuel=700",
            ai_mode="COLLECT",
            ai_mode_state="PICKUP",
            command_type="pickup_fuel",
            target_x=52,
            target_y=63,
            combat_target_id=-1,
            resource_target_kind="fuel",
            visible_threats=[],
            container_count=3,
        )
        decoded = decode_replay_tick_trace(encode_replay_tick_trace(trace))
        assert decoded == trace

    def test_round_trip_with_combat_state(self) -> None:
        """Round-trip preserves active combat state fields."""
        trace = ReplayTickTraceDict(
            tick_index=7,
            timestamp_ms=8000,
            self_x=100,
            self_y=100,
            fuel=150,
            behavior_mode="HUNT",
            behavior_score=950,
            behavior_reason="combat_shoot",
            ai_mode="HUNT",
            ai_mode_state="ENGAGE",
            command_type="shoot",
            target_x=101,
            target_y=100,
            combat_target_id=42,
            resource_target_kind="",
            visible_threats=[_sample_threat()],
            container_count=0,
        )
        decoded = decode_replay_tick_trace(encode_replay_tick_trace(trace))
        assert decoded == trace
        assert decoded["ai_mode"] == "HUNT"
        assert decoded["ai_mode_state"] == "ENGAGE"
        assert decoded["combat_target_id"] == 42
        assert len(decoded["visible_threats"]) == 1
        assert decoded["visible_threats"][0]["name"] == "Artax"

    def test_decode_threats_not_list_raises(self) -> None:
        """decode_replay_tick_trace raises when visible_threats is not a list."""
        encoded = encode_replay_tick_trace(_sample_trace())
        encoded["visible_threats"] = "not a list"
        with pytest.raises(ValueError, match="visible_threats must be a list"):
            decode_replay_tick_trace(encoded)

    def test_decode_threat_not_dict_raises(self) -> None:
        """decode_replay_tick_trace raises when a threat element is not a dict."""
        encoded = encode_replay_tick_trace(_sample_trace())
        encoded["visible_threats"] = ["not a dict"]
        with pytest.raises(ValueError, match=r"visible_threats\[0\] must be an object"):
            decode_replay_tick_trace(encoded)

    def test_decode_invalid_ai_mode_pair_raises(self) -> None:
        """decode_replay_tick_trace rejects invalid durable mode/state pairs."""
        encoded = encode_replay_tick_trace(_sample_trace())
        encoded["ai_mode"] = "UNSET"
        encoded["ai_mode_state"] = "SEARCH"
        with pytest.raises(ValueError, match="invalid for ai_mode"):
            decode_replay_tick_trace(encoded)


class TestReplaySessionResult:
    """Tests for ReplaySessionResultDict encode/decode."""

    def test_encode_returns_json_object(self) -> None:
        """encode_replay_session_result returns a dict with all fields."""
        result = _sample_result()
        encoded = encode_replay_session_result(result)
        assert encoded["session_id"] == "test-session-001"
        assert encoded["total_ticks"] == 1
        assert encoded["total_messages"] == 10
        decoded = decode_replay_session_result(encoded)
        assert len(decoded["traces"]) == 1

    def test_round_trip(self) -> None:
        """encode then decode produces identical result."""
        original = _sample_result()
        encoded = encode_replay_session_result(original)
        decoded = decode_replay_session_result(encoded)
        assert decoded == original

    def test_round_trip_empty_traces(self) -> None:
        """Round-trip works with zero traces."""
        result = ReplaySessionResultDict(
            session_id="empty-session",
            total_ticks=0,
            total_messages=5,
            traces=[],
        )
        decoded = decode_replay_session_result(encode_replay_session_result(result))
        assert decoded == result

    def test_round_trip_multiple_traces(self) -> None:
        """Round-trip preserves multiple traces in order."""
        traces = [
            ReplayTickTraceDict(
                tick_index=i,
                timestamp_ms=1000 + i * 100,
                self_x=100 + i,
                self_y=120 + i,
                fuel=500 - i * 10,
                behavior_mode="HUNT",
                behavior_score=900,
                behavior_reason="find_enemies",
                ai_mode="HUNT",
                ai_mode_state="REFRESH",
                command_type="map_open",
                target_x=0,
                target_y=0,
                combat_target_id=-1,
                resource_target_kind="",
                visible_threats=[],
                container_count=5 - i,
            )
            for i in range(5)
        ]
        result = ReplaySessionResultDict(
            session_id="multi-trace",
            total_ticks=5,
            total_messages=50,
            traces=traces,
        )
        decoded = decode_replay_session_result(encode_replay_session_result(result))
        assert decoded == result

    def test_decode_missing_field_raises(self) -> None:
        """decode_replay_session_result raises on missing required field."""
        encoded = encode_replay_session_result(_sample_result())
        del encoded["session_id"]
        with pytest.raises(JSONTypeError):
            decode_replay_session_result(encoded)

    def test_decode_traces_not_list_raises(self) -> None:
        """decode_replay_session_result raises when traces is not a list."""
        encoded = encode_replay_session_result(_sample_result())
        encoded["traces"] = "not a list"
        with pytest.raises(ValueError, match="traces must be a list"):
            decode_replay_session_result(encoded)

    def test_decode_trace_not_dict_raises(self) -> None:
        """decode_replay_session_result raises when a trace element is not a dict."""
        encoded = encode_replay_session_result(_sample_result())
        encoded["traces"] = ["not a dict"]
        with pytest.raises(ValueError, match=r"traces\[0\] must be an object"):
            decode_replay_session_result(encoded)
