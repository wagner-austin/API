"""Round-trip + validation tests for enemy tracking probe TypedDicts."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.action_lab.enemy_tracking_codecs import (
    decode_enemy_tracking_probe_session,
    decode_js_tank_belief,
    decode_our_tank_belief,
    decode_shot_event,
    decode_tracked_enemy,
    decode_tracking_observation,
    encode_enemy_tracking_probe_session,
    encode_js_tank_belief,
    encode_our_tank_belief,
    encode_shot_event,
    encode_tracked_enemy,
    encode_tracking_observation,
)
from tankpit_bot.action_lab.enemy_tracking_types import (
    EnemyTrackingProbeSessionDict,
    JSTankBeliefDict,
    OurTankBeliefDict,
    ShotEventDict,
    TrackedEnemyDict,
    TrackingObservationDict,
)
from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.action_lab.types import TeleportStartupTimingDict


def _make_our_belief() -> OurTankBeliefDict:
    return OurTankBeliefDict(
        tank_id=511,
        present=True,
        x=99,
        y=100,
        liveness="alive",
        last_wire_seen_ms=10_000,
        last_position_update_ms=9_500,
        wire_age_ms=500,
        position_age_ms=1_000,
        is_in_threats=True,
        would_locked_target_return=True,
        locked_target_source="threats",
    )


def _make_js_belief(*, present: bool = True) -> JSTankBeliefDict:
    fields: dict[str, int | float | bool | str | None] = (
        {"a": 511, "b": 99, "c": 100} if present else {}
    )
    return JSTankBeliefDict(present=present, fields=fields)


def _make_observation() -> TrackingObservationDict:
    return TrackingObservationDict(
        sample_index=3,
        sample_timestamp_ms=12_345,
        tank_id=511,
        tracked_label="orange-7",
        our_belief=_make_our_belief(),
        js_belief=_make_js_belief(),
        bot_combat_target_id=511,
        bot_mode_state="ENGAGE",
    )


def _make_tracked() -> TrackedEnemyDict:
    return TrackedEnemyDict(
        tank_id=511,
        name="orange-7",
        team=3,
        rank=4,
        acquired_x=99,
        acquired_y=100,
        tracked_js_key="a",
        tracked_js_value="511",
    )


def _make_shot() -> ShotEventDict:
    return ShotEventDict(
        target_tank_id=511,
        target_x=99,
        target_y=100,
        self_x=100,
        self_y=100,
        sent_ms=10_500,
        responded_ms=10_600,
        outcome="hit",
    )


def _make_snapshot() -> PageClientSnapshotDict:
    return PageClientSnapshotDict(
        timestamp_ms=11_000,
        client_present=True,
        map_visible=True,
        client_state=2,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=80,
        last_page_client_send_age_ms=60,
        last_bot_send_age_ms=40,
        ws_ready_state=1,
        current_send_label="shoot",
        sent_frame_meta_queue_length=0,
        self_fields={"x": 100, "y": 100},
        world_fields={"timestamp": 11_000},
        map_fields={},
        world_collections={"P.j": [{"a": 511, "b": 99, "c": 100}]},
    )


def _make_startup_timing() -> TeleportStartupTimingDict:
    return TeleportStartupTimingDict(
        game_ready_timestamp_ms=1_000,
        intel_ready_timestamp_ms=1_100,
        initial_sync_started_ms=1_200,
        initial_world_timestamp_ms=1_300,
        command_ready_timestamp_ms=1_400,
        first_attempt_started_ms=1_500,
        game_ready_to_intel_ready_ms=100,
        intel_ready_to_initial_world_ms=200,
        initial_world_to_command_ready_ms=100,
        command_ready_to_first_attempt_ms=100,
    )


def _make_session(*, with_shot: bool = True) -> EnemyTrackingProbeSessionDict:
    return EnemyTrackingProbeSessionDict(
        session_id="session-abc",
        start_timestamp_ms=1_000,
        end_timestamp_ms=2_000,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="runs/track/cap.json",
        initial_sync_timeout_ms=10_000,
        startup_timing=_make_startup_timing(),
        acquisition_timeout_ms=5_000,
        teleport_timeout_ms=10_000,
        shot_feedback_timeout_ms=4_000,
        sample_interval_ms=1_000,
        sample_duration_ms=120_000,
        tracked=[_make_tracked()],
        shot=_make_shot() if with_shot else None,
        snapshot_at_acquisition=_make_snapshot(),
        observations=[_make_observation()],
    )


def test_our_tank_belief_roundtrip_preserves_fields() -> None:
    """OurTankBeliefDict encode/decode round-trip preserves every field."""
    belief = _make_our_belief()
    decoded = decode_our_tank_belief(encode_our_tank_belief(belief))
    assert decoded == belief


def test_js_tank_belief_roundtrip_preserves_fields() -> None:
    """JSTankBeliefDict encode/decode round-trip preserves presence + fields."""
    belief = _make_js_belief()
    decoded = decode_js_tank_belief(encode_js_tank_belief(belief))
    assert decoded == belief


def test_js_tank_belief_roundtrip_handles_absent() -> None:
    """JSTankBeliefDict with present=False round-trips with empty fields."""
    belief = _make_js_belief(present=False)
    decoded = decode_js_tank_belief(encode_js_tank_belief(belief))
    assert decoded == belief


def test_tracking_observation_roundtrip_preserves_fields() -> None:
    """TrackingObservationDict encode/decode round-trip preserves every field."""
    observation = _make_observation()
    decoded = decode_tracking_observation(encode_tracking_observation(observation))
    assert decoded == observation


def test_tracked_enemy_roundtrip_preserves_fields() -> None:
    """TrackedEnemyDict encode/decode round-trip preserves every field."""
    tracked = _make_tracked()
    decoded = decode_tracked_enemy(encode_tracked_enemy(tracked))
    assert decoded == tracked


def test_shot_event_roundtrip_preserves_fields() -> None:
    """ShotEventDict encode/decode round-trip preserves every field."""
    shot = _make_shot()
    decoded = decode_shot_event(encode_shot_event(shot))
    assert decoded == shot


def test_session_roundtrip_with_shot_preserves_fields() -> None:
    """Session round-trip preserves every field when a shot fired."""
    session = _make_session(with_shot=True)
    decoded = decode_enemy_tracking_probe_session(
        encode_enemy_tracking_probe_session(session),
    )
    assert decoded == session


def test_session_roundtrip_without_shot_preserves_null() -> None:
    """Session round-trip preserves None when no shot fired."""
    session = _make_session(with_shot=False)
    decoded = decode_enemy_tracking_probe_session(
        encode_enemy_tracking_probe_session(session),
    )
    assert decoded["shot"] is None


def test_our_belief_decode_rejects_non_bool_present() -> None:
    """decode_our_tank_belief rejects a non-boolean ``present``."""
    encoded = encode_our_tank_belief(_make_our_belief())
    encoded["present"] = "yes"
    with pytest.raises(JSONTypeError):
        decode_our_tank_belief(encoded)


def test_js_belief_decode_rejects_non_object_fields() -> None:
    """decode_js_tank_belief rejects a non-object ``fields``."""
    encoded = encode_js_tank_belief(_make_js_belief())
    encoded["fields"] = []
    with pytest.raises(JSONTypeError):
        decode_js_tank_belief(encoded)


def test_observation_decode_rejects_non_object_our_belief() -> None:
    """decode_tracking_observation rejects a non-object nested belief."""
    encoded = encode_tracking_observation(_make_observation())
    encoded["our_belief"] = "oops"
    with pytest.raises(JSONTypeError):
        decode_tracking_observation(encoded)


def test_session_decode_rejects_non_object_shot() -> None:
    """decode_enemy_tracking_probe_session rejects a non-object shot field."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["shot"] = 42
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)


def test_session_decode_rejects_non_list_tracked() -> None:
    """decode_enemy_tracking_probe_session rejects a non-list tracked field."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["tracked"] = {}
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)


def test_session_decode_rejects_non_object_tracked_item() -> None:
    """decode_enemy_tracking_probe_session rejects non-object items in tracked."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["tracked"] = [42]
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)


def test_session_decode_rejects_non_list_observations() -> None:
    """decode_enemy_tracking_probe_session rejects a non-list observations field."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["observations"] = 42
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)


def test_session_decode_rejects_non_object_observation_item() -> None:
    """decode_enemy_tracking_probe_session rejects non-object items in observations."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["observations"] = [42]
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)


def test_session_decode_rejects_non_object_startup_timing() -> None:
    """decode_enemy_tracking_probe_session rejects a non-object startup_timing."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["startup_timing"] = "bad"
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)


def test_session_decode_rejects_non_object_snapshot() -> None:
    """decode_enemy_tracking_probe_session rejects a non-object snapshot field."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["snapshot_at_acquisition"] = []
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)
