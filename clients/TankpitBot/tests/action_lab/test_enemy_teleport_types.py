"""Tests for enemy-directed teleport probe TypedDict codecs."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.action_lab.enemy_teleport_types import (
    EnemyTeleportAttemptResultDict,
    EnemyTeleportProbeSessionDict,
    decode_enemy_teleport_attempt_result,
    decode_enemy_teleport_probe_session,
    encode_enemy_teleport_attempt_result,
    encode_enemy_teleport_probe_session,
)
from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.action_lab.types import TeleportStartupTimingDict, TeleportTargetDict
from tankpit_bot.bot.ai.world_types import (
    EnemyThreatDict,
    make_enemy_threat,
)


def _enemy() -> EnemyThreatDict:
    return make_enemy_threat(
        tank_id=55,
        x=120,
        y=130,
        distance=8,
        damage_state=1,
        rank=2,
        team=1,
        name="enemy",
        is_bot=False,
        timestamp_ms=9000,
    )


def _target() -> TeleportTargetDict:
    return TeleportTargetDict(label="enemy_55_120_130", x=119, y=130)


def _snapshot(timestamp_ms: int) -> PageClientSnapshotDict:
    return PageClientSnapshotDict(
        timestamp_ms=timestamp_ms,
        client_present=True,
        map_visible=False,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=10,
        last_page_client_send_age_ms=20,
        last_bot_send_age_ms=30,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        world_collections={},
        map_fields={},
    )


def _attempt() -> EnemyTeleportAttemptResultDict:
    return EnemyTeleportAttemptResultDict(
        acquisition_strategy="nearest_enemy",
        status="landed_adjacent",
        acquisition_started_ms=1000,
        acquisition_sync_timestamp_ms=1100,
        teleport_started_ms=1200,
        completion_timestamp_ms=1500,
        acquisition_elapsed_ms=100,
        teleport_elapsed_ms=300,
        fuel_before=900,
        fuel_after=820,
        world_timestamp_before=950,
        world_timestamp_after=1450,
        enemy=_enemy(),
        landing_target=_target(),
        landed_signal_received=True,
        landed_x=119,
        landed_y=130,
        enemy_still_visible=True,
        enemy_distance_after=1,
        enemy_x_after=120,
        enemy_y_after=130,
        message_start_index=10,
        message_end_index=16,
        snapshot_before=_snapshot(1000),
        snapshot_after=_snapshot(1500),
    )


def _session() -> EnemyTeleportProbeSessionDict:
    return EnemyTeleportProbeSessionDict(
        session_id="enemy-session",
        start_timestamp_ms=100,
        end_timestamp_ms=2000,
        base_url="https://tankpit.com/play",
        spawn_x=158,
        spawn_y=132,
        acquisition_strategy="nearest_enemy",
        max_attempts=3,
        capture_session_path="enemy_teleport_probe.capture_session.json",
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
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=500,
        heartbeat_interval_ms=0,
        attempts=[_attempt()],
    )


def test_enemy_teleport_attempt_round_trip() -> None:
    encoded = encode_enemy_teleport_attempt_result(_attempt())
    decoded = decode_enemy_teleport_attempt_result(encoded)

    assert decoded == _attempt()


def test_enemy_teleport_attempt_round_trip_with_optional_nulls() -> None:
    attempt = EnemyTeleportAttemptResultDict(
        acquisition_strategy="map_open",
        status="acquisition_timeout",
        acquisition_started_ms=1000,
        acquisition_sync_timestamp_ms=None,
        teleport_started_ms=None,
        completion_timestamp_ms=4000,
        acquisition_elapsed_ms=None,
        teleport_elapsed_ms=None,
        fuel_before=900,
        fuel_after=900,
        world_timestamp_before=950,
        world_timestamp_after=950,
        enemy=None,
        landing_target=None,
        landed_signal_received=False,
        landed_x=158,
        landed_y=132,
        enemy_still_visible=False,
        enemy_distance_after=None,
        enemy_x_after=None,
        enemy_y_after=None,
        message_start_index=1,
        message_end_index=2,
        snapshot_before=_snapshot(1000),
        snapshot_after=_snapshot(4000),
    )

    assert (
        decode_enemy_teleport_attempt_result(encode_enemy_teleport_attempt_result(attempt))
        == attempt
    )


def test_enemy_teleport_attempt_decode_rejects_invalid_strategy() -> None:
    encoded = encode_enemy_teleport_attempt_result(_attempt())
    encoded["acquisition_strategy"] = "bad"

    with pytest.raises(JSONTypeError, match="invalid acquisition strategy"):
        decode_enemy_teleport_attempt_result(encoded)


def test_enemy_teleport_attempt_decode_rejects_invalid_status() -> None:
    encoded = encode_enemy_teleport_attempt_result(_attempt())
    encoded["status"] = "bad"

    with pytest.raises(JSONTypeError, match="invalid enemy teleport status"):
        decode_enemy_teleport_attempt_result(encoded)


def test_enemy_teleport_attempt_decode_rejects_non_boolean_flag() -> None:
    encoded = encode_enemy_teleport_attempt_result(_attempt())
    encoded["enemy_still_visible"] = "bad"

    with pytest.raises(JSONTypeError, match="enemy_still_visible"):
        decode_enemy_teleport_attempt_result(encoded)


def test_enemy_teleport_attempt_decode_rejects_non_object_enemy() -> None:
    encoded = encode_enemy_teleport_attempt_result(_attempt())
    encoded["enemy"] = "bad"

    with pytest.raises(JSONTypeError, match="enemy"):
        decode_enemy_teleport_attempt_result(encoded)


@pytest.mark.parametrize(
    "status",
    [
        "landed_not_adjacent",
        "no_enemy",
        "no_landing_tile",
        "acquisition_timeout",
        "teleport_timeout",
    ],
)
def test_enemy_teleport_attempt_decode_accepts_all_supported_statuses(status: str) -> None:
    encoded = encode_enemy_teleport_attempt_result(_attempt())
    encoded["status"] = status

    assert decode_enemy_teleport_attempt_result(encoded)["status"] == status


def test_enemy_teleport_attempt_decode_rejects_invalid_optional_int() -> None:
    encoded = encode_enemy_teleport_attempt_result(_attempt())
    encoded["teleport_elapsed_ms"] = "bad"

    with pytest.raises(JSONTypeError, match="teleport_elapsed_ms"):
        decode_enemy_teleport_attempt_result(encoded)


def test_enemy_teleport_attempt_decode_rejects_non_object_target() -> None:
    encoded = encode_enemy_teleport_attempt_result(_attempt())
    encoded["landing_target"] = "bad"

    with pytest.raises(JSONTypeError, match="landing_target"):
        decode_enemy_teleport_attempt_result(encoded)


def test_enemy_teleport_attempt_decode_rejects_non_object_snapshot() -> None:
    encoded = encode_enemy_teleport_attempt_result(_attempt())
    encoded["snapshot_before"] = "bad"

    with pytest.raises(JSONTypeError, match="snapshot_before"):
        decode_enemy_teleport_attempt_result(encoded)


def test_enemy_teleport_session_round_trip() -> None:
    encoded = encode_enemy_teleport_probe_session(_session())
    decoded = decode_enemy_teleport_probe_session(encoded)

    assert decoded == _session()


def test_enemy_teleport_session_decode_rejects_non_object_attempt() -> None:
    encoded = encode_enemy_teleport_probe_session(_session())
    encoded["attempts"] = ["bad"]

    with pytest.raises(JSONTypeError, match="attempts"):
        decode_enemy_teleport_probe_session(encoded)


def test_enemy_teleport_session_decode_rejects_non_object_startup_timing() -> None:
    encoded = encode_enemy_teleport_probe_session(_session())
    encoded["startup_timing"] = "bad"

    with pytest.raises(JSONTypeError, match="startup_timing"):
        decode_enemy_teleport_probe_session(encoded)
