"""Encode/decode pairs for the enemy-tracking probe records.

Every record in :mod:`tankpit_bot.action_lab.enemy_tracking_types` has
its pair here, plus the JSON-narrowing helpers the decoders share.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_bool,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot.action_lab.enemy_tracking_types import (
    EnemyTrackingProbeSessionDict,
    JSTankBeliefDict,
    OurTankBeliefDict,
    ShotEventDict,
    TrackedEnemyDict,
    TrackingObservationDict,
)
from tankpit_bot.action_lab.types_codecs import (
    decode_teleport_startup_timing,
    encode_teleport_startup_timing,
)
from tankpit_bot.browser.page_client_snapshot import (
    decode_page_client_snapshot,
    encode_page_client_snapshot,
)
from tankpit_bot.browser.page_client_snapshot_codecs import (
    decode_client_field_map,
    encode_client_field_map,
)


def encode_our_tank_belief(belief: OurTankBeliefDict) -> JSONObject:
    """Encode an :class:`OurTankBeliefDict` to a JSON object.

    Args:
        belief: Belief row to encode.

    Returns:
        JSON-serializable object preserving field order.
    """
    return {
        "tank_id": belief["tank_id"],
        "present": belief["present"],
        "x": belief["x"],
        "y": belief["y"],
        "liveness": belief["liveness"],
        "last_wire_seen_ms": belief["last_wire_seen_ms"],
        "last_position_update_ms": belief["last_position_update_ms"],
        "wire_age_ms": belief["wire_age_ms"],
        "position_age_ms": belief["position_age_ms"],
        "is_in_threats": belief["is_in_threats"],
        "would_locked_target_return": belief["would_locked_target_return"],
        "locked_target_source": belief["locked_target_source"],
    }


def encode_js_tank_belief(belief: JSTankBeliefDict) -> JSONObject:
    """Encode a :class:`JSTankBeliefDict` to a JSON object.

    Args:
        belief: Belief row to encode.

    Returns:
        JSON-serializable object preserving field order.
    """
    return {
        "present": belief["present"],
        "fields": encode_client_field_map(belief["fields"]),
    }


def encode_tracking_observation(observation: TrackingObservationDict) -> JSONObject:
    """Encode a :class:`TrackingObservationDict` to a JSON object.

    Args:
        observation: Observation row to encode.

    Returns:
        JSON-serializable object preserving field order.
    """
    return {
        "sample_index": observation["sample_index"],
        "sample_timestamp_ms": observation["sample_timestamp_ms"],
        "tank_id": observation["tank_id"],
        "tracked_label": observation["tracked_label"],
        "our_belief": encode_our_tank_belief(observation["our_belief"]),
        "js_belief": encode_js_tank_belief(observation["js_belief"]),
        "bot_combat_target_id": observation["bot_combat_target_id"],
        "bot_mode_state": observation["bot_mode_state"],
    }


def encode_tracked_enemy(tracked: TrackedEnemyDict) -> JSONObject:
    """Encode a :class:`TrackedEnemyDict` to a JSON object.

    Args:
        tracked: Tracked-enemy record to encode.

    Returns:
        JSON-serializable object preserving field order.
    """
    return {
        "tank_id": tracked["tank_id"],
        "name": tracked["name"],
        "team": tracked["team"],
        "rank": tracked["rank"],
        "acquired_x": tracked["acquired_x"],
        "acquired_y": tracked["acquired_y"],
        "tracked_js_key": tracked["tracked_js_key"],
        "tracked_js_value": tracked["tracked_js_value"],
    }


def encode_shot_event(shot: ShotEventDict) -> JSONObject:
    """Encode a :class:`ShotEventDict` to a JSON object.

    Args:
        shot: Shot record to encode.

    Returns:
        JSON-serializable object preserving field order.
    """
    return {
        "target_tank_id": shot["target_tank_id"],
        "target_x": shot["target_x"],
        "target_y": shot["target_y"],
        "self_x": shot["self_x"],
        "self_y": shot["self_y"],
        "sent_ms": shot["sent_ms"],
        "responded_ms": shot["responded_ms"],
        "outcome": shot["outcome"],
    }


def encode_enemy_tracking_probe_session(
    session: EnemyTrackingProbeSessionDict,
) -> JSONObject:
    """Encode a :class:`EnemyTrackingProbeSessionDict` to a JSON object.

    Args:
        session: Session payload to encode.

    Returns:
        JSON-serializable object preserving field order.
    """
    encoded_tracked: list[JSONValue] = [encode_tracked_enemy(t) for t in session["tracked"]]
    encoded_observations: list[JSONValue] = [
        encode_tracking_observation(o) for o in session["observations"]
    ]
    shot_value: JSONValue = None if session["shot"] is None else encode_shot_event(session["shot"])
    return {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "spawn_x": session["spawn_x"],
        "spawn_y": session["spawn_y"],
        "capture_session_path": session["capture_session_path"],
        "initial_sync_timeout_ms": session["initial_sync_timeout_ms"],
        "startup_timing": encode_teleport_startup_timing(session["startup_timing"]),
        "acquisition_timeout_ms": session["acquisition_timeout_ms"],
        "teleport_timeout_ms": session["teleport_timeout_ms"],
        "shot_feedback_timeout_ms": session["shot_feedback_timeout_ms"],
        "sample_interval_ms": session["sample_interval_ms"],
        "sample_duration_ms": session["sample_duration_ms"],
        "tracked": encoded_tracked,
        "shot": shot_value,
        "snapshot_at_acquisition": encode_page_client_snapshot(session["snapshot_at_acquisition"]),
        "observations": encoded_observations,
    }


def _require_fields_map(
    data: JSONObject,
    field: str,
) -> dict[str, int | float | bool | str | None]:
    """Return a required JS field map from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Validated field map.

    Raises:
        JSONTypeError: If the field is missing or not an object.
    """
    raw = data.get(field)
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{field}' must be an object")
    return decode_client_field_map(raw, field=field)


def decode_our_tank_belief(data: JSONObject) -> OurTankBeliefDict:
    """Decode an :class:`OurTankBeliefDict` from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated belief row.

    Raises:
        JSONTypeError: If any required field is missing or invalid.
    """
    return OurTankBeliefDict(
        tank_id=require_int(data, "tank_id"),
        present=require_bool(data, "present"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        liveness=require_str(data, "liveness"),
        last_wire_seen_ms=require_int(data, "last_wire_seen_ms"),
        last_position_update_ms=require_int(data, "last_position_update_ms"),
        wire_age_ms=require_int(data, "wire_age_ms"),
        position_age_ms=require_int(data, "position_age_ms"),
        is_in_threats=require_bool(data, "is_in_threats"),
        would_locked_target_return=require_bool(data, "would_locked_target_return"),
        locked_target_source=require_str(data, "locked_target_source"),
    )


def decode_js_tank_belief(data: JSONObject) -> JSTankBeliefDict:
    """Decode a :class:`JSTankBeliefDict` from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated belief row.

    Raises:
        JSONTypeError: If any required field is missing or invalid.
    """
    return JSTankBeliefDict(
        present=require_bool(data, "present"),
        fields=_require_fields_map(data, "fields"),
    )


def _require_observation_object(data: JSONObject, field: str) -> JSONObject:
    """Return an inner JSON object field from a decoded row.

    Args:
        data: JSON object being decoded.
        field: Field name to read.

    Returns:
        Validated inner object.

    Raises:
        JSONTypeError: If the field is missing or not a JSON object.
    """
    raw = data.get(field)
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{field}' must be an object")
    return raw


def decode_tracking_observation(data: JSONObject) -> TrackingObservationDict:
    """Decode a :class:`TrackingObservationDict` from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated observation row.

    Raises:
        JSONTypeError: If any required field is missing or invalid.
    """
    return TrackingObservationDict(
        sample_index=require_int(data, "sample_index"),
        sample_timestamp_ms=require_int(data, "sample_timestamp_ms"),
        tank_id=require_int(data, "tank_id"),
        tracked_label=require_str(data, "tracked_label"),
        our_belief=decode_our_tank_belief(_require_observation_object(data, "our_belief")),
        js_belief=decode_js_tank_belief(_require_observation_object(data, "js_belief")),
        bot_combat_target_id=require_int(data, "bot_combat_target_id"),
        bot_mode_state=require_str(data, "bot_mode_state"),
    )


def decode_tracked_enemy(data: JSONObject) -> TrackedEnemyDict:
    """Decode a :class:`TrackedEnemyDict` from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated tracked-enemy record.

    Raises:
        JSONTypeError: If any required field is missing or invalid.
    """
    return TrackedEnemyDict(
        tank_id=require_int(data, "tank_id"),
        name=require_str(data, "name"),
        team=require_int(data, "team"),
        rank=require_int(data, "rank"),
        acquired_x=require_int(data, "acquired_x"),
        acquired_y=require_int(data, "acquired_y"),
        tracked_js_key=require_str(data, "tracked_js_key"),
        tracked_js_value=require_str(data, "tracked_js_value"),
    )


def decode_shot_event(data: JSONObject) -> ShotEventDict:
    """Decode a :class:`ShotEventDict` from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated shot record.

    Raises:
        JSONTypeError: If any required field is missing or invalid.
    """
    return ShotEventDict(
        target_tank_id=require_int(data, "target_tank_id"),
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        self_x=require_int(data, "self_x"),
        self_y=require_int(data, "self_y"),
        sent_ms=require_int(data, "sent_ms"),
        responded_ms=require_int(data, "responded_ms"),
        outcome=require_str(data, "outcome"),
    )


def _decode_optional_shot(data: JSONObject, field: str) -> ShotEventDict | None:
    """Decode an optional :class:`ShotEventDict` field.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Shot event or None.

    Raises:
        JSONTypeError: If the field is present but not an object.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{field}' must be an object or null")
    return decode_shot_event(raw)


def _decode_tracked_list(raw: JSONValue) -> list[TrackedEnemyDict]:
    """Decode the tracked-enemies list from raw JSON.

    Args:
        raw: Raw JSON list value.

    Returns:
        Decoded tracked-enemy records.

    Raises:
        JSONTypeError: If any element is not a JSON object.
    """
    result: list[TrackedEnemyDict] = []
    for item in require_list({"items": raw}, "items"):
        if not isinstance(item, dict):
            raise JSONTypeError("Field 'tracked' must contain only objects")
        result.append(decode_tracked_enemy(item))
    return result


def _decode_observation_list(raw: JSONValue) -> list[TrackingObservationDict]:
    """Decode the observations list from raw JSON.

    Args:
        raw: Raw JSON list value.

    Returns:
        Decoded observation rows.

    Raises:
        JSONTypeError: If any element is not a JSON object.
    """
    result: list[TrackingObservationDict] = []
    for item in require_list({"items": raw}, "items"):
        if not isinstance(item, dict):
            raise JSONTypeError("Field 'observations' must contain only objects")
        result.append(decode_tracking_observation(item))
    return result


def _require_inner_object(data: JSONObject, field: str) -> JSONObject:
    """Return an inner JSON object field from a session payload.

    Args:
        data: JSON object being decoded.
        field: Field name to read.

    Returns:
        Validated inner object.

    Raises:
        JSONTypeError: If the field is missing or not a JSON object.
    """
    raw = data.get(field)
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{field}' must be an object")
    return raw


def decode_enemy_tracking_probe_session(
    data: JSONObject,
) -> EnemyTrackingProbeSessionDict:
    """Decode a :class:`EnemyTrackingProbeSessionDict` from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated tracking-probe session payload.

    Raises:
        JSONTypeError: If any required field is missing or invalid.
    """
    return EnemyTrackingProbeSessionDict(
        session_id=require_str(data, "session_id"),
        start_timestamp_ms=require_int(data, "start_timestamp_ms"),
        end_timestamp_ms=require_int(data, "end_timestamp_ms"),
        base_url=require_str(data, "base_url"),
        spawn_x=require_int(data, "spawn_x"),
        spawn_y=require_int(data, "spawn_y"),
        capture_session_path=require_str(data, "capture_session_path"),
        initial_sync_timeout_ms=require_int(data, "initial_sync_timeout_ms"),
        startup_timing=decode_teleport_startup_timing(
            _require_inner_object(data, "startup_timing"),
        ),
        acquisition_timeout_ms=require_int(data, "acquisition_timeout_ms"),
        teleport_timeout_ms=require_int(data, "teleport_timeout_ms"),
        shot_feedback_timeout_ms=require_int(data, "shot_feedback_timeout_ms"),
        sample_interval_ms=require_int(data, "sample_interval_ms"),
        sample_duration_ms=require_int(data, "sample_duration_ms"),
        tracked=_decode_tracked_list(data.get("tracked")),
        shot=_decode_optional_shot(data, "shot"),
        snapshot_at_acquisition=decode_page_client_snapshot(
            _require_inner_object(data, "snapshot_at_acquisition"),
        ),
        observations=_decode_observation_list(data.get("observations")),
    )


__all__ = [
    "decode_enemy_tracking_probe_session",
    "decode_js_tank_belief",
    "decode_our_tank_belief",
    "decode_shot_event",
    "decode_tracked_enemy",
    "decode_tracking_observation",
    "encode_enemy_tracking_probe_session",
    "encode_js_tank_belief",
    "encode_our_tank_belief",
    "encode_shot_event",
    "encode_tracked_enemy",
    "encode_tracking_observation",
]
