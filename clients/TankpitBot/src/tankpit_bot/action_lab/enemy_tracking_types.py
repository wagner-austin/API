"""TypedDict models for the live enemy-tracking probe.

The probe exists to validate (or refute) the wire-presence heuristic
against the JS client's own tank registry. Each ``TrackingObservation``
row is a per-tank, per-sample snapshot pairing what *we* think is
true (our :class:`WorldStateDict` plus the derived ``analyze_threats``
output and the would-be ``get_locked_target`` result) with what the
JS client itself believes (``activeGame.P.j`` from the page-client
snapshot).

The point of cross-referencing both sides is that the user has
reported the bot abandoning targets after one shot. A row where the
JS truth still lists the tank but our wire-presence TTL has just
fired is direct evidence the TTL is wrong; a row where both sides
agree the tank is gone is direct evidence the lock-release decision
was correct.

No back-compat shims; encoders / decoders are strict.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot.action_lab.page_client_snapshot import (
    PageClientSnapshotDict,
    decode_client_field_map,
    decode_page_client_snapshot,
    encode_client_field_map,
    encode_page_client_snapshot,
)
from tankpit_bot.action_lab.types import TeleportStartupTimingDict
from tankpit_bot.action_lab.types_codecs import (
    decode_teleport_startup_timing,
    encode_teleport_startup_timing,
)


class OurTankBeliefDict(TypedDict):
    """Snapshot of our wire-derived belief about one tank at one tick.

    Mirrors :class:`tankpit_bot.state.types.TankStateDict` but flattens
    to the exact fields needed for tracking analysis. Capturing this
    row-by-row gives a frame-by-frame view of how our world state
    evolves -- specifically, the moment wire-presence ages out or the
    registry zeroes a position.

    Attributes:
        tank_id: Tank id in our registry.
        present: Whether ``world.tanks[id]`` had an entry at sample time.
        x: Last-known tile X (0 when ``present`` is False).
        y: Last-known tile Y (0 when ``present`` is False).
        liveness: Liveness label (``"alive"``, ``"deactivated"``, ``""`` when absent).
        last_wire_seen_ms: Timestamp of the tank's most recent wire packet.
        last_position_update_ms: Timestamp of the tank's most recent
            position-bearing wire packet.
        wire_age_ms: ``sample_timestamp_ms - last_wire_seen_ms``.
        position_age_ms: ``sample_timestamp_ms - last_position_update_ms``.
        is_in_threats: Whether the tank was included by
            ``analyze_threats`` for this sample.
        would_locked_target_return: True when, given a hypothetical
            ``combat_target_id == tank_id``, ``get_locked_target`` would
            return a non-None value. This is the field that pinpoints
            whether ``_decide_hunt_engage`` would drop the lock.
        locked_target_source: Where the would-be lock came from
            (``"threats"`` -- in current threat list,
            ``"world_fallback"`` -- synthesised from world.tanks,
            ``"none"`` -- dropped).
    """

    tank_id: int
    present: bool
    x: int
    y: int
    liveness: str
    last_wire_seen_ms: int
    last_position_update_ms: int
    wire_age_ms: int
    position_age_ms: int
    is_in_threats: bool
    would_locked_target_return: bool
    locked_target_source: str


class JSTankBeliefDict(TypedDict):
    """Snapshot of the JS client's own belief about one tank at one tick.

    Captured from ``world_collections`` keyed by ``P.j`` in the
    page-client snapshot (the live JS tank registry per
    :mod:`page_client_snapshot`). When the JS client still lists a
    tank but our wire-presence is dead, the JS side wins -- the JS
    client is the official source of truth for what's visible.

    Attributes:
        present: Whether *any* entry in ``P.j`` matched at sample time.
            Matching uses ``tracked_js_key`` (see
            :class:`TrackedEnemyDict`) for stable identity across ticks
            even if the position changes.
        fields: The matched entry's primitive fields verbatim, minified
            key names from the JS client. Empty when ``present`` is
            False.
    """

    present: bool
    fields: dict[str, int | float | bool | str | None]


class TrackingObservationDict(TypedDict):
    """One per-tank sample row from the tracking probe.

    Pairs our-side and JS-side belief at the same wall-clock instant
    so divergence is unambiguous in the output. ``tracked_label`` is
    the human-readable tank name captured at acquisition time so
    rows from later samples remain readable after the tank has left
    the threat list.

    Attributes:
        sample_index: Zero-based index of the sample within the probe.
        sample_timestamp_ms: Wall-clock time the sample was captured.
        tank_id: Stable tank id from our world state at acquisition.
        tracked_label: Human-readable tank name at acquisition time.
        our_belief: Our wire-derived view of the tank for this sample.
        js_belief: The JS client's view of the same tank for this sample.
        bot_combat_target_id: The bot's ``ai_state.combat_target_id``
            at sample time -- ``-1`` when no lock; ``tank_id`` when
            this tank is the lock; some other id when the bot has
            already moved on.
        bot_mode_state: The bot's ``ai_state.mode_state`` at sample
            time. Reading this tells you whether the bot was in
            ``ENGAGE`` (post-shot, about to drop lock) or some other
            state.
    """

    sample_index: int
    sample_timestamp_ms: int
    tank_id: int
    tracked_label: str
    our_belief: OurTankBeliefDict
    js_belief: JSTankBeliefDict
    bot_combat_target_id: int
    bot_mode_state: str


class TrackedEnemyDict(TypedDict):
    """One enemy the probe locked on to at acquisition time.

    Recording the JS-side key (the minified field name and value used
    to identify the entry in ``P.j``) makes the cross-tick join
    stable even if the position changes between samples -- a position
    match alone would lose identity the moment the enemy moves.

    Attributes:
        tank_id: Stable tank id from our world state.
        name: Human-readable tank name at acquisition.
        team: Team id at acquisition.
        rank: Rank at acquisition.
        acquired_x: Tile X at acquisition.
        acquired_y: Tile Y at acquisition.
        tracked_js_key: The minified field name within a ``P.j`` item
            used to identify the matching JS entry across samples.
            Empty when no JS-side entry could be paired at acquisition
            -- which is itself a data point.
        tracked_js_value: The value of ``tracked_js_key`` in the
            matched ``P.j`` entry. Encoded as a string so the field
            map's mixed primitive value type round-trips cleanly.
    """

    tank_id: int
    name: str
    team: int
    rank: int
    acquired_x: int
    acquired_y: int
    tracked_js_key: str
    tracked_js_value: str


class ShotEventDict(TypedDict):
    """One shot fired by the probe, with the server's response.

    The probe fires ONE shot at the closest enemy after teleporting
    adjacent -- that is the user-reported failure scenario ("fires
    one shot then finds a new one"). Recording the shot's
    boundaries gives the analysis script the line between
    pre-shot and post-shot samples.

    Attributes:
        target_tank_id: Tank id of the shot target.
        target_x: Target tile X at the moment of the shot.
        target_y: Target tile Y at the moment of the shot.
        self_x: Our tank's tile X at the moment of the shot.
        self_y: Our tank's tile Y at the moment of the shot.
        sent_ms: Wall-clock time the shot command was sent.
        responded_ms: Wall-clock time the shot response arrived
            (``-1`` when no response within timeout).
        outcome: ``"hit"``, ``"miss"``, or ``"timeout"``.
    """

    target_tank_id: int
    target_x: int
    target_y: int
    self_x: int
    self_y: int
    sent_ms: int
    responded_ms: int
    outcome: str


class EnemyTrackingProbeSessionDict(TypedDict):
    """Complete tracking-probe session payload.

    The file the analysis tools read to find the divergence row.
    Carries enough metadata to reproduce the run end-to-end:
    bootstrap timing, the enemies under track, the shot fired, and
    the per-tank sample stream.

    Attributes:
        session_id: Stable probe session id.
        start_timestamp_ms: Wall-clock time the probe entered bootstrap.
        end_timestamp_ms: Wall-clock time the probe finished sampling.
        base_url: Probe target URL.
        spawn_x: Tile X of our spawn point.
        spawn_y: Tile Y of our spawn point.
        capture_session_path: On-disk path of the raw wire capture.
        initial_sync_timeout_ms: Bootstrap timeout used.
        startup_timing: Standard startup timing payload.
        acquisition_timeout_ms: Map-open / acquisition timeout used.
        teleport_timeout_ms: Combat teleport timeout used.
        shot_feedback_timeout_ms: Per-shot feedback timeout used.
        sample_interval_ms: Sampling cadence used by the probe.
        sample_duration_ms: Total sampling window after the shot.
        tracked: Enemies the probe locked on to at acquisition.
        shot: The shot the probe fired (or ``None`` when acquisition
            never reached the engage tile).
        snapshot_at_acquisition: Page-client snapshot captured the
            tick the shot fired (or just after acquisition when no
            shot fired).
        observations: All per-tank, per-sample rows produced by the
            sampling loop. Ordered by ``sample_index`` then ``tank_id``.
    """

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    acquisition_timeout_ms: int
    teleport_timeout_ms: int
    shot_feedback_timeout_ms: int
    sample_interval_ms: int
    sample_duration_ms: int
    tracked: list[TrackedEnemyDict]
    shot: ShotEventDict | None
    snapshot_at_acquisition: PageClientSnapshotDict
    observations: list[TrackingObservationDict]


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


def _require_bool_field(data: JSONObject, field: str) -> bool:
    """Return a required boolean field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Boolean value.

    Raises:
        JSONTypeError: If the field is missing or not a boolean.
    """
    raw = data.get(field)
    if not isinstance(raw, bool):
        raise JSONTypeError(f"Field '{field}' must be a boolean")
    return raw


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
        present=_require_bool_field(data, "present"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        liveness=require_str(data, "liveness"),
        last_wire_seen_ms=require_int(data, "last_wire_seen_ms"),
        last_position_update_ms=require_int(data, "last_position_update_ms"),
        wire_age_ms=require_int(data, "wire_age_ms"),
        position_age_ms=require_int(data, "position_age_ms"),
        is_in_threats=_require_bool_field(data, "is_in_threats"),
        would_locked_target_return=_require_bool_field(data, "would_locked_target_return"),
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
        present=_require_bool_field(data, "present"),
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
    "EnemyTrackingProbeSessionDict",
    "JSTankBeliefDict",
    "OurTankBeliefDict",
    "ShotEventDict",
    "TrackedEnemyDict",
    "TrackingObservationDict",
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
