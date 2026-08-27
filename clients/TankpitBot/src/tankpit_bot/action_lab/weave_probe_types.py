"""TypedDict models for the shoot+move weave probe.

The weave probe answers one question the dodge doctrine hangs on:
does a queued shot still serve on a beat where the tank also moved?
Beats alternate shoot-only and shoot+move; served shots are counted
from server-refreshed 0x49 ammo snapshots exactly as the cadence
probe counts them.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_bool,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot.action_lab.types import TeleportStartupTimingDict
from tankpit_bot.action_lab.types_codecs import (
    decode_teleport_startup_timing,
    encode_teleport_startup_timing,
)


class WeaveBeatDict(TypedDict):
    """One beat of the weave pattern.

    ``moved`` is the experimental variable: on move beats a 1-tile
    walk was dispatched ~200 ms after the shot, inside the same serve
    window. ``move_x``/``move_y`` are ``-1`` on shoot-only beats.
    """

    beat_number: int
    dispatched_ms: int
    target_x: int
    target_y: int
    moved: bool
    move_x: int
    move_y: int


class WeaveBurstDict(TypedDict):
    """One alternating shoot-only / shoot+move burst at one target."""

    target_id: int
    target_name: str
    beats: list[WeaveBeatDict]
    shots_dispatched: int
    moves_dispatched: int
    dual_before: int
    dual_after: int
    homing_before: int
    homing_after: int
    fuel_before: int
    fuel_after: int
    served_hits: int
    target_killed: bool


class WeaveProbeSessionDict(TypedDict):
    """Complete weave probe session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    beats_per_burst: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    bursts: list[WeaveBurstDict]


def encode_weave_beat(beat: WeaveBeatDict) -> JSONObject:
    """Encode one weave beat to a JSON object.

    Args:
        beat: Beat to encode.

    Returns:
        JSON-serializable object representation.
    """
    return {
        "beat_number": beat["beat_number"],
        "dispatched_ms": beat["dispatched_ms"],
        "target_x": beat["target_x"],
        "target_y": beat["target_y"],
        "moved": beat["moved"],
        "move_x": beat["move_x"],
        "move_y": beat["move_y"],
    }


def encode_weave_burst(burst: WeaveBurstDict) -> JSONObject:
    """Encode one weave burst to a JSON object.

    Args:
        burst: Burst to encode.

    Returns:
        JSON-serializable object representation.
    """
    beats: list[JSONValue] = [encode_weave_beat(b) for b in burst["beats"]]
    return {
        "target_id": burst["target_id"],
        "target_name": burst["target_name"],
        "beats": beats,
        "shots_dispatched": burst["shots_dispatched"],
        "moves_dispatched": burst["moves_dispatched"],
        "dual_before": burst["dual_before"],
        "dual_after": burst["dual_after"],
        "homing_before": burst["homing_before"],
        "homing_after": burst["homing_after"],
        "fuel_before": burst["fuel_before"],
        "fuel_after": burst["fuel_after"],
        "served_hits": burst["served_hits"],
        "target_killed": burst["target_killed"],
    }


def encode_weave_probe_session(session: WeaveProbeSessionDict) -> JSONObject:
    """Encode a weave probe session to a JSON object.

    Args:
        session: Session to encode.

    Returns:
        JSON-serializable object representation.
    """
    bursts: list[JSONValue] = [encode_weave_burst(b) for b in session["bursts"]]
    return {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "spawn_x": session["spawn_x"],
        "spawn_y": session["spawn_y"],
        "beats_per_burst": session["beats_per_burst"],
        "capture_session_path": session["capture_session_path"],
        "initial_sync_timeout_ms": session["initial_sync_timeout_ms"],
        "startup_timing": encode_teleport_startup_timing(session["startup_timing"]),
        "bursts": bursts,
    }


def decode_weave_beat(data: JSONObject) -> WeaveBeatDict:
    """Decode one weave beat from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated beat.
    """
    return WeaveBeatDict(
        beat_number=require_int(data, "beat_number"),
        dispatched_ms=require_int(data, "dispatched_ms"),
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        moved=require_bool(data, "moved"),
        move_x=require_int(data, "move_x"),
        move_y=require_int(data, "move_y"),
    )


def decode_weave_burst(data: JSONObject) -> WeaveBurstDict:
    """Decode one weave burst from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated burst.
    """
    raw_beats = require_list(data, "beats")
    beats: list[WeaveBeatDict] = []
    for item in raw_beats:
        if not isinstance(item, dict):
            raise JSONTypeError("beats must contain objects")
        beats.append(decode_weave_beat(item))
    return WeaveBurstDict(
        target_id=require_int(data, "target_id"),
        target_name=require_str(data, "target_name"),
        beats=beats,
        shots_dispatched=require_int(data, "shots_dispatched"),
        moves_dispatched=require_int(data, "moves_dispatched"),
        dual_before=require_int(data, "dual_before"),
        dual_after=require_int(data, "dual_after"),
        homing_before=require_int(data, "homing_before"),
        homing_after=require_int(data, "homing_after"),
        fuel_before=require_int(data, "fuel_before"),
        fuel_after=require_int(data, "fuel_after"),
        served_hits=require_int(data, "served_hits"),
        target_killed=require_bool(data, "target_killed"),
    )


def decode_weave_probe_session(data: JSONObject) -> WeaveProbeSessionDict:
    """Decode a weave probe session from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated session.
    """
    startup_timing_raw = data.get("startup_timing")
    if not isinstance(startup_timing_raw, dict):
        raise JSONTypeError("Field 'startup_timing' must be an object")
    raw_bursts = require_list(data, "bursts")
    bursts: list[WeaveBurstDict] = []
    for item in raw_bursts:
        if not isinstance(item, dict):
            raise JSONTypeError("bursts must contain objects")
        bursts.append(decode_weave_burst(item))
    return WeaveProbeSessionDict(
        session_id=require_str(data, "session_id"),
        start_timestamp_ms=require_int(data, "start_timestamp_ms"),
        end_timestamp_ms=require_int(data, "end_timestamp_ms"),
        base_url=require_str(data, "base_url"),
        spawn_x=require_int(data, "spawn_x"),
        spawn_y=require_int(data, "spawn_y"),
        beats_per_burst=require_int(data, "beats_per_burst"),
        capture_session_path=require_str(data, "capture_session_path"),
        initial_sync_timeout_ms=require_int(data, "initial_sync_timeout_ms"),
        startup_timing=decode_teleport_startup_timing(startup_timing_raw),
        bursts=bursts,
    )


__all__ = [
    "WeaveBeatDict",
    "WeaveBurstDict",
    "WeaveProbeSessionDict",
    "decode_weave_beat",
    "decode_weave_burst",
    "decode_weave_probe_session",
    "encode_weave_beat",
    "encode_weave_burst",
    "encode_weave_probe_session",
]
