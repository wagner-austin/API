"""TypedDict models for the fire-cadence probe.

The cadence probe measures the server's shot serve rate: bursts of
shots dispatched at fixed spacings, with served shots counted from
server-refreshed 0x49 ammo snapshots (one dual/homing debit per
LANDED shot — [[weapon-selection]]'s per-shot ammo ledger).
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


class CadenceShotDict(TypedDict):
    """One dispatched shot inside a cadence burst."""

    shot_number: int
    dispatched_ms: int
    target_x: int
    target_y: int


class CadenceBurstDict(TypedDict):
    """One fixed-spacing burst against one target.

    ``served_hits`` is the ground truth: the drop in dual + homing
    counts across the burst, read from fresh 0x49 snapshots before
    and after. A served count below ``dispatched`` at some spacing
    (without the target dying) is the server's rate cap showing.
    """

    spacing_ms: int
    target_id: int
    target_name: str
    shots: list[CadenceShotDict]
    dispatched: int
    dual_before: int
    dual_after: int
    homing_before: int
    homing_after: int
    fuel_before: int
    fuel_after: int
    served_hits: int
    target_killed: bool


class CadenceProbeSessionDict(TypedDict):
    """Complete fire-cadence probe session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    shots_per_burst: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    bursts: list[CadenceBurstDict]


def encode_cadence_shot(shot: CadenceShotDict) -> JSONObject:
    """Encode one cadence shot to a JSON object.

    Args:
        shot: Shot to encode.

    Returns:
        JSON-serializable object representation.
    """
    return {
        "shot_number": shot["shot_number"],
        "dispatched_ms": shot["dispatched_ms"],
        "target_x": shot["target_x"],
        "target_y": shot["target_y"],
    }


def encode_cadence_burst(burst: CadenceBurstDict) -> JSONObject:
    """Encode one cadence burst to a JSON object.

    Args:
        burst: Burst to encode.

    Returns:
        JSON-serializable object representation.
    """
    shots: list[JSONValue] = [encode_cadence_shot(s) for s in burst["shots"]]
    return {
        "spacing_ms": burst["spacing_ms"],
        "target_id": burst["target_id"],
        "target_name": burst["target_name"],
        "shots": shots,
        "dispatched": burst["dispatched"],
        "dual_before": burst["dual_before"],
        "dual_after": burst["dual_after"],
        "homing_before": burst["homing_before"],
        "homing_after": burst["homing_after"],
        "fuel_before": burst["fuel_before"],
        "fuel_after": burst["fuel_after"],
        "served_hits": burst["served_hits"],
        "target_killed": burst["target_killed"],
    }


def encode_cadence_probe_session(session: CadenceProbeSessionDict) -> JSONObject:
    """Encode a cadence probe session to a JSON object.

    Args:
        session: Session to encode.

    Returns:
        JSON-serializable object representation.
    """
    bursts: list[JSONValue] = [encode_cadence_burst(b) for b in session["bursts"]]
    return {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "spawn_x": session["spawn_x"],
        "spawn_y": session["spawn_y"],
        "shots_per_burst": session["shots_per_burst"],
        "capture_session_path": session["capture_session_path"],
        "initial_sync_timeout_ms": session["initial_sync_timeout_ms"],
        "startup_timing": encode_teleport_startup_timing(session["startup_timing"]),
        "bursts": bursts,
    }


def decode_cadence_shot(data: JSONObject) -> CadenceShotDict:
    """Decode one cadence shot from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated shot.
    """
    return CadenceShotDict(
        shot_number=require_int(data, "shot_number"),
        dispatched_ms=require_int(data, "dispatched_ms"),
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
    )


def decode_cadence_burst(data: JSONObject) -> CadenceBurstDict:
    """Decode one cadence burst from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated burst.
    """
    raw_shots = require_list(data, "shots")
    shots: list[CadenceShotDict] = []
    for item in raw_shots:
        if not isinstance(item, dict):
            raise JSONTypeError("shots must contain objects")
        shots.append(decode_cadence_shot(item))
    return CadenceBurstDict(
        spacing_ms=require_int(data, "spacing_ms"),
        target_id=require_int(data, "target_id"),
        target_name=require_str(data, "target_name"),
        shots=shots,
        dispatched=require_int(data, "dispatched"),
        dual_before=require_int(data, "dual_before"),
        dual_after=require_int(data, "dual_after"),
        homing_before=require_int(data, "homing_before"),
        homing_after=require_int(data, "homing_after"),
        fuel_before=require_int(data, "fuel_before"),
        fuel_after=require_int(data, "fuel_after"),
        served_hits=require_int(data, "served_hits"),
        target_killed=require_bool(data, "target_killed"),
    )


def decode_cadence_probe_session(data: JSONObject) -> CadenceProbeSessionDict:
    """Decode a cadence probe session from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated session.
    """
    startup_timing_raw = data.get("startup_timing")
    if not isinstance(startup_timing_raw, dict):
        raise JSONTypeError("Field 'startup_timing' must be an object")
    raw_bursts = require_list(data, "bursts")
    bursts: list[CadenceBurstDict] = []
    for item in raw_bursts:
        if not isinstance(item, dict):
            raise JSONTypeError("bursts must contain objects")
        bursts.append(decode_cadence_burst(item))
    return CadenceProbeSessionDict(
        session_id=require_str(data, "session_id"),
        start_timestamp_ms=require_int(data, "start_timestamp_ms"),
        end_timestamp_ms=require_int(data, "end_timestamp_ms"),
        base_url=require_str(data, "base_url"),
        spawn_x=require_int(data, "spawn_x"),
        spawn_y=require_int(data, "spawn_y"),
        shots_per_burst=require_int(data, "shots_per_burst"),
        capture_session_path=require_str(data, "capture_session_path"),
        initial_sync_timeout_ms=require_int(data, "initial_sync_timeout_ms"),
        startup_timing=decode_teleport_startup_timing(startup_timing_raw),
        bursts=bursts,
    )


__all__ = [
    "CadenceBurstDict",
    "CadenceProbeSessionDict",
    "CadenceShotDict",
    "decode_cadence_burst",
    "decode_cadence_probe_session",
    "decode_cadence_shot",
    "encode_cadence_burst",
    "encode_cadence_probe_session",
    "encode_cadence_shot",
]
