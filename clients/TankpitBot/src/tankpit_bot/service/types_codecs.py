"""Encode/decode functions for the bot service TypedDicts.

Every ``encode_*`` serialises a TypedDict to
:class:`platform_core.json_utils.JSONObject`. Every ``decode_*``
validates the JSON object with ``require_*`` helpers and returns the
strictly-typed dict — no soft fallbacks, no ``Any`` reach-throughs.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_int,
    require_str,
)

from tankpit_bot.bus.frame_bus import FrameStatsDict
from tankpit_bot.bus.session_status import (
    WIRE_MODES,
    LiveStatsDict,
    SessionStatusDict,
    WireMode,
)
from tankpit_bot.service.types import ModeCommandDict
from tankpit_bot.types.modes import (
    is_valid_ai_mode_state,
    require_ai_mode,
    require_ai_mode_state,
)


def _require_wire_mode(data: JSONObject, key: str) -> WireMode:
    """Validate and extract a :data:`WireMode` from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated :data:`WireMode` value.

    Raises:
        ValueError: If the value is not one of :data:`WIRE_MODES`.
        JSONTypeError: If the value is missing or not a string.
    """
    raw = require_str(data, key)
    for mode in WIRE_MODES:
        if raw == mode:
            return mode
    raise ValueError(f"{key} must be one of {WIRE_MODES}, got {raw!r}")


# =========================================================================
# ModeCommandDict codecs
# =========================================================================


def encode_mode_command(cmd: ModeCommandDict) -> JSONObject:
    """Encode :class:`ModeCommandDict` to a JSON-serializable dict.

    Args:
        cmd: Command to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {"manual_mode": cmd["manual_mode"]}


def decode_mode_command(data: JSONObject) -> ModeCommandDict:
    """Decode :class:`ModeCommandDict` from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated :class:`ModeCommandDict`.

    Raises:
        ValueError: If ``manual_mode`` is not one of :data:`WIRE_MODES`.
        JSONTypeError: If ``manual_mode`` is missing or the wrong type.
    """
    return ModeCommandDict(manual_mode=_require_wire_mode(data, "manual_mode"))


# =========================================================================
# LiveStatsDict codecs
# =========================================================================


def encode_live_stats(stats: LiveStatsDict) -> JSONObject:
    """Encode :class:`LiveStatsDict` to a JSON-serializable dict.

    Args:
        stats: Stats to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "kills": stats["kills"],
        "hits": stats["hits"],
        "misses": stats["misses"],
        "radars_used": stats["radars_used"],
        "teleports": stats["teleports"],
    }


def decode_live_stats(data: JSONObject) -> LiveStatsDict:
    """Decode :class:`LiveStatsDict` from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated :class:`LiveStatsDict`.

    Raises:
        JSONTypeError: If any required field is missing or not an int.
    """
    return LiveStatsDict(
        kills=require_int(data, "kills"),
        hits=require_int(data, "hits"),
        misses=require_int(data, "misses"),
        radars_used=require_int(data, "radars_used"),
        teleports=require_int(data, "teleports"),
    )


# =========================================================================
# SessionStatusDict codecs
# =========================================================================


def encode_session_status(status: SessionStatusDict) -> JSONObject:
    """Encode :class:`SessionStatusDict` to a JSON-serializable dict.

    Args:
        status: Status snapshot to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "running": status["running"],
        "manual_mode": status["manual_mode"],
        "active_mode": status["active_mode"],
        "active_mode_state": status["active_mode_state"],
        "session_started_ms": status["session_started_ms"],
        "tick_timestamp_ms": status["tick_timestamp_ms"],
        "stats": encode_live_stats(status["stats"]),
    }


def _require_stats_dict(data: JSONObject, key: str) -> JSONObject:
    """Extract a nested JSON object for the stats field.

    Args:
        data: Outer JSON object.
        key: Key that should hold the stats object.

    Returns:
        The nested JSON object.

    Raises:
        ValueError: If the field is missing or not an object.
    """
    raw = data.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be an object")
    return raw


def decode_session_status(data: JSONObject) -> SessionStatusDict:
    """Decode :class:`SessionStatusDict` from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated :class:`SessionStatusDict`.

    Raises:
        ValueError: If ``manual_mode`` is not a :data:`WireMode`, if
            the ``active_mode`` / ``active_mode_state`` pair is invalid,
            or if ``stats`` is missing / not an object.
        JSONTypeError: If any required field is missing or the wrong
            primitive type.
    """
    active_mode = require_ai_mode(data, "active_mode")
    active_mode_state = require_ai_mode_state(data, "active_mode_state")
    if not is_valid_ai_mode_state(active_mode, active_mode_state):
        raise ValueError(
            f"active_mode_state {active_mode_state!r} is invalid for active_mode {active_mode!r}"
        )
    return SessionStatusDict(
        running=require_bool(data, "running"),
        manual_mode=_require_wire_mode(data, "manual_mode"),
        active_mode=active_mode,
        active_mode_state=active_mode_state,
        session_started_ms=require_int(data, "session_started_ms"),
        tick_timestamp_ms=require_int(data, "tick_timestamp_ms"),
        stats=decode_live_stats(_require_stats_dict(data, "stats")),
    )


# =========================================================================
# FrameStatsDict codecs
# =========================================================================


def encode_frame_stats(stats: FrameStatsDict) -> JSONObject:
    """Encode :class:`FrameStatsDict` to a JSON-serializable dict.

    Args:
        stats: Counts read off the frame bus.

    Returns:
        JSON object carrying every field.
    """
    return {
        "published": stats["published"],
        "delivered": stats["delivered"],
        "dropped": stats["dropped"],
        "subscribers": stats["subscribers"],
    }


def decode_frame_stats(data: JSONObject) -> FrameStatsDict:
    """Validate and decode a :class:`FrameStatsDict` from JSON.

    Args:
        data: JSON object to validate.

    Returns:
        Strictly-typed frame stats.

    Raises:
        JSONTypeError: If any field is missing or not an integer.
    """
    return FrameStatsDict(
        published=require_int(data, "published"),
        delivered=require_int(data, "delivered"),
        dropped=require_int(data, "dropped"),
        subscribers=require_int(data, "subscribers"),
    )


__all__ = [
    "decode_frame_stats",
    "decode_live_stats",
    "decode_mode_command",
    "decode_session_status",
    "encode_frame_stats",
    "encode_live_stats",
    "encode_mode_command",
    "encode_session_status",
]
