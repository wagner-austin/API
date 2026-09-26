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
)
from platform_core.members import require_member

from tankpit_bot.bus.session_status import (
    LiveStatsDict,
    SessionStatusDict,
    WireMode,
)
from tankpit_bot.service.types import ModeCommandDict
from tankpit_bot.types.modes import AIMode, AIModeState, is_valid_ai_mode_state

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
    return {"manual_mode": cmd["manual_mode"].value}


def decode_mode_command(data: JSONObject) -> ModeCommandDict:
    """Decode :class:`ModeCommandDict` from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated :class:`ModeCommandDict`.

    Raises:
        JSONTypeError: If ``manual_mode`` is missing, the wrong type, or
            not a :class:`WireMode` word.
    """
    return ModeCommandDict(manual_mode=require_member(data, "manual_mode", WireMode))


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
        "manual_mode": status["manual_mode"].value,
        "active_mode": status["active_mode"].value,
        "active_mode_state": status["active_mode_state"].value,
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
        ValueError: If the ``active_mode`` / ``active_mode_state`` pair is
            invalid, or if ``stats`` is missing / not an object.
        JSONTypeError: If any required field is missing, the wrong
            primitive type, or a mode word outside its vocabulary.
    """
    active_mode = require_member(data, "active_mode", AIMode)
    active_mode_state = require_member(data, "active_mode_state", AIModeState)
    if not is_valid_ai_mode_state(active_mode, active_mode_state):
        raise ValueError(
            f"active_mode_state {active_mode_state.value!r} is invalid"
            f" for active_mode {active_mode.value!r}"
        )
    return SessionStatusDict(
        running=require_bool(data, "running"),
        manual_mode=require_member(data, "manual_mode", WireMode),
        active_mode=active_mode,
        active_mode_state=active_mode_state,
        session_started_ms=require_int(data, "session_started_ms"),
        tick_timestamp_ms=require_int(data, "tick_timestamp_ms"),
        stats=decode_live_stats(_require_stats_dict(data, "stats")),
    )


__all__ = [
    "decode_live_stats",
    "decode_mode_command",
    "decode_session_status",
    "encode_live_stats",
    "encode_mode_command",
    "encode_session_status",
]
