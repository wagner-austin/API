"""TypedDicts for replay decision traces and session results.

Every TypedDict has encode/decode functions with require_* validation.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONValue, require_int, require_str
from typing_extensions import TypedDict

from tankpit_bot.bot.ai.modes import (
    AIMode,
    AIModeState,
    is_valid_ai_mode_state,
    require_ai_mode,
    require_ai_mode_state,
)
from tankpit_bot.bot.ai.types import EnemyThreatDict
from tankpit_bot.bot.ai.types_codecs import decode_enemy_threat, encode_enemy_threat


class ReplayTickTraceDict(TypedDict):
    """Structured record of one planner decision during replay.

    Attributes:
        tick_index: Zero-based tick counter within the replay.
        timestamp_ms: Game timestamp at decision time.
        self_x: Player X coordinate at decision time.
        self_y: Player Y coordinate at decision time.
        fuel: Player fuel at decision time.
        behavior_mode: Chosen behavior label (HUNT, COLLECT_FUEL, etc.).
        behavior_score: Priority score of the chosen behavior (0-1000).
        behavior_reason: Human-readable reason tag for the decision.
        ai_mode: Durable AI owner active for this tick.
        ai_mode_state: Durable AI substate active for this tick.
        command_type: The bot command type dispatched (move, shoot, etc.).
        target_x: Command target X coordinate.
        target_y: Command target Y coordinate.
        combat_target_id: Tank ID of the active combat target (-1 if none).
        resource_target_kind: Locked resource target kind ("", "fuel", or
            "equipment").
        visible_threats: Full threat summary sorted by distance ascending.
        container_count: Number of containers in world state this tick.
    """

    tick_index: int
    timestamp_ms: int
    self_x: int
    self_y: int
    fuel: int
    behavior_mode: str
    behavior_score: int
    behavior_reason: str
    ai_mode: AIMode
    ai_mode_state: AIModeState
    command_type: str
    target_x: int
    target_y: int
    combat_target_id: int
    resource_target_kind: str
    visible_threats: list[EnemyThreatDict]
    container_count: int


def encode_replay_tick_trace(trace: ReplayTickTraceDict) -> JSONObject:
    """Encode a replay tick trace to JSON-serializable dict.

    Args:
        trace: Tick trace to encode.

    Returns:
        JSON-serializable dict representation.
    """
    encoded_threats: list[JSONValue] = [encode_enemy_threat(t) for t in trace["visible_threats"]]
    return {
        "tick_index": trace["tick_index"],
        "timestamp_ms": trace["timestamp_ms"],
        "self_x": trace["self_x"],
        "self_y": trace["self_y"],
        "fuel": trace["fuel"],
        "behavior_mode": trace["behavior_mode"],
        "behavior_score": trace["behavior_score"],
        "behavior_reason": trace["behavior_reason"],
        "ai_mode": trace["ai_mode"],
        "ai_mode_state": trace["ai_mode_state"],
        "command_type": trace["command_type"],
        "target_x": trace["target_x"],
        "target_y": trace["target_y"],
        "combat_target_id": trace["combat_target_id"],
        "resource_target_kind": trace["resource_target_kind"],
        "visible_threats": encoded_threats,
        "container_count": trace["container_count"],
    }


def decode_replay_tick_trace(data: JSONObject) -> ReplayTickTraceDict:
    """Decode a replay tick trace from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ReplayTickTraceDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
        ValueError: If visible_threats is not a list of objects.
    """
    raw_threats = data.get("visible_threats")
    if not isinstance(raw_threats, list):
        raise ValueError("visible_threats must be a list")
    threats: list[EnemyThreatDict] = []
    for idx, raw_threat in enumerate(raw_threats):
        if not isinstance(raw_threat, dict):
            raise ValueError(f"visible_threats[{idx}] must be an object")
        threats.append(decode_enemy_threat(raw_threat))
    ai_mode = require_ai_mode(data, "ai_mode")
    ai_mode_state = require_ai_mode_state(data, "ai_mode_state")
    if not is_valid_ai_mode_state(ai_mode, ai_mode_state):
        raise ValueError(f"ai_mode_state {ai_mode_state!r} is invalid for ai_mode {ai_mode!r}")

    return ReplayTickTraceDict(
        tick_index=require_int(data, "tick_index"),
        timestamp_ms=require_int(data, "timestamp_ms"),
        self_x=require_int(data, "self_x"),
        self_y=require_int(data, "self_y"),
        fuel=require_int(data, "fuel"),
        behavior_mode=require_str(data, "behavior_mode"),
        behavior_score=require_int(data, "behavior_score"),
        behavior_reason=require_str(data, "behavior_reason"),
        ai_mode=ai_mode,
        ai_mode_state=ai_mode_state,
        command_type=require_str(data, "command_type"),
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        combat_target_id=require_int(data, "combat_target_id"),
        resource_target_kind=require_str(data, "resource_target_kind"),
        visible_threats=threats,
        container_count=require_int(data, "container_count"),
    )


class ReplaySessionResultDict(TypedDict):
    """Complete result of replaying a captured session.

    Attributes:
        session_id: Identifier from the source capture session.
        total_ticks: Number of planner ticks executed during replay.
        total_messages: Number of received messages processed.
        traces: Per-tick decision traces in chronological order.
    """

    session_id: str
    total_ticks: int
    total_messages: int
    traces: list[ReplayTickTraceDict]


def encode_replay_session_result(result: ReplaySessionResultDict) -> JSONObject:
    """Encode a replay session result to JSON-serializable dict.

    Args:
        result: Session result to encode.

    Returns:
        JSON-serializable dict representation.
    """
    encoded_traces: list[JSONValue] = [encode_replay_tick_trace(t) for t in result["traces"]]
    return {
        "session_id": result["session_id"],
        "total_ticks": result["total_ticks"],
        "total_messages": result["total_messages"],
        "traces": encoded_traces,
    }


def decode_replay_session_result(data: JSONObject) -> ReplaySessionResultDict:
    """Decode a replay session result from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ReplaySessionResultDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
        ValueError: If traces is not a list of objects.
    """
    raw_traces = data.get("traces")
    if not isinstance(raw_traces, list):
        raise ValueError("traces must be a list")
    traces: list[ReplayTickTraceDict] = []
    for idx, raw_trace in enumerate(raw_traces):
        if not isinstance(raw_trace, dict):
            raise ValueError(f"traces[{idx}] must be an object")
        traces.append(decode_replay_tick_trace(raw_trace))
    return ReplaySessionResultDict(
        session_id=require_str(data, "session_id"),
        total_ticks=require_int(data, "total_ticks"),
        total_messages=require_int(data, "total_messages"),
        traces=traces,
    )


__all__ = [
    "ReplaySessionResultDict",
    "ReplayTickTraceDict",
    "decode_replay_session_result",
    "decode_replay_tick_trace",
    "encode_replay_session_result",
    "encode_replay_tick_trace",
]
