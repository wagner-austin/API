"""Capture session and summary types."""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_int,
    optional_str,
    require_dict,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot.types.literals import int_dict_to_json, mixed_dict_to_json, str_dict_to_json
from tankpit_bot.types.message import (
    CapturedMessage,
    decode_captured_message,
    encode_captured_message,
)


class GameLogEntryWithTimestamp(TypedDict):
    """A game log entry with timestamp for capture session.

    Attributes:
        timestamp_ms: Unix timestamp when entry was captured.
        text: The log message text.
        category: Category of the log entry.
    """

    timestamp_ms: int
    text: str
    category: str


class CaptureSession(TypedDict):
    """A complete WebSocket capture session.

    Attributes:
        session_id: Unique identifier for this capture session.
        start_timestamp_ms: Unix timestamp when capture started.
        end_timestamp_ms: Unix timestamp when capture ended (None if ongoing).
        base_url: Base URL of the site being captured.
        messages: List of captured messages.
        magic: XOR magic key from tankpit.magic (None if not captured).
        game_log: List of game log entries with timestamps.
        tank_names: Dictionary mapping tank IDs to names.
    """

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int | None
    base_url: str
    messages: list[CapturedMessage]
    magic: str | None
    game_log: list[GameLogEntryWithTimestamp]
    tank_names: dict[str, str]  # str keys for JSON compatibility


def encode_capture_session(session: CaptureSession) -> JSONObject:
    """Encode CaptureSession to JSON-serializable dict.

    Args:
        session: CaptureSession to encode.

    Returns:
        JSON-serializable dict representation.
    """
    encoded_messages: list[JSONValue] = [encode_captured_message(m) for m in session["messages"]]
    encoded_game_log: list[JSONValue] = [
        {
            "timestamp_ms": entry["timestamp_ms"],
            "text": entry["text"],
            "category": entry["category"],
        }
        for entry in session["game_log"]
    ]
    # Convert tank_names dict (dict[str, str] -> JSONObject)
    tank_names_json: JSONObject = str_dict_to_json(session["tank_names"])
    result: JSONObject = {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "messages": encoded_messages,
        "magic": session["magic"],
        "game_log": encoded_game_log,
        "tank_names": tank_names_json,
    }
    return result


def decode_capture_session(data: JSONObject) -> CaptureSession:
    """Decode CaptureSession from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CaptureSession.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    raw_messages = require_list(data, "messages")
    messages: list[CapturedMessage] = []
    for idx, raw_msg in enumerate(raw_messages):
        if not isinstance(raw_msg, dict):
            raise JSONTypeError(f"messages[{idx}] must be an object")
        messages.append(decode_captured_message(raw_msg))

    game_log: list[GameLogEntryWithTimestamp] = []
    for idx, raw_entry in enumerate(require_list(data, "game_log")):
        if not isinstance(raw_entry, dict):
            raise JSONTypeError(f"game_log[{idx}] must be an object")
        game_log.append(
            GameLogEntryWithTimestamp(
                timestamp_ms=require_int(raw_entry, "timestamp_ms"),
                text=require_str(raw_entry, "text"),
                category=require_str(raw_entry, "category"),
            )
        )

    tank_names: dict[str, str] = {}
    for key, value in require_dict(data, "tank_names").items():
        if not isinstance(value, str):
            raise JSONTypeError(f"tank_names[{key!r}] must be a string")
        tank_names[key] = value

    return CaptureSession(
        session_id=require_str(data, "session_id"),
        start_timestamp_ms=require_int(data, "start_timestamp_ms"),
        end_timestamp_ms=optional_int(data, "end_timestamp_ms"),
        base_url=require_str(data, "base_url"),
        messages=messages,
        magic=optional_str(data, "magic"),
        game_log=game_log,
        tank_names=tank_names,
    )


# =============================================================================
# Session Summary (processed/decoded data)
# =============================================================================


class UnknownMessageEntry(TypedDict):
    """Entry for an unknown message type.

    Attributes:
        count: Number of occurrences.
        samples: Up to 3 hex samples of message data.
    """

    count: int
    samples: list[str]


class MessageStats(TypedDict):
    """Statistics about decoded vs unknown message types.

    Attributes:
        decoded: Dict of signature -> count for decoded messages.
        unknown: Dict of signature -> entry with count and samples.
        total_received: Total number of received messages.
        decode_coverage: Percentage of messages successfully decoded.
    """

    decoded: dict[str, int]
    unknown: dict[str, UnknownMessageEntry]
    total_received: int
    decode_coverage: str


class CombatEvent(TypedDict):
    """A combat event extracted from game log or WebSocket.

    Attributes:
        timestamp_ms: When the event occurred.
        event_type: Type of event (hit, hit_by, kill, killed_by, etc).
        target: Name of target (for outgoing) or attacker (for incoming).
        tank_id: Tank ID if correlated, None otherwise.
    """

    timestamp_ms: int
    event_type: str
    target: str
    tank_id: int | None


def encode_combat_event(event: CombatEvent) -> JSONObject:
    """Encode CombatEvent to JSON-serializable dict."""
    return {
        "timestamp_ms": event["timestamp_ms"],
        "event_type": event["event_type"],
        "target": event["target"],
        "tank_id": event["tank_id"],
    }


def encode_message_stats(stats: MessageStats) -> JSONObject:
    """Encode MessageStats to JSON-serializable dict.

    Args:
        stats: MessageStats to encode.

    Returns:
        JSON-serializable dict.
    """
    # Convert decoded dict (dict[str, int] -> dict[str, JSONValue])
    decoded_json: JSONObject = int_dict_to_json(stats["decoded"])

    # Convert unknown dict (UnknownMessageEntry to JSON)
    unknown_json: JSONObject = {}
    for sig, data in stats["unknown"].items():
        # Convert samples list[str] to list[JSONValue]
        samples_json: list[JSONValue] = []
        for sample in data["samples"]:
            samples_json.append(sample)
        entry: JSONObject = {
            "count": data["count"],
            "samples": samples_json,
        }
        unknown_json[sig] = entry

    return {
        "decoded": decoded_json,
        "unknown": unknown_json,
        "total_received": stats["total_received"],
        "decode_coverage": stats["decode_coverage"],
    }


class SessionSummary(TypedDict):
    """Processed/decoded session data for easy analysis.

    Attributes:
        session_id: Unique identifier matching raw capture.
        start_timestamp_ms: When capture started.
        end_timestamp_ms: When capture ended.
        magic: XOR magic key used for decoding.
        tanks: Tank ID to name mappings.
        combat: List of combat events.
        equipment_gains: List of equipment gain events.
        game_log: Filtered game log entries (combat only).
        message_stats: Decoded vs unknown message statistics.
    """

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int | None
    magic: str | None
    tanks: dict[str, str]
    combat: list[CombatEvent]
    equipment_gains: list[dict[str, int | str]]
    game_log: list[GameLogEntryWithTimestamp]
    message_stats: MessageStats


def encode_session_summary(summary: SessionSummary) -> JSONObject:
    """Encode SessionSummary to JSON-serializable dict."""
    # Convert tanks dict (dict[str, str] -> JSONObject)
    tanks_json: JSONObject = str_dict_to_json(summary["tanks"])

    # Convert equipment_gains list (list[dict[str, int | str]] -> list[JSONValue])
    equipment_json: list[JSONValue] = []
    for gain in summary["equipment_gains"]:
        entry: JSONObject = mixed_dict_to_json(gain)
        equipment_json.append(entry)

    return {
        "session_id": summary["session_id"],
        "start_timestamp_ms": summary["start_timestamp_ms"],
        "end_timestamp_ms": summary["end_timestamp_ms"],
        "magic": summary["magic"],
        "tanks": tanks_json,
        "combat": [encode_combat_event(e) for e in summary["combat"]],
        "equipment_gains": equipment_json,
        "game_log": [
            {"timestamp_ms": e["timestamp_ms"], "text": e["text"], "category": e["category"]}
            for e in summary["game_log"]
        ],
        "message_stats": encode_message_stats(summary["message_stats"]),
    }


__all__ = [
    "CaptureSession",
    "CombatEvent",
    "GameLogEntryWithTimestamp",
    "MessageStats",
    "SessionSummary",
    "UnknownMessageEntry",
    "decode_capture_session",
    "encode_capture_session",
    "encode_combat_event",
    "encode_message_stats",
    "encode_session_summary",
]
