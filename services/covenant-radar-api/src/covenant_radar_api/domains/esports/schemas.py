"""Match state event schemas for the esports streaming pipeline.

Provides TypedDict definitions and encoder/decoder functions for the match
snapshots consumed from Kafka. Each event is the state of one game at one
moment, from which win-probability features are derived.

Event types:
- MatchEventV1: Input match state snapshot from Kafka

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_str,
)

# =============================================================================
# Event Type Discriminator
# =============================================================================

MatchEventType = Literal["esports.match_state.v1"]

# =============================================================================
# Input Event: MatchEventV1
# =============================================================================


class MatchEventV1(TypedDict):
    """Single match state snapshot consumed from Kafka.

    Describes one game of a series at one point in time. Gold is carried in
    whole units and every count is an integer, so the whole event is exactly
    representable in JSON with no float rounding between producer and
    consumer.

    Attributes:
        type: Event type discriminator.
        event_id: UUID for deduplication.
        match_id: Match identifier (partition key).
        game_number: Which game of the series this is, starting at 1.
        game_time_seconds: Elapsed game time in seconds.
        blue_kills: Kills by the blue side.
        red_kills: Kills by the red side.
        blue_gold: Total gold earned by the blue side.
        red_gold: Total gold earned by the red side.
        blue_towers: Towers destroyed by the blue side.
        red_towers: Towers destroyed by the red side.
        blue_dragons: Dragons taken by the blue side.
        red_dragons: Dragons taken by the red side.
        blue_barons: Barons taken by the blue side.
        red_barons: Barons taken by the red side.
        timestamp: ISO datetime when the snapshot was taken.
    """

    type: MatchEventType
    event_id: str
    match_id: str
    game_number: int
    game_time_seconds: int
    blue_kills: int
    red_kills: int
    blue_gold: int
    red_gold: int
    blue_towers: int
    red_towers: int
    blue_dragons: int
    red_dragons: int
    blue_barons: int
    red_barons: int
    timestamp: str


# =============================================================================
# Factory Function
# =============================================================================


def make_match_event(
    *,
    event_id: str,
    match_id: str,
    game_number: int,
    game_time_seconds: int,
    blue_kills: int,
    red_kills: int,
    blue_gold: int,
    red_gold: int,
    blue_towers: int,
    red_towers: int,
    blue_dragons: int,
    red_dragons: int,
    blue_barons: int,
    red_barons: int,
    timestamp: str,
) -> MatchEventV1:
    """Create a match state event.

    Args:
        event_id: UUID for deduplication.
        match_id: Match identifier.
        game_number: Which game of the series, starting at 1.
        game_time_seconds: Elapsed game time in seconds.
        blue_kills: Kills by the blue side.
        red_kills: Kills by the red side.
        blue_gold: Total gold earned by the blue side.
        red_gold: Total gold earned by the red side.
        blue_towers: Towers destroyed by the blue side.
        red_towers: Towers destroyed by the red side.
        blue_dragons: Dragons taken by the blue side.
        red_dragons: Dragons taken by the red side.
        blue_barons: Barons taken by the blue side.
        red_barons: Barons taken by the red side.
        timestamp: ISO datetime when the snapshot was taken.

    Returns:
        MatchEventV1 instance.
    """
    return {
        "type": "esports.match_state.v1",
        "event_id": event_id,
        "match_id": match_id,
        "game_number": game_number,
        "game_time_seconds": game_time_seconds,
        "blue_kills": blue_kills,
        "red_kills": red_kills,
        "blue_gold": blue_gold,
        "red_gold": red_gold,
        "blue_towers": blue_towers,
        "red_towers": red_towers,
        "blue_dragons": blue_dragons,
        "red_dragons": red_dragons,
        "blue_barons": blue_barons,
        "red_barons": red_barons,
        "timestamp": timestamp,
    }


# =============================================================================
# Encoder Function
# =============================================================================


def encode_match_event(event: MatchEventV1) -> str:
    """Serialize a match event to a JSON string.

    Args:
        event: MatchEventV1 to serialize.

    Returns:
        Compact JSON string.
    """
    return dump_json_str(event)


# =============================================================================
# Literal Type Parser
# =============================================================================


def _parse_match_event_type(raw: str) -> MatchEventType:
    """Parse the match event type discriminator.

    Args:
        raw: Raw string value.

    Returns:
        Validated MatchEventType literal.

    Raises:
        JSONTypeError: If the value is not the match state discriminator.
    """
    if raw == "esports.match_state.v1":
        return "esports.match_state.v1"
    raise JSONTypeError(f"Expected 'esports.match_state.v1', got '{raw}'")


# =============================================================================
# Decoder Function
# =============================================================================


def decode_match_event(payload: str) -> MatchEventV1:
    """Parse and validate a match event from a JSON string.

    Args:
        payload: JSON string to parse.

    Returns:
        Validated MatchEventV1.

    Raises:
        JSONTypeError: If a required field is missing, has the wrong type, or
            the discriminator does not match.
        InvalidJsonError: If the payload is not valid JSON.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))
    type_raw = require_str(decoded, "type")
    event_type = _parse_match_event_type(type_raw)
    return {
        "type": event_type,
        "event_id": require_str(decoded, "event_id"),
        "match_id": require_str(decoded, "match_id"),
        "game_number": require_int(decoded, "game_number"),
        "game_time_seconds": require_int(decoded, "game_time_seconds"),
        "blue_kills": require_int(decoded, "blue_kills"),
        "red_kills": require_int(decoded, "red_kills"),
        "blue_gold": require_int(decoded, "blue_gold"),
        "red_gold": require_int(decoded, "red_gold"),
        "blue_towers": require_int(decoded, "blue_towers"),
        "red_towers": require_int(decoded, "red_towers"),
        "blue_dragons": require_int(decoded, "blue_dragons"),
        "red_dragons": require_int(decoded, "red_dragons"),
        "blue_barons": require_int(decoded, "blue_barons"),
        "red_barons": require_int(decoded, "red_barons"),
        "timestamp": require_str(decoded, "timestamp"),
    }


__all__ = [
    "MatchEventType",
    "MatchEventV1",
    "decode_match_event",
    "encode_match_event",
    "make_match_event",
]
