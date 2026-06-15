"""Parse decoded WebSocket messages into structured types.

Protocol analysis based on captured session data:

Lobby messages (text, pipe-delimited):
- + received: Room list (room_id|name|field_id|modes|default_troop|mode_code|image|year)
- + sent: Join room request (encrypted payload)
- * sent: Room query (room_id)
- - sent/received: Leave notification
- % sent: Authentication
- $ received: System status (room_id|value)
- = received: Game record (room_id|date|player|stats...)

Game messages (binary):
- ! sent: Game commands (type_byte + cmd_byte + data)
- . received: State updates (binary)
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_list,
    require_str,
)

# =============================================================================
# Lobby Message Prefixes (verified from capture)
# =============================================================================

PREFIX_ROOM_LIST: Literal["+"] = "+"  # Room info (received) or join request (sent)
PREFIX_ROOM_QUERY: Literal["*"] = "*"  # Request room info
PREFIX_LEAVE: Literal["-"] = "-"  # Leave notification
PREFIX_AUTH: Literal["%"] = "%"  # Authentication
PREFIX_STATUS: Literal["$"] = "$"  # System status
PREFIX_RECORD: Literal["="] = "="  # Game record/history

LobbyPrefix = Literal["+", "*", "-", "%", "$", "="]

LOBBY_PREFIXES: tuple[LobbyPrefix, ...] = (
    PREFIX_ROOM_LIST,
    PREFIX_ROOM_QUERY,
    PREFIX_LEAVE,
    PREFIX_AUTH,
    PREFIX_STATUS,
    PREFIX_RECORD,
)


# =============================================================================
# Room Info (from + messages)
# =============================================================================


class RoomInfo(TypedDict):
    """Information about a game room.

    Format: room_id|name|field_id|game_modes|default_troop|mode_code|image|year

    Attributes:
        room_id: Unique room identifier.
        name: Room display name.
        field_id: Field/map identifier used to resolve the room image.
        game_modes: Comma-separated game mode flags.
        default_troop: Default troop/team code sent by the client for room entry.
        mode_code: One-letter room mode code from the lobby payload.
        image: Background image filename.
        year: Year value (purpose unknown).
    """

    room_id: str
    name: str
    field_id: int
    game_modes: str
    default_troop: int
    mode_code: str
    image: str
    year: str


def is_room_info_text(text: str) -> bool:
    """Return whether text matches the ROOM_LIST wire format.

    Args:
        text: Pipe-delimited room info without the leading ``+`` prefix.

    Returns:
        True when the text has the expected ROOM_LIST field structure.
    """
    parts = text.split("|")
    if len(parts) < 8:
        return False
    if not parts[0].isdigit():
        return False
    if not parts[2].isdigit():
        return False
    if not parts[4].isdigit():
        return False
    if not parts[6].endswith(".gif"):
        return False
    return parts[7].isdigit()


def encode_room_info(room: RoomInfo) -> JSONObject:
    """Encode RoomInfo to JSON-serializable dict.

    Args:
        room: RoomInfo to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "room_id": room["room_id"],
        "name": room["name"],
        "field_id": room["field_id"],
        "game_modes": room["game_modes"],
        "default_troop": room["default_troop"],
        "mode_code": room["mode_code"],
        "image": room["image"],
        "year": room["year"],
    }


def decode_room_info(data: JSONObject) -> RoomInfo:
    """Decode RoomInfo from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated RoomInfo.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return RoomInfo(
        room_id=require_str(data, "room_id"),
        name=require_str(data, "name"),
        field_id=require_int(data, "field_id"),
        game_modes=require_str(data, "game_modes"),
        default_troop=require_int(data, "default_troop"),
        mode_code=require_str(data, "mode_code"),
        image=require_str(data, "image"),
        year=require_str(data, "year"),
    )


# =============================================================================
# Game Record (from = messages)
# =============================================================================


class GameRecord(TypedDict):
    """Game record/history entry.

    Format: room_id|date|player_name|stat1|stat2|stat3|stat4|stat5

    Attributes:
        room_id: Room where game was played.
        date: Date string (e.g., "Sep. 25, 2012").
        player_name: Player who achieved the record.
        stats: List of stat values.
    """

    room_id: str
    date: str
    player_name: str
    stats: list[str]


def encode_game_record(record: GameRecord) -> JSONObject:
    """Encode GameRecord to JSON-serializable dict.

    Args:
        record: GameRecord to encode.

    Returns:
        JSON-serializable dict representation.
    """
    # Create JSONValue-compatible list from stats
    stats_json: list[JSONValue] = list(record["stats"])
    return {
        "room_id": record["room_id"],
        "date": record["date"],
        "player_name": record["player_name"],
        "stats": stats_json,
    }


def decode_game_record(data: JSONObject) -> GameRecord:
    """Decode GameRecord from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated GameRecord.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    stats_raw = require_list(data, "stats")
    stats: list[str] = []
    for item in stats_raw:
        if not isinstance(item, str):
            raise JSONTypeError("Stats must be list of strings")
        stats.append(item)

    return GameRecord(
        room_id=require_str(data, "room_id"),
        date=require_str(data, "date"),
        player_name=require_str(data, "player_name"),
        stats=stats,
    )


# =============================================================================
# System Status (from $ messages)
# =============================================================================


class SystemStatus(TypedDict):
    """System status message.

    Format: room_id|value

    Attributes:
        room_id: Room identifier.
        value: Status value.
    """

    room_id: str
    value: str


def encode_system_status(status: SystemStatus) -> JSONObject:
    """Encode SystemStatus to JSON-serializable dict.

    Args:
        status: SystemStatus to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "room_id": status["room_id"],
        "value": status["value"],
    }


def decode_system_status(data: JSONObject) -> SystemStatus:
    """Decode SystemStatus from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated SystemStatus.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return SystemStatus(
        room_id=require_str(data, "room_id"),
        value=require_str(data, "value"),
    )


__all__ = [
    "GameRecord",
    "RoomInfo",
    "SystemStatus",
    "decode_game_record",
    "decode_room_info",
    "decode_system_status",
    "encode_game_record",
    "encode_room_info",
    "encode_system_status",
    "is_room_info_text",
]
