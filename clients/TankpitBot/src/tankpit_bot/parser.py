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


# =============================================================================
# Parsed Message Types
# =============================================================================


class ParsedRoomListMessage(TypedDict):
    """Parsed room list message (+ received).

    Attributes:
        kind: Message kind discriminator.
        rooms: List of available rooms.
    """

    kind: Literal["room_list"]
    rooms: list[RoomInfo]


def encode_parsed_room_list_message(msg: ParsedRoomListMessage) -> JSONObject:
    """Encode ParsedRoomListMessage to JSON-serializable dict.

    Args:
        msg: ParsedRoomListMessage to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "kind": msg["kind"],
        "rooms": [encode_room_info(r) for r in msg["rooms"]],
    }


def decode_parsed_room_list_message(data: JSONObject) -> ParsedRoomListMessage:
    """Decode ParsedRoomListMessage from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ParsedRoomListMessage.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    kind = require_str(data, "kind")
    if kind != "room_list":
        raise JSONTypeError(f"Expected kind='room_list', got '{kind}'")

    rooms_raw = data.get("rooms")
    if not isinstance(rooms_raw, list):
        raise JSONTypeError("Missing or invalid 'rooms' field")

    rooms: list[RoomInfo] = []
    for item in rooms_raw:
        if not isinstance(item, dict):
            raise JSONTypeError("Room item must be a dict")
        rooms.append(decode_room_info(item))

    return ParsedRoomListMessage(kind="room_list", rooms=rooms)


class ParsedGameRecordMessage(TypedDict):
    """Parsed game record message (= received).

    Attributes:
        kind: Message kind discriminator.
        record: Game record data.
    """

    kind: Literal["game_record"]
    record: GameRecord


def encode_parsed_game_record_message(msg: ParsedGameRecordMessage) -> JSONObject:
    """Encode ParsedGameRecordMessage to JSON-serializable dict.

    Args:
        msg: ParsedGameRecordMessage to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "kind": msg["kind"],
        "record": encode_game_record(msg["record"]),
    }


def decode_parsed_game_record_message(data: JSONObject) -> ParsedGameRecordMessage:
    """Decode ParsedGameRecordMessage from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ParsedGameRecordMessage.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    kind = require_str(data, "kind")
    if kind != "game_record":
        raise JSONTypeError(f"Expected kind='game_record', got '{kind}'")

    record_raw = data.get("record")
    if not isinstance(record_raw, dict):
        raise JSONTypeError("Missing or invalid 'record' field")

    return ParsedGameRecordMessage(
        kind="game_record",
        record=decode_game_record(record_raw),
    )


class ParsedStatusMessage(TypedDict):
    """Parsed system status message ($ received).

    Attributes:
        kind: Message kind discriminator.
        status: Status data.
    """

    kind: Literal["status"]
    status: SystemStatus


def encode_parsed_status_message(msg: ParsedStatusMessage) -> JSONObject:
    """Encode ParsedStatusMessage to JSON-serializable dict.

    Args:
        msg: ParsedStatusMessage to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "kind": msg["kind"],
        "status": encode_system_status(msg["status"]),
    }


def decode_parsed_status_message(data: JSONObject) -> ParsedStatusMessage:
    """Decode ParsedStatusMessage from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ParsedStatusMessage.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    kind = require_str(data, "kind")
    if kind != "status":
        raise JSONTypeError(f"Expected kind='status', got '{kind}'")

    status_raw = data.get("status")
    if not isinstance(status_raw, dict):
        raise JSONTypeError("Missing or invalid 'status' field")

    return ParsedStatusMessage(
        kind="status",
        status=decode_system_status(status_raw),
    )


class ParsedLeaveMessage(TypedDict):
    """Parsed leave notification (- sent/received).

    Attributes:
        kind: Message kind discriminator.
    """

    kind: Literal["leave"]


def encode_parsed_leave_message(msg: ParsedLeaveMessage) -> JSONObject:
    """Encode ParsedLeaveMessage to JSON-serializable dict.

    Args:
        msg: ParsedLeaveMessage to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {"kind": msg["kind"]}


def decode_parsed_leave_message(data: JSONObject) -> ParsedLeaveMessage:
    """Decode ParsedLeaveMessage from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ParsedLeaveMessage.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    kind = require_str(data, "kind")
    if kind != "leave":
        raise JSONTypeError(f"Expected kind='leave', got '{kind}'")

    return ParsedLeaveMessage(kind="leave")


class ParsedRoomQueryMessage(TypedDict):
    """Parsed room query message (* sent).

    Attributes:
        kind: Message kind discriminator.
        room_id: Room to query.
    """

    kind: Literal["room_query"]
    room_id: str


def encode_parsed_room_query_message(msg: ParsedRoomQueryMessage) -> JSONObject:
    """Encode ParsedRoomQueryMessage to JSON-serializable dict.

    Args:
        msg: ParsedRoomQueryMessage to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "kind": msg["kind"],
        "room_id": msg["room_id"],
    }


def decode_parsed_room_query_message(data: JSONObject) -> ParsedRoomQueryMessage:
    """Decode ParsedRoomQueryMessage from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ParsedRoomQueryMessage.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    kind = require_str(data, "kind")
    if kind != "room_query":
        raise JSONTypeError(f"Expected kind='room_query', got '{kind}'")

    return ParsedRoomQueryMessage(
        kind="room_query",
        room_id=require_str(data, "room_id"),
    )


# Union type for all parsed lobby messages
ParsedLobbyMessage = (
    ParsedRoomListMessage
    | ParsedGameRecordMessage
    | ParsedStatusMessage
    | ParsedLeaveMessage
    | ParsedRoomQueryMessage
)


# =============================================================================
# Parsing Functions
# =============================================================================


class ParserError(Exception):
    """Error during message parsing."""


def parse_room_info(text: str) -> RoomInfo:
    """Parse single room from pipe-delimited text.

    Format: room_id|name|field_id|game_modes|default_troop|mode_code|image|year

    Args:
        text: Pipe-delimited room info.

    Returns:
        RoomInfo dict.

    Raises:
        ParserError: If format is invalid.
    """
    if not is_room_info_text(text):
        raise ParserError(f"Invalid room info: {text}")
    parts = text.split("|")

    return RoomInfo(
        room_id=parts[0],
        name=parts[1],
        field_id=int(parts[2]),
        game_modes=parts[3],
        default_troop=int(parts[4]),
        mode_code=parts[5],
        image=parts[6],
        year=parts[7],
    )


def parse_game_record(text: str) -> GameRecord:
    """Parse game record from pipe-delimited text.

    Format: room_id|date|player_name|stat1|stat2|...

    Args:
        text: Pipe-delimited game record.

    Returns:
        GameRecord dict.

    Raises:
        ParserError: If format is invalid.
    """
    parts = text.split("|")
    if len(parts) < 4:
        raise ParserError(f"Game record needs at least 4 fields, got {len(parts)}")

    return GameRecord(
        room_id=parts[0],
        date=parts[1],
        player_name=parts[2],
        stats=parts[3:],
    )


def parse_system_status(text: str) -> SystemStatus:
    """Parse system status from pipe-delimited text.

    Format: room_id|value

    Args:
        text: Pipe-delimited status.

    Returns:
        SystemStatus dict.

    Raises:
        ParserError: If format is invalid.
    """
    parts = text.split("|")
    if len(parts) < 2:
        raise ParserError(f"Status needs 2 fields, got {len(parts)}")

    return SystemStatus(
        room_id=parts[0],
        value=parts[1],
    )


def parse_lobby_message(prefix: str, text: str) -> ParsedLobbyMessage:
    """Parse a lobby message based on its prefix.

    Args:
        prefix: Single character prefix.
        text: Message text content (after prefix).

    Returns:
        Appropriate ParsedLobbyMessage variant.

    Raises:
        ParserError: If message format is invalid.
    """
    if prefix == PREFIX_ROOM_LIST:
        room = parse_room_info(text)
        return ParsedRoomListMessage(kind="room_list", rooms=[room])

    if prefix == PREFIX_RECORD:
        record = parse_game_record(text)
        return ParsedGameRecordMessage(kind="game_record", record=record)

    if prefix == PREFIX_STATUS:
        status = parse_system_status(text)
        return ParsedStatusMessage(kind="status", status=status)

    if prefix == PREFIX_LEAVE:
        return ParsedLeaveMessage(kind="leave")

    if prefix == PREFIX_ROOM_QUERY:
        return ParsedRoomQueryMessage(kind="room_query", room_id=text)

    raise ParserError(f"Unknown lobby prefix: {prefix}")


__all__ = [
    "LOBBY_PREFIXES",
    "PREFIX_AUTH",
    "PREFIX_LEAVE",
    "PREFIX_RECORD",
    "PREFIX_ROOM_LIST",
    "PREFIX_ROOM_QUERY",
    "PREFIX_STATUS",
    "GameRecord",
    "LobbyPrefix",
    "ParsedGameRecordMessage",
    "ParsedLeaveMessage",
    "ParsedLobbyMessage",
    "ParsedRoomListMessage",
    "ParsedRoomQueryMessage",
    "ParsedStatusMessage",
    "ParserError",
    "RoomInfo",
    "SystemStatus",
    "decode_game_record",
    "decode_parsed_game_record_message",
    "decode_parsed_leave_message",
    "decode_parsed_room_list_message",
    "decode_parsed_room_query_message",
    "decode_parsed_status_message",
    "decode_room_info",
    "decode_system_status",
    "encode_game_record",
    "encode_parsed_game_record_message",
    "encode_parsed_leave_message",
    "encode_parsed_room_list_message",
    "encode_parsed_room_query_message",
    "encode_parsed_status_message",
    "encode_room_info",
    "encode_system_status",
    "is_room_info_text",
    "parse_game_record",
    "parse_lobby_message",
    "parse_room_info",
    "parse_system_status",
]
