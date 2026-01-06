"""Tests for tankpit_bot.parser module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.parser import (
    LOBBY_PREFIXES,
    PREFIX_AUTH,
    PREFIX_LEAVE,
    PREFIX_RECORD,
    PREFIX_ROOM_LIST,
    PREFIX_ROOM_QUERY,
    PREFIX_STATUS,
    GameRecord,
    ParsedGameRecordMessage,
    ParsedLeaveMessage,
    ParsedRoomListMessage,
    ParsedRoomQueryMessage,
    ParsedStatusMessage,
    ParserError,
    RoomInfo,
    SystemStatus,
    decode_game_record,
    decode_parsed_game_record_message,
    decode_parsed_leave_message,
    decode_parsed_room_list_message,
    decode_parsed_room_query_message,
    decode_parsed_status_message,
    decode_room_info,
    decode_system_status,
    encode_game_record,
    encode_parsed_game_record_message,
    encode_parsed_leave_message,
    encode_parsed_room_list_message,
    encode_parsed_room_query_message,
    encode_parsed_status_message,
    encode_room_info,
    encode_system_status,
    parse_game_record,
    parse_lobby_message,
    parse_room_info,
    parse_system_status,
)

# =============================================================================
# Constants Tests
# =============================================================================


def test_prefix_room_list() -> None:
    """Test PREFIX_ROOM_LIST is '+'."""
    assert PREFIX_ROOM_LIST == "+"


def test_prefix_room_query() -> None:
    """Test PREFIX_ROOM_QUERY is '*'."""
    assert PREFIX_ROOM_QUERY == "*"


def test_prefix_leave() -> None:
    """Test PREFIX_LEAVE is '-'."""
    assert PREFIX_LEAVE == "-"


def test_prefix_auth() -> None:
    """Test PREFIX_AUTH is '%'."""
    assert PREFIX_AUTH == "%"


def test_prefix_status() -> None:
    """Test PREFIX_STATUS is '$'."""
    assert PREFIX_STATUS == "$"


def test_prefix_record() -> None:
    """Test PREFIX_RECORD is '='."""
    assert PREFIX_RECORD == "="


def test_lobby_prefixes_contains_all() -> None:
    """Test LOBBY_PREFIXES contains all prefix constants."""
    assert PREFIX_ROOM_LIST in LOBBY_PREFIXES
    assert PREFIX_ROOM_QUERY in LOBBY_PREFIXES
    assert PREFIX_LEAVE in LOBBY_PREFIXES
    assert PREFIX_AUTH in LOBBY_PREFIXES
    assert PREFIX_STATUS in LOBBY_PREFIXES
    assert PREFIX_RECORD in LOBBY_PREFIXES
    assert len(LOBBY_PREFIXES) == 6


# =============================================================================
# RoomInfo Encode/Decode Tests
# =============================================================================


def test_encode_room_info() -> None:
    """Test encoding RoomInfo to JSON."""
    room = RoomInfo(
        room_id="4",
        name="World (Meltdown)",
        player_count=42,
        game_modes="1,1,1,0,1,0,0",
        image="field42.gif",
        year="2025",
    )

    result = encode_room_info(room)

    assert result["room_id"] == "4"
    assert result["name"] == "World (Meltdown)"
    assert result["player_count"] == 42
    assert result["game_modes"] == "1,1,1,0,1,0,0"
    assert result["image"] == "field42.gif"
    assert result["year"] == "2025"


def test_decode_room_info() -> None:
    """Test decoding RoomInfo from JSON."""
    data: JSONObject = {
        "room_id": "3",
        "name": "Practice",
        "player_count": 1,
        "game_modes": "0,0,0,0,0,0,0",
        "image": "field01.gif",
        "year": "2025",
    }

    result = decode_room_info(data)

    assert result["room_id"] == "3"
    assert result["name"] == "Practice"
    assert result["player_count"] == 1


def test_decode_room_info_missing_field_raises() -> None:
    """Test decode_room_info raises for missing field."""
    data: JSONObject = {"room_id": "123", "name": "Test"}

    with pytest.raises(JSONTypeError):
        decode_room_info(data)


# =============================================================================
# GameRecord Encode/Decode Tests
# =============================================================================


def test_encode_game_record() -> None:
    """Test encoding GameRecord to JSON."""
    record = GameRecord(
        room_id="4",
        date="Sep. 25, 2012",
        player_name="Yuppler",
        stats=["4", "9", "9", "9", "9"],
    )

    result = encode_game_record(record)

    assert result["room_id"] == "4"
    assert result["date"] == "Sep. 25, 2012"
    assert result["player_name"] == "Yuppler"
    assert result["stats"] == ["4", "9", "9", "9", "9"]


def test_decode_game_record() -> None:
    """Test decoding GameRecord from JSON."""
    data: JSONObject = {
        "room_id": "4",
        "date": "Sep. 25, 2012",
        "player_name": "Yuppler",
        "stats": ["4", "9", "9", "9", "9"],
    }

    result = decode_game_record(data)

    assert result["room_id"] == "4"
    assert result["player_name"] == "Yuppler"
    assert len(result["stats"]) == 5


def test_decode_game_record_non_string_stats_raises() -> None:
    """Test decode_game_record raises for non-string stats."""
    data: JSONObject = {
        "room_id": "1",
        "date": "date",
        "player_name": "player",
        "stats": [1, 2, 3],
    }

    with pytest.raises(JSONTypeError, match="Stats must be list of strings"):
        decode_game_record(data)


# =============================================================================
# SystemStatus Encode/Decode Tests
# =============================================================================


def test_encode_system_status() -> None:
    """Test encoding SystemStatus to JSON."""
    status = SystemStatus(room_id="4", value="0")

    result = encode_system_status(status)

    assert result["room_id"] == "4"
    assert result["value"] == "0"


def test_decode_system_status() -> None:
    """Test decoding SystemStatus from JSON."""
    data: JSONObject = {"room_id": "4", "value": "0"}

    result = decode_system_status(data)

    assert result["room_id"] == "4"
    assert result["value"] == "0"


# =============================================================================
# ParsedRoomListMessage Encode/Decode Tests
# =============================================================================


def test_encode_parsed_room_list_message() -> None:
    """Test encoding ParsedRoomListMessage to JSON."""
    msg = ParsedRoomListMessage(
        kind="room_list",
        rooms=[
            RoomInfo(
                room_id="4",
                name="World",
                player_count=42,
                game_modes="1,1,1,0,1,0,0",
                image="field42.gif",
                year="2025",
            ),
        ],
    )

    result = encode_parsed_room_list_message(msg)

    assert result["kind"] == "room_list"
    # Verify structure by round-tripping through decode
    decoded = decode_parsed_room_list_message(result)
    assert len(decoded["rooms"]) == 1


def test_decode_parsed_room_list_message() -> None:
    """Test decoding ParsedRoomListMessage from JSON."""
    data: JSONObject = {
        "kind": "room_list",
        "rooms": [
            {
                "room_id": "3",
                "name": "Practice",
                "player_count": 1,
                "game_modes": "0,0,0,0,0,0,0",
                "image": "field01.gif",
                "year": "2025",
            }
        ],
    }

    result = decode_parsed_room_list_message(data)

    assert result["kind"] == "room_list"
    assert len(result["rooms"]) == 1


def test_decode_parsed_room_list_message_wrong_kind_raises() -> None:
    """Test decode_parsed_room_list_message raises for wrong kind."""
    data: JSONObject = {"kind": "status", "rooms": []}

    with pytest.raises(JSONTypeError, match="Expected kind='room_list'"):
        decode_parsed_room_list_message(data)


def test_decode_parsed_room_list_message_missing_rooms_raises() -> None:
    """Test decode_parsed_room_list_message raises for missing rooms."""
    data: JSONObject = {"kind": "room_list"}

    with pytest.raises(JSONTypeError, match="Missing or invalid 'rooms'"):
        decode_parsed_room_list_message(data)


def test_decode_parsed_room_list_message_rooms_not_list_raises() -> None:
    """Test decode_parsed_room_list_message raises when rooms is not list."""
    data: JSONObject = {"kind": "room_list", "rooms": "not a list"}

    with pytest.raises(JSONTypeError, match="Missing or invalid 'rooms'"):
        decode_parsed_room_list_message(data)


def test_decode_parsed_room_list_message_room_not_dict_raises() -> None:
    """Test decode_parsed_room_list_message raises when room item is not dict."""
    data: JSONObject = {"kind": "room_list", "rooms": ["not a dict"]}

    with pytest.raises(JSONTypeError, match="Room item must be a dict"):
        decode_parsed_room_list_message(data)


# =============================================================================
# ParsedGameRecordMessage Encode/Decode Tests
# =============================================================================


def test_encode_parsed_game_record_message() -> None:
    """Test encoding ParsedGameRecordMessage to JSON."""
    msg = ParsedGameRecordMessage(
        kind="game_record",
        record=GameRecord(
            room_id="4",
            date="Sep. 25, 2012",
            player_name="Yuppler",
            stats=["4", "9", "9", "9", "9"],
        ),
    )

    result = encode_parsed_game_record_message(msg)

    assert result["kind"] == "game_record"
    # Verify structure by round-tripping through decode
    decoded = decode_parsed_game_record_message(result)
    assert decoded["record"]["player_name"] == "Yuppler"


def test_decode_parsed_game_record_message() -> None:
    """Test decoding ParsedGameRecordMessage from JSON."""
    data: JSONObject = {
        "kind": "game_record",
        "record": {
            "room_id": "4",
            "date": "Sep. 25, 2012",
            "player_name": "Yuppler",
            "stats": ["4", "9"],
        },
    }

    result = decode_parsed_game_record_message(data)

    assert result["kind"] == "game_record"
    assert result["record"]["player_name"] == "Yuppler"


def test_decode_parsed_game_record_message_wrong_kind_raises() -> None:
    """Test decode_parsed_game_record_message raises for wrong kind."""
    data: JSONObject = {
        "kind": "room_list",
        "record": {"room_id": "1", "date": "d", "player_name": "p", "stats": []},
    }

    with pytest.raises(JSONTypeError, match="Expected kind='game_record'"):
        decode_parsed_game_record_message(data)


def test_decode_parsed_game_record_message_missing_record_raises() -> None:
    """Test decode_parsed_game_record_message raises for missing record."""
    data: JSONObject = {"kind": "game_record"}

    with pytest.raises(JSONTypeError, match="Missing or invalid 'record'"):
        decode_parsed_game_record_message(data)


def test_decode_parsed_game_record_message_record_not_dict_raises() -> None:
    """Test decode_parsed_game_record_message raises when record is not dict."""
    data: JSONObject = {"kind": "game_record", "record": "not a dict"}

    with pytest.raises(JSONTypeError, match="Missing or invalid 'record'"):
        decode_parsed_game_record_message(data)


# =============================================================================
# ParsedStatusMessage Encode/Decode Tests
# =============================================================================


def test_encode_parsed_status_message() -> None:
    """Test encoding ParsedStatusMessage to JSON."""
    msg = ParsedStatusMessage(
        kind="status",
        status=SystemStatus(room_id="4", value="0"),
    )

    result = encode_parsed_status_message(msg)

    assert result["kind"] == "status"
    # Verify structure by round-tripping through decode
    decoded = decode_parsed_status_message(result)
    assert decoded["status"]["room_id"] == "4"


def test_decode_parsed_status_message() -> None:
    """Test decoding ParsedStatusMessage from JSON."""
    data: JSONObject = {
        "kind": "status",
        "status": {"room_id": "4", "value": "0"},
    }

    result = decode_parsed_status_message(data)

    assert result["kind"] == "status"
    assert result["status"]["value"] == "0"


def test_decode_parsed_status_message_wrong_kind_raises() -> None:
    """Test decode_parsed_status_message raises for wrong kind."""
    data: JSONObject = {"kind": "room_list", "status": {"room_id": "1", "value": "0"}}

    with pytest.raises(JSONTypeError, match="Expected kind='status'"):
        decode_parsed_status_message(data)


def test_decode_parsed_status_message_missing_status_raises() -> None:
    """Test decode_parsed_status_message raises for missing status."""
    data: JSONObject = {"kind": "status"}

    with pytest.raises(JSONTypeError, match="Missing or invalid 'status'"):
        decode_parsed_status_message(data)


def test_decode_parsed_status_message_status_not_dict_raises() -> None:
    """Test decode_parsed_status_message raises when status is not dict."""
    data: JSONObject = {"kind": "status", "status": "not a dict"}

    with pytest.raises(JSONTypeError, match="Missing or invalid 'status'"):
        decode_parsed_status_message(data)


# =============================================================================
# ParsedLeaveMessage Encode/Decode Tests
# =============================================================================


def test_encode_parsed_leave_message() -> None:
    """Test encoding ParsedLeaveMessage to JSON."""
    msg = ParsedLeaveMessage(kind="leave")

    result = encode_parsed_leave_message(msg)

    assert result["kind"] == "leave"


def test_decode_parsed_leave_message() -> None:
    """Test decoding ParsedLeaveMessage from JSON."""
    data: JSONObject = {"kind": "leave"}

    result = decode_parsed_leave_message(data)

    assert result["kind"] == "leave"


def test_decode_parsed_leave_message_wrong_kind_raises() -> None:
    """Test decode_parsed_leave_message raises for wrong kind."""
    data: JSONObject = {"kind": "status"}

    with pytest.raises(JSONTypeError, match="Expected kind='leave'"):
        decode_parsed_leave_message(data)


# =============================================================================
# ParsedRoomQueryMessage Encode/Decode Tests
# =============================================================================


def test_encode_parsed_room_query_message() -> None:
    """Test encoding ParsedRoomQueryMessage to JSON."""
    msg = ParsedRoomQueryMessage(kind="room_query", room_id="4")

    result = encode_parsed_room_query_message(msg)

    assert result["kind"] == "room_query"
    assert result["room_id"] == "4"


def test_decode_parsed_room_query_message() -> None:
    """Test decoding ParsedRoomQueryMessage from JSON."""
    data: JSONObject = {"kind": "room_query", "room_id": "4"}

    result = decode_parsed_room_query_message(data)

    assert result["kind"] == "room_query"
    assert result["room_id"] == "4"


def test_decode_parsed_room_query_message_wrong_kind_raises() -> None:
    """Test decode_parsed_room_query_message raises for wrong kind."""
    data: JSONObject = {"kind": "leave", "room_id": "4"}

    with pytest.raises(JSONTypeError, match="Expected kind='room_query'"):
        decode_parsed_room_query_message(data)


# =============================================================================
# parse_room_info Tests (using real captured data format)
# =============================================================================


def test_parse_room_info_world() -> None:
    """Test parsing real room info from capture: World (Meltdown)."""
    text = "4|World (Meltdown)|42|1,1,1,0,1,0,0|2|n|field42.gif|2025"

    result = parse_room_info(text)

    assert result["room_id"] == "4"
    assert result["name"] == "World (Meltdown)"
    assert result["player_count"] == 42
    assert result["game_modes"] == "1,1,1,0,1,0,0"
    assert result["image"] == "field42.gif"
    assert result["year"] == "2025"


def test_parse_room_info_practice() -> None:
    """Test parsing real room info from capture: Practice."""
    text = "3|Practice|1|0,0,0,0,0,0,0|1|p|field01.gif|2025"

    result = parse_room_info(text)

    assert result["room_id"] == "3"
    assert result["name"] == "Practice"
    assert result["player_count"] == 1
    assert result["game_modes"] == "0,0,0,0,0,0,0"
    assert result["image"] == "field01.gif"


def test_parse_room_info_too_few_fields_raises() -> None:
    """Test parse_room_info raises for insufficient fields."""
    text = "1|Room|5"

    with pytest.raises(ParserError, match="needs 8 fields"):
        parse_room_info(text)


def test_parse_room_info_invalid_count_raises() -> None:
    """Test parse_room_info raises for non-numeric player count."""
    text = "1|Room|abc|modes|a|b|image|year"

    with pytest.raises(ParserError, match="Invalid player count"):
        parse_room_info(text)


# =============================================================================
# parse_game_record Tests (using real captured data format)
# =============================================================================


def test_parse_game_record_real() -> None:
    """Test parsing real game record from capture."""
    text = "4|Sep. 25, 2012|Yuppler|4|9|9|9|9"

    result = parse_game_record(text)

    assert result["room_id"] == "4"
    assert result["date"] == "Sep. 25, 2012"
    assert result["player_name"] == "Yuppler"
    assert result["stats"] == ["4", "9", "9", "9", "9"]


def test_parse_game_record_minimal() -> None:
    """Test parsing minimal game record."""
    text = "1|date|player|stat"

    result = parse_game_record(text)

    assert result["room_id"] == "1"
    assert result["stats"] == ["stat"]


def test_parse_game_record_too_few_fields_raises() -> None:
    """Test parse_game_record raises for insufficient fields."""
    text = "1|date|player"

    with pytest.raises(ParserError, match="needs at least 4 fields"):
        parse_game_record(text)


# =============================================================================
# parse_system_status Tests (using real captured data format)
# =============================================================================


def test_parse_system_status_real() -> None:
    """Test parsing real system status from capture: $4|0."""
    text = "4|0"

    result = parse_system_status(text)

    assert result["room_id"] == "4"
    assert result["value"] == "0"


def test_parse_system_status_too_few_fields_raises() -> None:
    """Test parse_system_status raises for insufficient fields."""
    text = "4"

    with pytest.raises(ParserError, match="needs 2 fields"):
        parse_system_status(text)


# =============================================================================
# parse_lobby_message Tests
# =============================================================================


def test_parse_lobby_message_room_list() -> None:
    """Test parsing room list message (+)."""
    result = parse_lobby_message("+", "4|World|42|modes|a|b|image.gif|2025")

    assert result["kind"] == "room_list"
    assert len(result["rooms"]) == 1
    assert result["rooms"][0]["name"] == "World"


def test_parse_lobby_message_game_record() -> None:
    """Test parsing game record message (=)."""
    result = parse_lobby_message("=", "4|Sep. 25, 2012|Yuppler|4|9|9|9|9")

    assert result["kind"] == "game_record"
    assert result["record"]["player_name"] == "Yuppler"


def test_parse_lobby_message_status() -> None:
    """Test parsing status message ($)."""
    result = parse_lobby_message("$", "4|0")

    assert result["kind"] == "status"
    assert result["status"]["room_id"] == "4"


def test_parse_lobby_message_leave() -> None:
    """Test parsing leave message (-)."""
    result = parse_lobby_message("-", "")

    assert result["kind"] == "leave"


def test_parse_lobby_message_room_query() -> None:
    """Test parsing room query message (*)."""
    result = parse_lobby_message("*", "4")

    assert result["kind"] == "room_query"
    assert result["room_id"] == "4"


def test_parse_lobby_message_unknown_prefix_raises() -> None:
    """Test parse_lobby_message raises for unknown prefix."""
    with pytest.raises(ParserError, match="Unknown lobby prefix"):
        parse_lobby_message("@", "data")


# =============================================================================
# Error Class Tests
# =============================================================================


def test_parser_error_is_exception() -> None:
    """Test ParserError is an Exception."""
    assert issubclass(ParserError, Exception)
    err = ParserError("test error")
    assert str(err) == "test error"


# =============================================================================
# Round-trip Tests
# =============================================================================


def test_room_info_roundtrip() -> None:
    """Test RoomInfo encode then decode returns equivalent."""
    original = RoomInfo(
        room_id="4",
        name="World",
        player_count=42,
        game_modes="1,1,1,0,1,0,0",
        image="field.gif",
        year="2025",
    )

    encoded = encode_room_info(original)
    decoded = decode_room_info(encoded)

    assert decoded == original


def test_game_record_roundtrip() -> None:
    """Test GameRecord encode then decode returns equivalent."""
    original = GameRecord(
        room_id="4",
        date="Sep. 25, 2012",
        player_name="Yuppler",
        stats=["4", "9"],
    )

    encoded = encode_game_record(original)
    decoded = decode_game_record(encoded)

    assert decoded == original


def test_system_status_roundtrip() -> None:
    """Test SystemStatus encode then decode returns equivalent."""
    original = SystemStatus(room_id="4", value="0")

    encoded = encode_system_status(original)
    decoded = decode_system_status(encoded)

    assert decoded == original


def test_parsed_room_list_message_roundtrip() -> None:
    """Test ParsedRoomListMessage encode then decode returns equivalent."""
    original = ParsedRoomListMessage(
        kind="room_list",
        rooms=[
            RoomInfo(
                room_id="4",
                name="World",
                player_count=42,
                game_modes="1,1,1,0,1,0,0",
                image="field.gif",
                year="2025",
            )
        ],
    )

    encoded = encode_parsed_room_list_message(original)
    decoded = decode_parsed_room_list_message(encoded)

    assert decoded["kind"] == original["kind"]
    assert len(decoded["rooms"]) == len(original["rooms"])


def test_parsed_game_record_message_roundtrip() -> None:
    """Test ParsedGameRecordMessage encode then decode returns equivalent."""
    original = ParsedGameRecordMessage(
        kind="game_record",
        record=GameRecord(
            room_id="4",
            date="Sep. 25, 2012",
            player_name="Yuppler",
            stats=["4", "9"],
        ),
    )

    encoded = encode_parsed_game_record_message(original)
    decoded = decode_parsed_game_record_message(encoded)

    assert decoded["kind"] == original["kind"]


def test_parsed_status_message_roundtrip() -> None:
    """Test ParsedStatusMessage encode then decode returns equivalent."""
    original = ParsedStatusMessage(
        kind="status",
        status=SystemStatus(room_id="4", value="0"),
    )

    encoded = encode_parsed_status_message(original)
    decoded = decode_parsed_status_message(encoded)

    assert decoded == original


def test_parsed_leave_message_roundtrip() -> None:
    """Test ParsedLeaveMessage encode then decode returns equivalent."""
    original = ParsedLeaveMessage(kind="leave")

    encoded = encode_parsed_leave_message(original)
    decoded = decode_parsed_leave_message(encoded)

    assert decoded == original


def test_parsed_room_query_message_roundtrip() -> None:
    """Test ParsedRoomQueryMessage encode then decode returns equivalent."""
    original = ParsedRoomQueryMessage(kind="room_query", room_id="4")

    encoded = encode_parsed_room_query_message(original)
    decoded = decode_parsed_room_query_message(encoded)

    assert decoded == original
