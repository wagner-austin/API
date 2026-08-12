"""Tests for lobby message parsing.

The prefix table, ``parse_lobby_message`` and its per-kind parsers, and
the room-info text discriminator. ``test_parser.py`` was 846 lines; the
encode/decode pairs are now a sibling.
"""

from __future__ import annotations

import pytest

from tankpit_bot.parser import (
    LOBBY_PREFIXES,
    PREFIX_AUTH,
    PREFIX_LEAVE,
    PREFIX_RECORD,
    PREFIX_ROOM_LIST,
    PREFIX_ROOM_QUERY,
    PREFIX_STATUS,
    is_room_info_text,
)
from tankpit_bot.parser_messages import (
    ParserError,
    parse_game_record,
    parse_lobby_message,
    parse_room_info,
    parse_system_status,
)


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


def test_parse_room_info_world() -> None:
    """Test parsing real room info from capture: World (Meltdown)."""
    text = "4|World (Meltdown)|24|1,1,1,0,1,0,0|2|n|field24.gif|2025"

    result = parse_room_info(text)

    assert result["room_id"] == "4"
    assert result["name"] == "World (Meltdown)"
    assert result["field_id"] == 24
    assert result["game_modes"] == "1,1,1,0,1,0,0"
    assert result["default_troop"] == 2
    assert result["mode_code"] == "n"
    assert result["image"] == "field24.gif"
    assert result["year"] == "2025"


def test_parse_room_info_practice() -> None:
    """Test parsing real room info from capture: Practice."""
    text = "3|Practice|1|0,0,0,0,0,0,0|2|p|field01.gif|2025"

    result = parse_room_info(text)

    assert result["room_id"] == "3"
    assert result["name"] == "Practice"
    assert result["field_id"] == 1
    assert result["game_modes"] == "0,0,0,0,0,0,0"
    assert result["default_troop"] == 2
    assert result["mode_code"] == "p"
    assert result["image"] == "field01.gif"


def test_is_room_info_text_accepts_valid_room_payload() -> None:
    """Valid ROOM_LIST payloads are recognized structurally."""
    text = "1|Practice|1|0,0,0,0,0,0,0|1|p|field01.gif|2026"

    assert is_room_info_text(text) is True


def test_is_room_info_text_rejects_manual_map_click_payload() -> None:
    """Manual map-click action payloads are not treated as ROOM_LIST data."""
    text = "1|2|118|101|manual-click"

    assert is_room_info_text(text) is False


def test_is_room_info_text_rejects_non_numeric_room_id() -> None:
    """Room payloads need a numeric room identifier."""
    text = "practice|Practice|1|0,0,0,0,0,0,0|1|p|field01.gif|2026"

    assert is_room_info_text(text) is False


def test_is_room_info_text_rejects_non_numeric_player_count() -> None:
    """Room payloads need a numeric player count in field 2.

    The sibling checks on fields 0 and 4 are already pinned above and
    below; this is the same structural rule for the field between them,
    which no payload in the suite happened to violate.
    """
    text = "1|Practice|x|0,0,0,0,0,0,0|1|p|field01.gif|2026"

    assert is_room_info_text(text) is False


def test_is_room_info_text_rejects_non_gif_image_name() -> None:
    """Room payloads need a field-image filename ending in .gif."""
    text = "1|Practice|1|0,0,0,0,0,0,0|1|p|field01.png|2026"

    assert is_room_info_text(text) is False


def test_is_room_info_text_rejects_non_numeric_default_troop() -> None:
    """Room payloads need a numeric default troop field."""
    text = "1|Practice|1|0,0,0,0,0,0,0|blue|p|field01.gif|2026"

    assert is_room_info_text(text) is False


def test_is_room_info_text_rejects_world_payload_with_non_gif_image() -> None:
    """ROOM_LIST payloads reject non-gif world image names as well."""
    text = "4|World|24|1,1,1,0,1,0,0|2|n|field24.png|2026"

    assert is_room_info_text(text) is False


def test_parse_room_info_too_few_fields_raises() -> None:
    """Test parse_room_info raises for insufficient fields."""
    text = "1|Room|5"

    with pytest.raises(ParserError, match="Invalid room info"):
        parse_room_info(text)


def test_parse_room_info_invalid_count_raises() -> None:
    """Test parse_room_info raises for non-numeric player count."""
    text = "1|Room|abc|modes|a|b|image|year"

    with pytest.raises(ParserError, match="Invalid room info"):
        parse_room_info(text)


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


def test_parse_lobby_message_room_list() -> None:
    """Test parsing room list message (+)."""
    result = parse_lobby_message("+", "4|World|24|modes|2|n|image.gif|2025")

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


def test_parser_error_is_exception() -> None:
    """Test ParserError is an Exception."""
    assert issubclass(ParserError, Exception)
    err = ParserError("test error")
    assert str(err) == "test error"
