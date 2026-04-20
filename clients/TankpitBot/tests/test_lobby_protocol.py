"""Tests for typed lobby packet builders."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.protocol.codec import ProtocolCodec
from tankpit_bot.protocol.framing import decode_frame
from tankpit_bot.protocol.lobby import (
    ROOM_ENTRY_DEFAULT_X,
    ROOM_ENTRY_DEFAULT_Y,
    ROOM_ENTRY_METADATA_SUFFIX,
    RoomEnterRequestDict,
    RoomSelectRequestDict,
    build_room_enter_metadata,
    decode_room_enter_request,
    decode_room_select_request,
    encode_room_enter_request,
    encode_room_select_request,
    serialize_room_enter_request,
    serialize_room_select_request,
)


def test_encode_room_select_request() -> None:
    """Room-select requests encode to JSON-ready objects."""
    request: RoomSelectRequestDict = {"room_id": "1"}

    result = encode_room_select_request(request)

    assert result == {"room_id": "1"}


def test_decode_room_select_request() -> None:
    """Room-select requests decode from validated JSON."""
    data: JSONObject = {"room_id": "4"}

    result = decode_room_select_request(data)

    assert result["room_id"] == "4"


def test_encode_room_enter_request() -> None:
    """Room-enter requests encode to JSON-ready objects."""
    request: RoomEnterRequestDict = {
        "room_id": "1",
        "troop": 2,
        "preview_x": ROOM_ENTRY_DEFAULT_X,
        "preview_y": ROOM_ENTRY_DEFAULT_Y,
        "metadata": "https://tankpit.com/play|https://tankpit.com/game/tpclient.js|j2lk",
    }

    result = encode_room_enter_request(request)

    assert result["room_id"] == "1"
    assert result["troop"] == 2
    assert result["preview_x"] == ROOM_ENTRY_DEFAULT_X
    assert result["preview_y"] == ROOM_ENTRY_DEFAULT_Y
    assert (
        result["metadata"] == "https://tankpit.com/play|https://tankpit.com/game/tpclient.js|j2lk"
    )


def test_decode_room_enter_request() -> None:
    """Room-enter requests decode from validated JSON."""
    data: JSONObject = {
        "room_id": "1",
        "troop": 2,
        "preview_x": 105,
        "preview_y": 101,
        "metadata": "https://tankpit.com/play|https://tankpit.com/game/tpclient.js|j2lk",
    }

    result = decode_room_enter_request(data)

    assert result["room_id"] == "1"
    assert result["troop"] == 2
    assert result["preview_x"] == 105
    assert result["preview_y"] == 101
    assert result["metadata"].endswith("|j2lk")


def test_decode_room_enter_request_rejects_negative_troop() -> None:
    """Room-enter decode rejects negative troop values."""
    data: JSONObject = {
        "room_id": "1",
        "troop": -1,
        "preview_x": 105,
        "preview_y": 101,
        "metadata": "meta",
    }

    with pytest.raises(JSONTypeError, match="troop must be non-negative"):
        decode_room_enter_request(data)


def test_decode_room_enter_request_rejects_negative_preview_x() -> None:
    """Room-enter decode rejects negative preview_x values."""
    data: JSONObject = {
        "room_id": "1",
        "troop": 2,
        "preview_x": -1,
        "preview_y": 101,
        "metadata": "meta",
    }

    with pytest.raises(JSONTypeError, match="preview_x must be non-negative"):
        decode_room_enter_request(data)


def test_decode_room_enter_request_rejects_negative_preview_y() -> None:
    """Room-enter decode rejects negative preview_y values."""
    data: JSONObject = {
        "room_id": "1",
        "troop": 2,
        "preview_x": 105,
        "preview_y": -1,
        "metadata": "meta",
    }

    with pytest.raises(JSONTypeError, match="preview_y must be non-negative"):
        decode_room_enter_request(data)


def test_build_room_enter_metadata_truncates_to_client_limit() -> None:
    """Room-enter metadata matches the client suffix and truncation rule."""
    page_url = "https://tankpit.com/play?" + ("a" * 260)
    tpclient_url = "https://tankpit.com/game/tpclient-test.js"

    result = build_room_enter_metadata(page_url, tpclient_url)

    assert len(result) == 255
    assert result.startswith("https://tankpit.com/play?")


def test_build_room_enter_metadata_preserves_suffix_when_under_limit() -> None:
    """Room-enter metadata includes the client suffix when not truncated."""
    result = build_room_enter_metadata(
        "https://tankpit.com/play",
        "https://tankpit.com/game/tpclient-test.js",
    )

    assert result.endswith("|" + ROOM_ENTRY_METADATA_SUFFIX)


def test_serialize_room_select_request() -> None:
    """Room-select serialization produces a framed `*room_id` packet."""
    request: RoomSelectRequestDict = {"room_id": "1"}

    framed = serialize_room_select_request(request)
    body, remaining = decode_frame(framed)

    assert remaining == b""
    assert body == b"*1"


def test_serialize_room_enter_request() -> None:
    """Room-enter serialization produces the verified `+...` wire format."""
    codec = ProtocolCodec("A" * 1000, "B" * 20)
    metadata = "https://tankpit.com/play|https://tankpit.com/game/tpclient.js|j2lk"
    request: RoomEnterRequestDict = {
        "room_id": "1",
        "troop": 2,
        "preview_x": 128,
        "preview_y": 128,
        "metadata": metadata,
    }

    framed = serialize_room_enter_request(request, codec)
    body, remaining = decode_frame(framed)

    assert remaining == b""
    assert body.startswith(b"+1|2|128|128|")

    encoded_metadata = body.split(b"|", 4)[4]
    decoded_metadata = codec.decode(encoded_metadata).decode("utf-8")
    assert decoded_metadata == metadata
