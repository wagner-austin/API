"""Typed lobby packet builders for room selection and entry."""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import JSONObject, JSONTypeError, require_int, require_str

from tankpit_bot.protocol.codec import ProtocolCodec
from tankpit_bot.protocol.framing import encode_frame

ROOM_ENTRY_DEFAULT_X = 128
ROOM_ENTRY_DEFAULT_Y = 128
ROOM_ENTRY_METADATA_SUFFIX = "j2lk"
_ROOM_ENTRY_METADATA_MAX_LEN = 255


class RoomSelectRequestDict(TypedDict):
    """Typed room-selection request."""

    room_id: str


class RoomEnterRequestDict(TypedDict):
    """Typed room-entry request."""

    room_id: str
    troop: int
    preview_x: int
    preview_y: int
    metadata: str


def encode_room_select_request(request: RoomSelectRequestDict) -> JSONObject:
    """Encode a room-selection request.

    Args:
        request: Typed room-selection request.

    Returns:
        JSON-serializable object.
    """
    return {"room_id": request["room_id"]}


def decode_room_select_request(data: JSONObject) -> RoomSelectRequestDict:
    """Decode a room-selection request.

    Args:
        data: JSON object to decode.

    Returns:
        Validated room-selection request.
    """
    return RoomSelectRequestDict(room_id=require_str(data, "room_id"))


def encode_room_enter_request(request: RoomEnterRequestDict) -> JSONObject:
    """Encode a room-entry request.

    Args:
        request: Typed room-entry request.

    Returns:
        JSON-serializable object.
    """
    return {
        "room_id": request["room_id"],
        "troop": request["troop"],
        "preview_x": request["preview_x"],
        "preview_y": request["preview_y"],
        "metadata": request["metadata"],
    }


def decode_room_enter_request(data: JSONObject) -> RoomEnterRequestDict:
    """Decode a room-entry request.

    Args:
        data: JSON object to decode.

    Returns:
        Validated room-entry request.

    Raises:
        JSONTypeError: If any integer field is negative.
    """
    troop = require_int(data, "troop")
    preview_x = require_int(data, "preview_x")
    preview_y = require_int(data, "preview_y")
    if troop < 0:
        raise JSONTypeError("troop must be non-negative")
    if preview_x < 0:
        raise JSONTypeError("preview_x must be non-negative")
    if preview_y < 0:
        raise JSONTypeError("preview_y must be non-negative")
    return RoomEnterRequestDict(
        room_id=require_str(data, "room_id"),
        troop=troop,
        preview_x=preview_x,
        preview_y=preview_y,
        metadata=require_str(data, "metadata"),
    )


def build_room_enter_metadata(page_url: str, tpclient_url: str) -> str:
    """Build the plaintext room-entry metadata string.

    Args:
        page_url: Current game page URL.
        tpclient_url: Loaded tpclient script URL.

    Returns:
        Metadata string truncated to the client-enforced limit.
    """
    metadata = f"{page_url}|{tpclient_url}|{ROOM_ENTRY_METADATA_SUFFIX}"
    return metadata[:_ROOM_ENTRY_METADATA_MAX_LEN]


def serialize_room_select_request(request: RoomSelectRequestDict) -> bytes:
    """Serialize a room-selection request to a framed text packet.

    Args:
        request: Typed room-selection request.

    Returns:
        Framed websocket payload.
    """
    body = f"*{request['room_id']}".encode()
    return encode_frame(body)


def serialize_room_enter_request(
    request: RoomEnterRequestDict,
    codec: ProtocolCodec,
) -> bytes:
    """Serialize a room-entry request to a framed text packet.

    Args:
        request: Typed room-entry request.
        codec: Session codec used to XOR-encode metadata bytes.

    Returns:
        Framed websocket payload.
    """
    metadata_bytes = codec.encode(request["metadata"].encode("utf-8"))
    body = (
        b"+"
        + request["room_id"].encode("utf-8")
        + b"|"
        + str(request["troop"]).encode("utf-8")
        + b"|"
        + str(request["preview_x"]).encode("utf-8")
        + b"|"
        + str(request["preview_y"]).encode("utf-8")
        + b"|"
        + metadata_bytes
    )
    return encode_frame(body)


__all__ = [
    "ROOM_ENTRY_DEFAULT_X",
    "ROOM_ENTRY_DEFAULT_Y",
    "ROOM_ENTRY_METADATA_SUFFIX",
    "RoomEnterRequestDict",
    "RoomSelectRequestDict",
    "build_room_enter_metadata",
    "decode_room_enter_request",
    "decode_room_select_request",
    "encode_room_enter_request",
    "encode_room_select_request",
    "serialize_room_enter_request",
    "serialize_room_select_request",
]
