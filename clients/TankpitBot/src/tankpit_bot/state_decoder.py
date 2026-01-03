"""Decode binary state messages from the Tankpit game protocol.

State messages start with '.' (0x2e) followed by a subtype byte.
Different subtypes encode different game state data.

Fuel Encoding (discovered and verified):
- 14-byte state messages contain fuel as u16 little-endian at bytes 12-13
- XOR encoded: decoded = (body[12] ^ xor_table[12]) | ((body[13] ^ xor_table[13]) << 8)
- Entity ID in bytes 2-6 identifies the tank/container
- Subtype byte varies per session (also XOR encoded)

Verified Fuel Costs:
- Radar (S key): -10 fuel
- Movement: -1 fuel per tile
- Fuel deposit: -100 fuel
- Fuel pickup: +100 fuel

Known subtypes (varies per session due to XOR encoding):
- 0x03, 0x06, 0x13, 0x15, 0x40: Tank/entity state (14 bytes)
- 0x1c: Entity/state updates (17-20 bytes)
- 0x7b: Short sync (4 bytes)
- Large messages (500+ bytes): Map data

See docs/fuel_encoding.md for full research details.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_int,
    require_str,
)


class TankStatus(TypedDict):
    """Tank status from subtype 0x14 message.

    Attributes:
        fuel: Current fuel level (0-65535).
        subtype_byte1: First identifier byte after subtype.
        subtype_byte2: Second identifier byte after subtype.
    """

    fuel: int
    subtype_byte1: int
    subtype_byte2: int


class StateMessage(TypedDict):
    """Generic decoded state message.

    Attributes:
        timestamp_ms: When the message was captured.
        direction: Message direction (sent/received).
        subtype: State message subtype byte.
        body_hex: Raw body bytes as hex string.
        length: Body length in bytes.
    """

    timestamp_ms: int
    direction: Literal["sent", "received"]
    subtype: int
    body_hex: str
    length: int


class DecodedStateMessage(TypedDict):
    """State message with decoded payload.

    Attributes:
        timestamp_ms: When the message was captured.
        direction: Message direction (sent/received).
        subtype: State message subtype byte.
        subtype_name: Human-readable subtype name.
        body_hex: Raw body bytes as hex string.
        decoded: Decoded payload (type depends on subtype).
    """

    timestamp_ms: int
    direction: Literal["sent", "received"]
    subtype: int
    subtype_name: str
    body_hex: str
    decoded: TankStatus | None


# Subtype names for documentation
SUBTYPE_NAMES: dict[int, str] = {
    0x00: "HEARTBEAT",
    0x05: "SYNC",
    0x07: "TANK_OTHER",
    0x12: "ENTITY_REF",
    0x14: "TANK_STATUS",
    0x1B: "ENTITY",
    0x4E: "TOGGLE",
    0x5D: "POSITION_REF",
    0x60: "ENTITY_UPDATE",
    0x6E: "SYNC",
    0x73: "MAP_DATA",
    0x75: "SHORT_REF",
    0x79: "POSITION",
    0x7C: "SYNC_REF",
    0x7D: "UPDATE",
}


def decode_tank_status(body: bytes) -> TankStatus | None:
    """Decode a tank status message (subtype 0x14).

    Args:
        body: Raw message body starting with 0x2e.

    Returns:
        TankStatus if valid, None if body too short.
    """
    # Need at least 13 bytes: 0x2e + subtype + b1 + b2 + ... + fuel(2)
    if len(body) < 13:
        return None

    # Fuel is stored at bytes 11-12 as little-endian
    fuel = int.from_bytes(body[11:13], "little")

    return TankStatus(
        fuel=fuel,
        subtype_byte1=body[1],
        subtype_byte2=body[2],
    )


def decode_state_message(
    timestamp_ms: int,
    direction: Literal["sent", "received"],
    body: bytes,
) -> DecodedStateMessage | None:
    """Decode a state message from raw bytes.

    Args:
        timestamp_ms: Message timestamp.
        direction: Message direction.
        body: Raw message body starting with 0x2e.

    Returns:
        DecodedStateMessage if valid state message, None otherwise.
    """
    if len(body) < 2:
        return None

    if body[0] != 0x2E:
        return None

    subtype = body[1]
    subtype_name = SUBTYPE_NAMES.get(subtype, f"UNKNOWN_0x{subtype:02X}")

    decoded: TankStatus | None = None

    # Decode based on subtype
    if subtype == 0x14 and 13 <= len(body) <= 16:
        decoded = decode_tank_status(body)

    return DecodedStateMessage(
        timestamp_ms=timestamp_ms,
        direction=direction,
        subtype=subtype,
        subtype_name=subtype_name,
        body_hex=body.hex(),
        decoded=decoded,
    )


def is_state_message(body: bytes) -> bool:
    """Check if body is a state message.

    Args:
        body: Raw message body.

    Returns:
        True if starts with 0x2e (state message prefix).
    """
    return len(body) >= 2 and body[0] == 0x2E


def get_subtype_name(subtype: int) -> str:
    """Get human-readable name for a subtype.

    Args:
        subtype: State message subtype byte.

    Returns:
        Subtype name or 'UNKNOWN_0xNN'.
    """
    return SUBTYPE_NAMES.get(subtype, f"UNKNOWN_0x{subtype:02X}")


def encode_tank_status(status: TankStatus) -> JSONObject:
    """Encode TankStatus to JSON-serializable dict.

    Args:
        status: TankStatus to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "fuel": status["fuel"],
        "subtype_byte1": status["subtype_byte1"],
        "subtype_byte2": status["subtype_byte2"],
    }


def decode_tank_status_json(data: JSONObject) -> TankStatus:
    """Decode TankStatus from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated TankStatus.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return TankStatus(
        fuel=require_int(data, "fuel"),
        subtype_byte1=require_int(data, "subtype_byte1"),
        subtype_byte2=require_int(data, "subtype_byte2"),
    )


def encode_state_message(msg: StateMessage) -> JSONObject:
    """Encode StateMessage to JSON-serializable dict.

    Args:
        msg: StateMessage to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "timestamp_ms": msg["timestamp_ms"],
        "direction": msg["direction"],
        "subtype": msg["subtype"],
        "body_hex": msg["body_hex"],
        "length": msg["length"],
    }


def decode_state_message_json(data: JSONObject) -> StateMessage:
    """Decode StateMessage from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated StateMessage.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    direction = require_str(data, "direction")
    if direction not in ("sent", "received"):
        raise JSONTypeError(f"Invalid direction: {direction}")

    direction_literal: Literal["sent", "received"] = "sent" if direction == "sent" else "received"

    return StateMessage(
        timestamp_ms=require_int(data, "timestamp_ms"),
        direction=direction_literal,
        subtype=require_int(data, "subtype"),
        body_hex=require_str(data, "body_hex"),
        length=require_int(data, "length"),
    )


def encode_decoded_state_message(msg: DecodedStateMessage) -> JSONObject:
    """Encode DecodedStateMessage to JSON-serializable dict.

    Args:
        msg: DecodedStateMessage to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "timestamp_ms": msg["timestamp_ms"],
        "direction": msg["direction"],
        "subtype": msg["subtype"],
        "subtype_name": msg["subtype_name"],
        "body_hex": msg["body_hex"],
    }

    if msg["decoded"] is not None:
        result["decoded"] = encode_tank_status(msg["decoded"])
    else:
        result["decoded"] = None

    return result


def decode_decoded_state_message_json(data: JSONObject) -> DecodedStateMessage:
    """Decode DecodedStateMessage from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated DecodedStateMessage.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    direction = require_str(data, "direction")
    if direction not in ("sent", "received"):
        raise JSONTypeError(f"Invalid direction: {direction}")

    direction_literal: Literal["sent", "received"] = "sent" if direction == "sent" else "received"

    decoded_raw = data.get("decoded")
    decoded: TankStatus | None = None
    if decoded_raw is not None and isinstance(decoded_raw, dict):
        decoded = decode_tank_status_json(decoded_raw)

    return DecodedStateMessage(
        timestamp_ms=require_int(data, "timestamp_ms"),
        direction=direction_literal,
        subtype=require_int(data, "subtype"),
        subtype_name=require_str(data, "subtype_name"),
        body_hex=require_str(data, "body_hex"),
        decoded=decoded,
    )


__all__ = [
    "SUBTYPE_NAMES",
    "DecodedStateMessage",
    "StateMessage",
    "TankStatus",
    "decode_decoded_state_message_json",
    "decode_state_message",
    "decode_state_message_json",
    "decode_tank_status",
    "decode_tank_status_json",
    "encode_decoded_state_message",
    "encode_state_message",
    "encode_tank_status",
    "get_subtype_name",
    "is_state_message",
]
