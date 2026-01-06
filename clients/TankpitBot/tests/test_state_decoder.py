"""Tests for state_decoder module.

Strict typing: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Literal

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.state_decoder import (
    SUBTYPE_NAMES,
    DecodedStateMessage,
    StateMessage,
    TankStatus,
    decode_decoded_state_message_json,
    decode_state_message,
    decode_state_message_json,
    decode_tank_status,
    decode_tank_status_json,
    encode_decoded_state_message,
    encode_state_message,
    encode_tank_status,
    get_subtype_name,
    is_state_message,
)

# =============================================================================
# Helper Functions
# =============================================================================


def require_tank_status(body: bytes) -> TankStatus:
    """Decode tank status, raising if None.

    Args:
        body: Raw message body.

    Returns:
        Valid TankStatus.

    Raises:
        ValueError: If decode returns None.
    """
    result = decode_tank_status(body)
    if result is None:
        raise ValueError("Expected valid TankStatus")
    return result


def require_decoded_state(
    timestamp_ms: int,
    direction: str,
    body: bytes,
) -> DecodedStateMessage:
    """Decode state message, raising if None.

    Args:
        timestamp_ms: Message timestamp.
        direction: Message direction.
        body: Raw message body.

    Returns:
        Valid DecodedStateMessage.

    Raises:
        ValueError: If decode returns None.
    """
    dir_literal: Literal["sent", "received"] = "sent" if direction == "sent" else "received"
    result = decode_state_message(timestamp_ms, dir_literal, body)
    if result is None:
        raise ValueError("Expected valid DecodedStateMessage")
    return result


# =============================================================================
# TankStatus Tests
# =============================================================================


def test_decode_tank_status_valid() -> None:
    """Decode valid tank status message."""
    # Real captured message: fuel=28988
    body = bytes.fromhex("2e14584114351c310fcf063c7141")

    result = require_tank_status(body)

    # Roundtrip to verify
    encoded = encode_tank_status(result)
    roundtrip = decode_tank_status_json(encoded)
    assert roundtrip["fuel"] == 28988
    assert roundtrip["subtype_byte1"] == 0x14
    assert roundtrip["subtype_byte2"] == 0x58


def test_decode_tank_status_too_short() -> None:
    """Return None for too short message."""
    body = bytes.fromhex("2e14584114351c31")  # Only 8 bytes

    result = decode_tank_status(body)

    assert result is None


def test_decode_tank_status_13_bytes() -> None:
    """Decode 13-byte message (minimum valid length)."""
    # Construct 13-byte message with fuel=1000 at bytes 11-12
    # 1000 = 0x03E8 little-endian = E8 03
    body = bytes.fromhex("2e14584114351c310fcf06e803")

    result = require_tank_status(body)

    encoded = encode_tank_status(result)
    roundtrip = decode_tank_status_json(encoded)
    assert roundtrip["fuel"] == 1000


def test_encode_tank_status() -> None:
    """Encode TankStatus to JSON."""
    status: TankStatus = TankStatus(fuel=15420, subtype_byte1=20, subtype_byte2=88)

    result = encode_tank_status(status)

    assert result["fuel"] == 15420
    assert result["subtype_byte1"] == 20
    assert result["subtype_byte2"] == 88


def test_decode_tank_status_json() -> None:
    """Decode TankStatus from JSON."""
    data: JSONObject = {"fuel": 28988, "subtype_byte1": 20, "subtype_byte2": 88}

    result = decode_tank_status_json(data)

    assert result["fuel"] == 28988
    assert result["subtype_byte1"] == 20
    assert result["subtype_byte2"] == 88


def test_decode_tank_status_json_missing_field() -> None:
    """Raise error for missing field."""
    data: JSONObject = {"fuel": 28988, "subtype_byte1": 20}

    with pytest.raises(JSONTypeError):
        decode_tank_status_json(data)


def test_tank_status_roundtrip() -> None:
    """Encode then decode produces same values."""
    original: TankStatus = TankStatus(fuel=50000, subtype_byte1=7, subtype_byte2=89)

    encoded = encode_tank_status(original)
    decoded = decode_tank_status_json(encoded)

    assert decoded["fuel"] == original["fuel"]
    assert decoded["subtype_byte1"] == original["subtype_byte1"]
    assert decoded["subtype_byte2"] == original["subtype_byte2"]


# =============================================================================
# StateMessage Tests
# =============================================================================


def test_encode_state_message() -> None:
    """Encode StateMessage to JSON."""
    msg: StateMessage = StateMessage(
        timestamp_ms=12345678,
        direction="received",
        subtype=0x14,
        body_hex="2e14584114351c310fcf063c7141",
        length=14,
    )

    result = encode_state_message(msg)

    assert result["timestamp_ms"] == 12345678
    assert result["direction"] == "received"
    assert result["subtype"] == 0x14
    assert result["body_hex"] == "2e14584114351c310fcf063c7141"
    assert result["length"] == 14


def test_decode_state_message_json() -> None:
    """Decode StateMessage from JSON."""
    data: JSONObject = {
        "timestamp_ms": 12345678,
        "direction": "sent",
        "subtype": 0x14,
        "body_hex": "2e14abcd",
        "length": 4,
    }

    result = decode_state_message_json(data)

    assert result["timestamp_ms"] == 12345678
    assert result["direction"] == "sent"
    assert result["subtype"] == 0x14


def test_decode_state_message_json_invalid_direction() -> None:
    """Raise error for invalid direction."""
    data: JSONObject = {
        "timestamp_ms": 12345678,
        "direction": "invalid",
        "subtype": 0x14,
        "body_hex": "2e14abcd",
        "length": 4,
    }

    with pytest.raises(JSONTypeError, match="Invalid direction"):
        decode_state_message_json(data)


def test_state_message_roundtrip() -> None:
    """Encode then decode produces same values."""
    original: StateMessage = StateMessage(
        timestamp_ms=99999,
        direction="received",
        subtype=0x7D,
        body_hex="2e7d0102030405",
        length=7,
    )

    encoded = encode_state_message(original)
    decoded = decode_state_message_json(encoded)

    assert decoded["timestamp_ms"] == original["timestamp_ms"]
    assert decoded["direction"] == original["direction"]
    assert decoded["subtype"] == original["subtype"]


# =============================================================================
# DecodedStateMessage Tests
# =============================================================================


def _require_decoded_payload(msg: DecodedStateMessage) -> TankStatus:
    """Extract decoded payload, raising if None."""
    decoded = msg["decoded"]
    if decoded is None:
        raise ValueError("Expected decoded payload")
    return decoded


def test_decode_state_message_tank_status() -> None:
    """Decode tank status message."""
    body = bytes.fromhex("2e14584114351c310fcf063c7141")

    result = require_decoded_state(12345, "received", body)

    assert result["subtype"] == 0x14
    assert result["subtype_name"] == "TANK_STATUS"
    # Verify decoded payload by encoding and decoding
    decoded_payload = _require_decoded_payload(result)
    encoded = encode_tank_status(decoded_payload)
    roundtrip = decode_tank_status_json(encoded)
    assert roundtrip["fuel"] == 28988


def test_decode_state_message_unknown_subtype() -> None:
    """Decode message with unknown subtype."""
    body = bytes.fromhex("2eFF010203040506070809")

    result = require_decoded_state(12345, "received", body)

    assert result["subtype"] == 0xFF
    assert result["subtype_name"] == "UNKNOWN_0xFF"
    assert result["decoded"] is None


def test_decode_state_message_heartbeat() -> None:
    """Decode heartbeat message."""
    body = bytes.fromhex("2e0000")

    result = require_decoded_state(12345, "received", body)

    assert result["subtype"] == 0x00
    assert result["subtype_name"] == "HEARTBEAT"


def test_decode_state_message_not_state() -> None:
    """Return None for non-state message."""
    body = bytes.fromhex("21020304")  # Starts with '!' not '.'

    result = decode_state_message(12345, "received", body)

    assert result is None


def test_decode_state_message_too_short() -> None:
    """Return None for too short message."""
    body = bytes.fromhex("2e")  # Only 1 byte

    result = decode_state_message(12345, "received", body)

    assert result is None


def test_decode_state_message_empty() -> None:
    """Return None for empty message."""
    body = b""

    result = decode_state_message(12345, "received", body)

    assert result is None


def test_encode_decoded_state_message_with_decoded() -> None:
    """Encode DecodedStateMessage with decoded payload."""
    msg: DecodedStateMessage = DecodedStateMessage(
        timestamp_ms=12345,
        direction="received",
        subtype=0x14,
        subtype_name="TANK_STATUS",
        body_hex="2e14584114351c310fcf063c7141",
        decoded=TankStatus(fuel=28988, subtype_byte1=20, subtype_byte2=88),
    )

    result = encode_decoded_state_message(msg)

    assert result["timestamp_ms"] == 12345
    assert result["subtype_name"] == "TANK_STATUS"
    # Verify by decoding the result
    decoded_msg = decode_decoded_state_message_json(result)
    decoded_payload = _require_decoded_payload(decoded_msg)
    assert decoded_payload["fuel"] == 28988


def test_encode_decoded_state_message_without_decoded() -> None:
    """Encode DecodedStateMessage without decoded payload."""
    msg: DecodedStateMessage = DecodedStateMessage(
        timestamp_ms=12345,
        direction="received",
        subtype=0xFF,
        subtype_name="UNKNOWN_0xFF",
        body_hex="2eff0102",
        decoded=None,
    )

    result = encode_decoded_state_message(msg)

    assert result["decoded"] is None


def test_decode_decoded_state_message_json_with_decoded() -> None:
    """Decode DecodedStateMessage from JSON with decoded payload."""
    data: JSONObject = {
        "timestamp_ms": 12345,
        "direction": "received",
        "subtype": 0x14,
        "subtype_name": "TANK_STATUS",
        "body_hex": "2e14abcd",
        "decoded": {"fuel": 50000, "subtype_byte1": 7, "subtype_byte2": 89},
    }

    result = decode_decoded_state_message_json(data)

    assert result["timestamp_ms"] == 12345
    assert result["subtype_name"] == "TANK_STATUS"
    decoded_payload = _require_decoded_payload(result)
    assert decoded_payload["fuel"] == 50000


def test_decode_decoded_state_message_json_without_decoded() -> None:
    """Decode DecodedStateMessage from JSON without decoded payload."""
    data: JSONObject = {
        "timestamp_ms": 12345,
        "direction": "received",
        "subtype": 0xFF,
        "subtype_name": "UNKNOWN_0xFF",
        "body_hex": "2eff",
        "decoded": None,
    }

    result = decode_decoded_state_message_json(data)

    assert result["decoded"] is None


def test_decode_decoded_state_message_json_invalid_direction() -> None:
    """Raise error for invalid direction."""
    data: JSONObject = {
        "timestamp_ms": 12345,
        "direction": "bad",
        "subtype": 0x14,
        "subtype_name": "TANK_STATUS",
        "body_hex": "2e14",
        "decoded": None,
    }

    with pytest.raises(JSONTypeError, match="Invalid direction"):
        decode_decoded_state_message_json(data)


def test_decoded_state_message_roundtrip() -> None:
    """Encode then decode produces same values."""
    original: DecodedStateMessage = DecodedStateMessage(
        timestamp_ms=99999,
        direction="sent",
        subtype=0x14,
        subtype_name="TANK_STATUS",
        body_hex="2e14aabbccdd",
        decoded=TankStatus(fuel=12345, subtype_byte1=14, subtype_byte2=20),
    )

    encoded = encode_decoded_state_message(original)
    decoded = decode_decoded_state_message_json(encoded)

    assert decoded["timestamp_ms"] == original["timestamp_ms"]
    assert decoded["direction"] == original["direction"]
    assert decoded["subtype"] == original["subtype"]
    decoded_payload = _require_decoded_payload(decoded)
    assert decoded_payload["fuel"] == 12345


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_is_state_message_true() -> None:
    """Return True for valid state message."""
    body = bytes.fromhex("2e140102")

    assert is_state_message(body) is True


def test_is_state_message_false_wrong_prefix() -> None:
    """Return False for wrong prefix."""
    body = bytes.fromhex("210102")  # '!' prefix

    assert is_state_message(body) is False


def test_is_state_message_false_too_short() -> None:
    """Return False for too short message."""
    body = bytes.fromhex("2e")  # Only 1 byte

    assert is_state_message(body) is False


def test_is_state_message_false_empty() -> None:
    """Return False for empty message."""
    body = b""

    assert is_state_message(body) is False


def test_get_subtype_name_known() -> None:
    """Return name for known subtype."""
    assert get_subtype_name(0x14) == "TANK_STATUS"
    assert get_subtype_name(0x7D) == "UPDATE"
    assert get_subtype_name(0x00) == "HEARTBEAT"


def test_get_subtype_name_unknown() -> None:
    """Return hex string for unknown subtype."""
    assert get_subtype_name(0xAB) == "UNKNOWN_0xAB"
    assert get_subtype_name(0xFF) == "UNKNOWN_0xFF"


def test_subtype_names_contains_key_values() -> None:
    """SUBTYPE_NAMES dictionary has expected entries."""
    assert 0x14 in SUBTYPE_NAMES
    assert SUBTYPE_NAMES[0x14] == "TANK_STATUS"
    assert 0x7D in SUBTYPE_NAMES
    assert SUBTYPE_NAMES[0x7D] == "UPDATE"


# =============================================================================
# Real Captured Data Tests
# =============================================================================


def test_decode_real_tank_message_fuel_28988() -> None:
    """Decode real captured tank message with fuel 28988."""
    body = bytes.fromhex("2e14584114351c310fcf063c7141")

    result = require_tank_status(body)

    encoded = encode_tank_status(result)
    roundtrip = decode_tank_status_json(encoded)
    assert roundtrip["fuel"] == 28988


def test_decode_real_tank_message_fuel_54278() -> None:
    """Decode real captured tank message with fuel 54278."""
    body = bytes.fromhex("2e07584114a1683936150606d443")

    result = require_tank_status(body)

    # bytes[11:13] = d4 06 -> little-endian = 0x06d4? No wait, let's check
    # hex: 2e 07 58 41 14 a1 68 39 36 15 06 06 d4 43
    # bytes 11-12 = 06 d4 -> little-endian = 0xd406 = 54278. Correct!
    encoded = encode_tank_status(result)
    roundtrip = decode_tank_status_json(encoded)
    assert roundtrip["fuel"] == 54278


def test_decode_heartbeat_message() -> None:
    """Decode heartbeat message."""
    body = bytes.fromhex("2e0000")

    result = require_decoded_state(12345, "received", body)

    assert result["subtype"] == 0x00
    assert result["subtype_name"] == "HEARTBEAT"
    assert result["body_hex"] == "2e0000"


def test_decode_sync_message() -> None:
    """Decode sync message."""
    body = bytes.fromhex("2e055a")

    result = require_decoded_state(12345, "received", body)

    assert result["subtype"] == 0x05
    assert result["subtype_name"] == "SYNC"
    assert result["body_hex"] == "2e055a"
