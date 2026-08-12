"""Tests for tankpit_bot.capture.signature module."""

from __future__ import annotations

import base64

from tankpit_bot.capture.signature import extract_message_signature, identify_message

_XOR_TABLE = bytes(range(1, 9))


class TestExtractMessageSignature:
    """Signature extraction from a raw capture payload."""

    def test_payload_that_is_not_base64_returns_none(self) -> None:
        """A malformed payload is refused before any decode is attempted.

        ``"abc"`` matches the base64 alphabet but its length is not a
        multiple of four, so :func:`base64.b64decode` raises
        ``binascii.Error`` on it. The validity check exists to keep that
        exception out of the capture path, where payloads arrive
        unvalidated from the wire.
        """
        assert extract_message_signature("abc", _XOR_TABLE) is None

    def test_empty_payload_returns_none(self) -> None:
        """An empty payload carries no signature."""
        assert extract_message_signature("", _XOR_TABLE) is None

    def test_payload_without_a_leading_dot_returns_none(self) -> None:
        """The dot delimiter must appear in the first three bytes."""
        payload = base64.b64encode(b"abcd").decode("ascii")

        assert extract_message_signature(payload, _XOR_TABLE) is None

    def test_payload_with_a_dot_decodes_the_remainder(self) -> None:
        """Bytes after the dot are XOR-decoded against the table."""
        payload = base64.b64encode(b"A." + bytes([0x10, 0x20, 0x30])).decode("ascii")

        result = extract_message_signature(payload, _XOR_TABLE)

        assert result == bytes([0x10 ^ 1, 0x20 ^ 2, 0x30 ^ 3])


class TestIdentifyMessage:
    """Tests for identify_message function."""

    def test_unknown_message_type_returns_none(self) -> None:
        """Returns None for unknown structures.

        After the 2026-06-20 container deletion sweep, 4/6/7/13/16-20 byte
        bodies no longer identify as PlayerList/TankLeave/DeactivationDeath
        /PositionUpdate/TankRegistry respectively.
        """
        data = bytes([0xFF] * 8)
        result = identify_message(data)
        assert result is None

    def test_teleport_landed_1_byte(self) -> None:
        """1-byte body identifies as teleport_landed."""
        data = bytes([0x0C])
        result = identify_message(data)
        assert result == ("teleport_landed", 100)  # DecodeLevel.FULL = 100

    def test_container_pickup_5_bytes_with_subtype(self) -> None:
        """5-byte body with 0x43 subtype identifies as container_pickup."""
        data = bytes([0x43, 0x88, 0x5E, 0x00, 0x00])
        result = identify_message(data)
        assert result == ("container_pickup", 100)

    def test_short_body_returns_none(self) -> None:
        """Short bodies that no longer match any container subtype are unknown."""
        data = bytes([0x79, 0x02, 0x03, 0x04])
        result = identify_message(data)
        assert result is None

    def test_position_13_bytes_returns_none(self) -> None:
        """13-byte 0x24-subtype bodies no longer match PositionUpdate (deleted)."""
        data = bytes([0x24] + [0x01] * 12)
        result = identify_message(data)
        assert result is None
