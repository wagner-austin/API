"""Tests for tankpit_bot.capture.signature module."""

from __future__ import annotations

from tankpit_bot.capture.signature import identify_message


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
