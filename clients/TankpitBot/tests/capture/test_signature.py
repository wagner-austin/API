"""Tests for tankpit_bot.capture.signature module."""

from __future__ import annotations

from tankpit_bot.capture.signature import identify_message


class TestIdentifyMessage:
    """Tests for identify_message function."""

    def test_unknown_message_type_returns_none(self) -> None:
        """Test returns None for unknown message type."""
        # 8 bytes doesn't match any known structure
        # (not 1, 2-3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16-20, 29+, 80+, 500+)
        data = bytes([0xFF] * 8)
        result = identify_message(data)
        assert result is None

    def test_player_list_short_4_bytes(self) -> None:
        """4-byte message with 0x79 subtype identifies as player_list_short.

        Subtype guard was added to is_player_list_short_structure to
        prevent silent absorption of unrelated 4-byte container subtypes
        (same bug class as the 0x41 deactivation_kill regression).
        """
        data = bytes([0x79, 0x02, 0x03, 0x04])
        result = identify_message(data)
        assert result == ("player_list_short", 100)  # DecodeLevel.FULL = 100

    def test_player_list_short_rejects_non_0x79_subtype(self) -> None:
        """4-byte message with wrong subtype must not identify as player_list_short."""
        data = bytes([0x01, 0x02, 0x03, 0x04])
        result = identify_message(data)
        assert result is None

    def test_position_update_13_bytes(self) -> None:
        """Test identifies 13-byte message as position_update."""
        data = bytes([0x24] + [0x01] * 12)
        result = identify_message(data)
        assert result == ("position_update", 100)  # DecodeLevel.FULL = 100

    def test_non_position_13_bytes_returns_none(self) -> None:
        """Test non-position 13-byte payload is not mislabeled as position."""
        data = bytes([0x01] * 13)
        result = identify_message(data)
        assert result is None
