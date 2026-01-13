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
        """Test identifies 4-byte message as player_list_short."""
        # 4 bytes matches player_list_short structure
        data = bytes([0x01, 0x02, 0x03, 0x04])
        result = identify_message(data)
        assert result == ("player_list_short", 100)  # DecodeLevel.FULL = 100

    def test_position_update_13_bytes(self) -> None:
        """Test identifies 13-byte message as position_update."""
        # 13 bytes matches position_update structure
        data = bytes([0x01] * 13)
        result = identify_message(data)
        assert result == ("position_update", 100)  # DecodeLevel.FULL = 100
