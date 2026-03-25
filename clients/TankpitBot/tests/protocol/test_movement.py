"""Tests for movement message decoders.

Tests for movement and movement response decoders.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import (
    DecodeError,
    decode_movement,
    decode_movement_response,
    x24,
)


class TestDecodeMovement:
    """Tests for decode_movement function."""

    def test_decodes_movement(self) -> None:
        """Decodes movement message."""
        # tank_id=0x0102, start=(50, 60), dir=3, flag=1, fuel=0x030405
        data = bytes([0x02, 0x01, 50, 60, 3, 1, 0x03, 0x04, 0x05])
        result = decode_movement(data)
        assert result["msg_type"] == 0x47
        assert result["tank_id"] == 0x0102
        assert result["start_x"] == 50
        assert result["start_y"] == 60
        assert result["direction"] == 3
        assert result["flag"] == 1
        assert result["leaderboard_position"] == x24(0x03, 0x04, 0x05)
        assert result["waypoints"] == []

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_movement(bytes([1, 2, 3, 4]))


class TestDecodeMovementResponse:
    """Tests for decode_movement_response function."""

    def test_decodes_movement_response(self) -> None:
        """Decodes movement response message."""
        # team=1, tank_id=0x0102, x=50, y=60, dir=3, skip 1, rank=4, lb_pos=0x050607
        data = bytes([1, 0x02, 0x01, 50, 60, 3, 0x00, 4, 0x05, 0x06, 0x07])
        result = decode_movement_response(data)
        assert result["msg_type"] == 0x3D
        assert result["team"] == 1
        assert result["tank_id"] == 0x0102
        assert result["x"] == 50
        assert result["y"] == 60
        assert result["direction"] == 3
        assert result["rank"] == 4
        assert result["leaderboard_position"] == x24(0x05, 0x06, 0x07)

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_movement_response(bytes([1, 2, 3, 4, 5]))
