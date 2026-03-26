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

    def test_decodes_movement_with_waypoints(self) -> None:
        """Decodes movement message with nsew waypoint characters at bytes 11+.

        Path: eesn → east, east, south, north
        Start: (50, 60) → Final: (52, 60)
        """
        header = bytes([0x02, 0x01, 50, 60, 3, 1, 0x03, 0x04, 0x05])
        padding = bytes([0x00, 0x00])  # bytes 9-10 (before waypoints)
        waypoints = b"eesn"  # bytes 11+ (ASCII direction chars)
        data = header + padding + waypoints
        result = decode_movement(data)
        assert result["msg_type"] == 0x47
        assert result["start_x"] == 50
        assert result["start_y"] == 60
        # Final position: e→51, e→52, s→61, n→60 → (52, 60)
        assert result["waypoints"] == [(52, 60)]

    def test_decodes_movement_with_mixed_data_and_waypoints(self) -> None:
        """Decodes movement ignoring non-nsew bytes in waypoint region."""
        header = bytes([0x02, 0x01, 100, 80, 3, 1, 0x03, 0x04, 0x05])
        padding = bytes([0x00, 0x00])
        # Mix of valid waypoint chars and non-direction bytes
        waypoints = bytes([ord("n"), 0xFF, ord("n"), ord("w"), ord("e")])
        data = header + padding + waypoints
        result = decode_movement(data)
        # nn → y-=2, w → x-=1, e → x+=1 → (100, 78)
        assert result["waypoints"] == [(100, 78)]

    def test_decodes_movement_empty_waypoints_region(self) -> None:
        """Decodes movement with data at bytes 11+ but no nsew chars."""
        header = bytes([0x02, 0x01, 50, 60, 3, 1, 0x03, 0x04, 0x05])
        padding = bytes([0x00, 0x00])
        non_direction = bytes([0xFF, 0x00, 0x42])
        data = header + padding + non_direction
        result = decode_movement(data)
        # No valid waypoint chars → empty waypoints
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
