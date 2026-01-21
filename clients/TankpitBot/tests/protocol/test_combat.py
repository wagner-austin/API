"""Tests for combat message decoders.

Tests for shoot, hit confirmation, deactivation, and mine decoders.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import (
    DecodeError,
    decode_deactivation,
    decode_hit_confirmation,
    decode_mine_detonation,
    decode_mine_placement,
    decode_shoot_event,
    x24,
)


class TestDecodeShootEvent:
    """Tests for decode_shoot_event function."""

    def test_decodes_valid_shoot_event(self) -> None:
        """Decodes valid shooting event."""
        # shooter_id=0x0102, target=(10,20), proj=(15,25), fuel=0x030405, weapon=1, ammo=5, ff=0
        data = bytes([0x02, 0x01, 10, 20, 15, 25, 0x03, 0x04, 0x05, 1, 5, 0])
        result = decode_shoot_event(data)
        assert result["msg_type"] == 0x53
        assert result["shooter_id"] == 0x0102
        assert result["target_x"] == 10
        assert result["target_y"] == 20
        assert result["projectile_x"] == 15
        assert result["projectile_y"] == 25
        assert result["fuel"] == x24(0x03, 0x04, 0x05)
        assert result["weapon"] == 1
        assert result["ammo"] == 5
        assert result["friendly_fire"] is False

    def test_decodes_friendly_fire(self) -> None:
        """Decodes friendly fire flag correctly."""
        data = bytes([0x02, 0x01, 10, 20, 15, 25, 0x03, 0x04, 0x05, 1, 5, 1])
        result = decode_shoot_event(data)
        assert result["friendly_fire"] is True

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_shoot_event(bytes([1, 2, 3, 4, 5]))


class TestDecodeHitConfirmation:
    """Tests for decode_hit_confirmation function."""

    def test_decodes_valid_hit_confirmation(self) -> None:
        """Decodes valid hit confirmation."""
        # 12 bytes starting with 0x2E
        # After XOR decode: decoded[5]=target_y, decoded[6]=target_x
        # data[6] -> decoded[5], data[7] -> decoded[6]
        data = bytes([0x2E, 0x01, 0x02, 0x03, 0x04, 0x05, 0x35, 0x50, 0x08, 0x09, 0x0A, 0x0B])
        xor_table = bytes([0x00] * 11)  # No-op XOR
        result = decode_hit_confirmation(data, xor_table)
        assert result["msg_type"] == 0x2E
        assert result["target_y"] == 0x35  # decoded[5] = data[6]
        assert result["target_x"] == 0x50  # decoded[6] = data[7]

    def test_raises_on_wrong_length(self) -> None:
        """Raises DecodeError on wrong length."""
        data = bytes([0x2E, 0x01, 0x02])
        xor_table = bytes([0x00] * 3)
        with pytest.raises(DecodeError):
            decode_hit_confirmation(data, xor_table)

    def test_raises_on_wrong_prefix(self) -> None:
        """Raises DecodeError on wrong prefix."""
        data = bytes([0x3E] + [0x00] * 11)
        xor_table = bytes([0x00] * 11)
        with pytest.raises(DecodeError) as exc:
            decode_hit_confirmation(data, xor_table)
        assert "expected 0x2E prefix" in str(exc.value)


class TestDecodeDeactivation:
    """Tests for decode_deactivation function."""

    def test_decodes_with_points(self) -> None:
        """Decodes deactivation with points field."""
        # victim=0x0102, killer=0x0304, rank=5, points=0x0607
        data = bytes([0x02, 0x01, 0x04, 0x03, 5, 0x07, 0x06])
        result = decode_deactivation(data)
        assert result["msg_type"] == 0x41
        assert result["victim_id"] == 0x0102
        assert result["killer_id"] == 0x0304
        assert result["rank"] == 5
        assert result["points"] == 0x0607

    def test_decodes_without_points(self) -> None:
        """Decodes deactivation without points field."""
        data = bytes([0x02, 0x01, 0x04, 0x03, 5])
        result = decode_deactivation(data)
        assert result["points"] == 0

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_deactivation(bytes([1, 2, 3]))


class TestDecodeMinePlacement:
    """Tests for decode_mine_placement function."""

    def test_decodes_mine_placement(self) -> None:
        """Decodes mine placement message."""
        # type=1, tank_id=0x0102, count=2, positions=[(10,20), (30,40)]
        data = bytes([1, 0x02, 0x01, 2, 10, 20, 30, 40])
        result = decode_mine_placement(data)
        assert result["msg_type"] == 0x4B
        assert result["mine_type"] == 1
        assert result["tank_id"] == 0x0102
        assert result["positions"] == [(10, 20), (30, 40)]

    def test_handles_truncated_positions(self) -> None:
        """Handles truncated position data."""
        data = bytes([1, 0x02, 0x01, 3, 10, 20])  # Claims 3 positions but only has 1
        result = decode_mine_placement(data)
        assert result["positions"] == [(10, 20)]

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_mine_placement(bytes([1, 2]))


class TestDecodeMineDetonation:
    """Tests for decode_mine_detonation function."""

    def test_decodes_mine_detonation(self) -> None:
        """Decodes mine detonation message."""
        data = bytes([10, 20, 30, 40])  # Two positions
        result = decode_mine_detonation(data)
        assert result["msg_type"] == 0x45
        assert result["positions"] == [(10, 20), (30, 40)]

    def test_handles_empty_data(self) -> None:
        """Handles empty position data."""
        result = decode_mine_detonation(b"")
        assert result["positions"] == []
