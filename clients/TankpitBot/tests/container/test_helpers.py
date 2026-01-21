"""Tests for container decoder helper functions.

Tests for validation helpers and extraction functions.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container import (
    ContainerDecodeError,
    extract_uint16_le,
    require_exact_length,
    require_length_range,
    require_min_length,
)


class TestRequireMinLength:
    """Tests for require_min_length validation."""

    def test_passes_when_length_sufficient(self) -> None:
        """Validation passes when data meets minimum length."""
        data = bytes([0x01, 0x02, 0x03])
        require_min_length(data, 3, "Test")  # Should not raise

    def test_passes_when_length_exceeds_minimum(self) -> None:
        """Validation passes when data exceeds minimum length."""
        data = bytes([0x01, 0x02, 0x03, 0x04])
        require_min_length(data, 2, "Test")  # Should not raise

    def test_raises_when_length_insufficient(self) -> None:
        """Validation raises when data is too short."""
        data = bytes([0x01, 0x02])
        with pytest.raises(ContainerDecodeError) as exc:
            require_min_length(data, 5, "TestContext")
        assert "TestContext" in str(exc.value)
        assert "need at least 5 bytes" in str(exc.value)
        assert "got 2" in str(exc.value)


class TestRequireExactLength:
    """Tests for require_exact_length validation."""

    def test_passes_when_length_matches(self) -> None:
        """Validation passes when length matches exactly."""
        data = bytes([0x01, 0x02, 0x03])
        require_exact_length(data, 3, "Test")  # Should not raise

    def test_raises_when_length_too_short(self) -> None:
        """Validation raises when data is too short."""
        data = bytes([0x01, 0x02])
        with pytest.raises(ContainerDecodeError) as exc:
            require_exact_length(data, 5, "TestContext")
        assert "expected 5 bytes" in str(exc.value)
        assert "got 2" in str(exc.value)

    def test_raises_when_length_too_long(self) -> None:
        """Validation raises when data is too long."""
        data = bytes([0x01, 0x02, 0x03, 0x04, 0x05])
        with pytest.raises(ContainerDecodeError) as exc:
            require_exact_length(data, 3, "TestContext")
        assert "expected 3 bytes" in str(exc.value)
        assert "got 5" in str(exc.value)


class TestRequireLengthRange:
    """Tests for require_length_range validation."""

    def test_passes_at_minimum(self) -> None:
        """Validation passes at minimum of range."""
        data = bytes([0x01, 0x02, 0x03])
        require_length_range(data, 3, 5, "Test")  # Should not raise

    def test_passes_at_maximum(self) -> None:
        """Validation passes at maximum of range."""
        data = bytes([0x01, 0x02, 0x03, 0x04, 0x05])
        require_length_range(data, 3, 5, "Test")  # Should not raise

    def test_passes_within_range(self) -> None:
        """Validation passes within range."""
        data = bytes([0x01, 0x02, 0x03, 0x04])
        require_length_range(data, 3, 5, "Test")  # Should not raise

    def test_raises_below_minimum(self) -> None:
        """Validation raises below minimum."""
        data = bytes([0x01, 0x02])
        with pytest.raises(ContainerDecodeError) as exc:
            require_length_range(data, 3, 5, "TestContext")
        assert "expected 3-5 bytes" in str(exc.value)
        assert "got 2" in str(exc.value)

    def test_raises_above_maximum(self) -> None:
        """Validation raises above maximum."""
        data = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
        with pytest.raises(ContainerDecodeError) as exc:
            require_length_range(data, 3, 5, "TestContext")
        assert "expected 3-5 bytes" in str(exc.value)
        assert "got 6" in str(exc.value)


class TestExtractUint16Le:
    """Tests for extract_uint16_le extraction."""

    def test_extracts_little_endian_value(self) -> None:
        """Correctly extracts little-endian uint16."""
        # 0x5380 in little-endian is bytes [0x80, 0x53]
        data = bytes([0x00, 0x00, 0x80, 0x53, 0x00])
        result = extract_uint16_le(data, 2, "Test")
        assert result == 0x5380

    def test_extracts_at_offset_zero(self) -> None:
        """Extracts from start of data."""
        data = bytes([0x34, 0x12])
        result = extract_uint16_le(data, 0, "Test")
        assert result == 0x1234

    def test_raises_when_offset_out_of_bounds(self) -> None:
        """Raises when offset exceeds data length."""
        data = bytes([0x01, 0x02])
        with pytest.raises(ContainerDecodeError) as exc:
            extract_uint16_le(data, 1, "TestContext")
        assert "cannot read uint16 at offset 1" in str(exc.value)
        assert "data length 2" in str(exc.value)
