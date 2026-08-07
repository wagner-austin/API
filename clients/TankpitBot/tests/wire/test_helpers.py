"""Tests for :mod:`tankpit_bot.wire.helpers`.

Covers the byte-combining primitives (x16, x24) and the ``require_*``
validators that raise :class:`DecodeError`.
"""

from __future__ import annotations

import pytest

from tankpit_bot.wire.helpers import (
    DecodeError,
    require_exact_length,
    require_min_length,
    require_parts,
    require_prefix,
    x16,
    x24,
)


class TestX16:
    """Tests for _x16 helper function."""

    def test_combines_bytes_little_endian(self) -> None:
        """Combines two bytes into uint16 little-endian."""
        assert x16(0x34, 0x12) == 0x1234
        assert x16(0x00, 0x00) == 0x0000
        assert x16(0xFF, 0xFF) == 0xFFFF
        assert x16(0x01, 0x00) == 0x0001
        assert x16(0x00, 0x01) == 0x0100

    def test_masks_to_byte_range(self) -> None:
        """Masks input values to byte range."""
        assert x16(0x134, 0x112) == x16(0x34, 0x12)


class TestX24:
    """Tests for _x24 helper function."""

    def test_combines_bytes_big_endian(self) -> None:
        """Combines three bytes into uint24 big-endian."""
        assert x24(0x12, 0x34, 0x56) == 0x123456
        assert x24(0x00, 0x00, 0x00) == 0x000000
        assert x24(0xFF, 0xFF, 0xFF) == 0xFFFFFF
        assert x24(0x01, 0x00, 0x00) == 0x010000
        assert x24(0x00, 0x01, 0x00) == 0x000100


class TestRequireMinLength:
    """Tests for _require_min_length validation."""

    def test_passes_when_sufficient(self) -> None:
        """Validation passes when length is sufficient."""
        require_min_length(bytes([1, 2, 3]), 3, "Test")  # Should not raise
        require_min_length(bytes([1, 2, 3, 4]), 3, "Test")  # Should not raise

    def test_raises_when_insufficient(self) -> None:
        """Validation raises DecodeError when length is insufficient."""
        with pytest.raises(DecodeError) as exc:
            require_min_length(bytes([1, 2]), 5, "TestContext")
        assert "TestContext" in str(exc.value)
        assert ">= 5 bytes" in str(exc.value)
        assert "got 2" in str(exc.value)


class TestRequireExactLength:
    """Tests for _require_exact_length validation."""

    def test_passes_when_exact(self) -> None:
        """Validation passes when length matches exactly."""
        require_exact_length(bytes([1, 2, 3]), 3, "Test")  # Should not raise

    def test_raises_when_wrong_length(self) -> None:
        """Validation raises DecodeError when length is wrong."""
        with pytest.raises(DecodeError) as exc:
            require_exact_length(bytes([1, 2]), 5, "TestContext")
        assert "TestContext" in str(exc.value)
        assert "expected 5 bytes" in str(exc.value)


class TestRequirePrefix:
    """Tests for _require_prefix validation."""

    def test_passes_with_correct_prefix(self) -> None:
        """Validation passes with correct prefix."""
        require_prefix("=team|data", "=", "Test")  # Should not raise
        require_prefix("+info|data", "+", "Test")  # Should not raise

    def test_raises_without_prefix(self) -> None:
        """Validation raises DecodeError without expected prefix."""
        with pytest.raises(DecodeError) as exc:
            require_prefix("team|data", "=", "TestContext")
        assert "TestContext" in str(exc.value)
        assert "expected prefix '='" in str(exc.value)


class TestRequireParts:
    """Tests for _require_parts validation."""

    def test_passes_with_enough_parts(self) -> None:
        """Validation passes with sufficient parts."""
        require_parts(["a", "b", "c"], 3, "Test")  # Should not raise
        require_parts(["a", "b", "c", "d"], 3, "Test")  # Should not raise

    def test_raises_with_insufficient_parts(self) -> None:
        """Validation raises DecodeError with too few parts."""
        with pytest.raises(DecodeError) as exc:
            require_parts(["a", "b"], 5, "TestContext")
        assert "TestContext" in str(exc.value)
        assert ">= 5 parts" in str(exc.value)
