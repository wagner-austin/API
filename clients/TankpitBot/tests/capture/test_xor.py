"""Tests for tankpit_bot.capture.xor module."""

from __future__ import annotations

from tankpit_bot.capture import (
    decode_base64_safe,
    xor_decode_body,
)
from tankpit_bot.capture.xor import is_valid_base64


class TestIsValidBase64:
    """Tests for is_valid_base64 function."""

    def test_valid_base64(self) -> None:
        """Test valid base64 strings return True."""
        assert is_valid_base64("SGVsbG8=") is True
        assert is_valid_base64("QUJDRA==") is True
        assert is_valid_base64("dGVzdA==") is True

    def test_valid_base64_no_padding(self) -> None:
        """Test valid base64 without padding returns True."""
        # Length must be multiple of 4
        assert is_valid_base64("QUJD") is True
        assert is_valid_base64("YWJjZA==") is True

    def test_empty_string_returns_false(self) -> None:
        """Test empty string returns False."""
        assert is_valid_base64("") is False

    def test_invalid_chars_returns_false(self) -> None:
        """Test invalid characters return False."""
        assert is_valid_base64("ABC!") is False
        assert is_valid_base64("AB@D") is False
        assert is_valid_base64("test$") is False

    def test_invalid_length_returns_false(self) -> None:
        """Test strings with length not multiple of 4 return False."""
        assert is_valid_base64("abc") is False
        assert is_valid_base64("abcde") is False
        assert is_valid_base64("ab") is False


class TestDecodeBase64Safe:
    """Tests for decode_base64_safe function."""

    def test_valid_base64_decodes(self) -> None:
        """Test valid base64 decodes correctly."""
        result = decode_base64_safe("SGVsbG8=")
        assert result == b"Hello"

    def test_valid_base64_no_padding(self) -> None:
        """Test valid base64 without padding decodes correctly."""
        result = decode_base64_safe("QUJD")
        assert result == b"ABC"

    def test_invalid_base64_returns_none(self) -> None:
        """Test invalid base64 returns None."""
        result = decode_base64_safe("abc")
        assert result is None

    def test_invalid_chars_returns_none(self) -> None:
        """Test invalid characters return None."""
        result = decode_base64_safe("ABC!")
        assert result is None

    def test_empty_returns_none(self) -> None:
        """Test empty string returns None."""
        result = decode_base64_safe("")
        assert result is None


class TestXorDecodeBody:
    """Tests for xor_decode_body function."""

    def test_basic_decode(self) -> None:
        """Test basic XOR decoding without offset."""
        body = bytes([0x41, 0x42, 0x43, 0x44])  # ABCD
        xor_table = bytes([0x10, 0x20, 0x30, 0x40])
        result = xor_decode_body(body, xor_table)
        assert result == bytes([0x51, 0x62, 0x73, 0x04])

    def test_decode_with_offset(self) -> None:
        """Test XOR decoding with offset skips bytes."""
        body = bytes([0xFF, 0xFF, 0x41, 0x42, 0x43])  # Skip first 2 bytes
        xor_table = bytes([0x10, 0x20, 0x30, 0x40, 0x50])
        result = xor_decode_body(body, xor_table, offset=2)
        # Decodes bytes 2,3,4 using table positions 0,1,2
        assert result == bytes([0x51, 0x62, 0x73])

    def test_decode_empty_body(self) -> None:
        """Test decoding empty body returns empty."""
        body = b""
        xor_table = bytes([0x10, 0x20])
        result = xor_decode_body(body, xor_table)
        assert result == b""

    def test_decode_body_shorter_than_table(self) -> None:
        """Test decoding body shorter than XOR table."""
        body = bytes([0x41, 0x42])
        xor_table = bytes([0x10, 0x20, 0x30, 0x40])
        result = xor_decode_body(body, xor_table)
        assert result == bytes([0x51, 0x62])
