"""Tests for encoding detection functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts.discover_datasets.encoding import (
    _has_continuation_bytes,
    _is_valid_utf8,
    detect_encoding,
)


class TestHasContinuationBytes:
    """Tests for _has_continuation_bytes function."""

    def test_valid_continuation(self) -> None:
        """Test valid continuation bytes."""
        raw = b"\xc2\x80"
        result = _has_continuation_bytes(raw, 1, 1)
        assert result is True

    def test_not_enough_bytes(self) -> None:
        """Test when not enough bytes remain for continuation."""
        raw = b"\xc2"
        result = _has_continuation_bytes(raw, 1, 1)
        assert result is False

    def test_invalid_continuation(self) -> None:
        """Test invalid continuation byte value."""
        raw = b"\xc2\x40"
        result = _has_continuation_bytes(raw, 1, 1)
        assert result is False


class TestIsValidUtf8:
    """Tests for _is_valid_utf8 function."""

    def test_valid_2byte(self) -> None:
        """Test valid 2-byte UTF-8 sequence."""
        raw = b"\xc2\x80"
        result = _is_valid_utf8(raw)
        assert result is True

    def test_valid_3byte(self) -> None:
        """Test valid 3-byte UTF-8 sequence."""
        raw = b"\xe4\xb8\xad"
        result = _is_valid_utf8(raw)
        assert result is True

    def test_valid_4byte(self) -> None:
        """Test valid 4-byte UTF-8 sequence."""
        raw = b"\xf0\x9f\x98\x80"
        result = _is_valid_utf8(raw)
        assert result is True

    def test_invalid_2byte_missing_continuation(self) -> None:
        """Test invalid 2-byte sequence missing continuation."""
        raw = b"\xc2,hello"
        result = _is_valid_utf8(raw)
        assert result is False

    def test_invalid_3byte_missing_continuation(self) -> None:
        """Test invalid 3-byte sequence missing continuations."""
        raw = b"\xe4,hello"
        result = _is_valid_utf8(raw)
        assert result is False

    def test_invalid_4byte_missing_continuation(self) -> None:
        """Test invalid 4-byte sequence missing continuations."""
        raw = b"\xf0,hello"
        result = _is_valid_utf8(raw)
        assert result is False

    def test_invalid_bare_continuation(self) -> None:
        """Test bare continuation byte without leading byte."""
        raw = b"\x80hello"
        result = _is_valid_utf8(raw)
        assert result is False


class TestDetectEncoding:
    """Tests for detect_encoding function."""

    def test_utf8_no_bom(self) -> None:
        """Test UTF-8 encoding detection without BOM."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            f.write(b"hello,world\n")
            path = Path(f.name)

        result = detect_encoding(path)
        path.unlink()

        assert result == "utf-8"

    def test_utf8_with_bom(self) -> None:
        """Test UTF-8 encoding detection with BOM."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            f.write(b"\xef\xbb\xbfhello,world\n")
            path = Path(f.name)

        result = detect_encoding(path)
        path.unlink()

        assert result == "utf-8-sig"

    def test_utf16_bom_le(self) -> None:
        """Test UTF-16 LE BOM detection (returns latin-1 as fallback)."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            f.write(b"\xff\xfeh\x00e\x00l\x00l\x00o\x00")
            path = Path(f.name)

        result = detect_encoding(path)
        path.unlink()

        assert result == "latin-1"

    def test_utf16_bom_be(self) -> None:
        """Test UTF-16 BE BOM detection (returns latin-1 as fallback)."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            f.write(b"\xfe\xff\x00h\x00e\x00l\x00l\x00o")
            path = Path(f.name)

        result = detect_encoding(path)
        path.unlink()

        assert result == "latin-1"

    def test_latin1_encoding(self) -> None:
        """Test Latin-1 encoding detection."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            f.write(b"caf\xe9,\xf1,\xfc,\xe0,\xe8\n" * 20)
            path = Path(f.name)

        result = detect_encoding(path)
        path.unlink()

        assert result == "latin-1"

    def test_valid_utf8_multibyte(self) -> None:
        """Test valid UTF-8 with multibyte characters."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            f.write("name,greeting\ntest,\u3053\u3093\u306b\u3061\u306f\n".encode())
            path = Path(f.name)

        result = detect_encoding(path)
        path.unlink()

        assert result == "utf-8"
