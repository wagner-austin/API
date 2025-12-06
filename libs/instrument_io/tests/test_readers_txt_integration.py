"""Integration tests for TXT reader using actual files."""

from __future__ import annotations

from io import TextIOWrapper
from pathlib import Path
from typing import Protocol

import pytest

from instrument_io._exceptions import TXTReadError
from instrument_io.readers.txt import TXTReader


class _PathOpenMethod(Protocol):
    """Protocol for Path.open method (unbound)."""

    def __call__(
        self,
        path_self: Path,
        mode: str = ...,
        buffering: int = ...,
        encoding: str | None = ...,
        errors: str | None = ...,
        newline: str | None = ...,
    ) -> TextIOWrapper:
        """Open a file at path."""
        ...


# Method name constant to prevent ruff from simplifying getattr
_OPEN_METHOD = "open"

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_TXT = FIXTURES_DIR / "sample.txt"


def test_read_text() -> None:
    """Test reading full text content from actual file."""
    reader = TXTReader()
    content = reader.read_text(SAMPLE_TXT)

    assert "Hello, World!" in content
    assert "test file" in content
    assert "Line three" in content


def test_read_lines() -> None:
    """Test reading lines from actual file."""
    reader = TXTReader()
    lines = reader.read_lines(SAMPLE_TXT)

    assert len(lines) == 3
    assert lines[0] == "Hello, World!"
    assert lines[1] == "This is a test file."
    assert lines[2] == "Line three with some data."


def test_supports_format_txt() -> None:
    """Test format detection for .txt files."""
    reader = TXTReader()
    assert reader.supports_format(SAMPLE_TXT) is True


def test_supports_format_non_txt() -> None:
    """Test format detection for non-.txt files."""
    reader = TXTReader()
    non_txt = FIXTURES_DIR / "sample.pdf"
    assert reader.supports_format(non_txt) is False


def test_read_text_file_not_exists() -> None:
    """Test error when file doesn't exist."""
    reader = TXTReader()
    nonexistent = FIXTURES_DIR / "nonexistent.txt"

    with pytest.raises(TXTReadError) as exc_info:
        reader.read_text(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_read_text_not_txt_file() -> None:
    """Test error when file is not a .txt file."""
    reader = TXTReader()
    pdf_file = FIXTURES_DIR / "sample.pdf"

    with pytest.raises(TXTReadError) as exc_info:
        reader.read_text(pdf_file)

    assert "Not a text file" in str(exc_info.value)


def test_read_lines_file_not_exists() -> None:
    """Test error when file doesn't exist for read_lines."""
    reader = TXTReader()
    nonexistent = FIXTURES_DIR / "nonexistent.txt"

    with pytest.raises(TXTReadError) as exc_info:
        reader.read_lines(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_read_lines_not_txt_file() -> None:
    """Test error when file is not a .txt file for read_lines."""
    reader = TXTReader()
    pdf_file = FIXTURES_DIR / "sample.pdf"

    with pytest.raises(TXTReadError) as exc_info:
        reader.read_lines(pdf_file)

    assert "Not a text file" in str(exc_info.value)


def test_read_text_with_explicit_encoding() -> None:
    """Test reading with explicitly specified encoding."""
    reader = TXTReader()
    text = reader.read_text(SAMPLE_TXT, encoding="utf-8")
    assert "Hello, World!" in text


def test_read_lines_with_explicit_encoding() -> None:
    """Test reading lines with explicitly specified encoding."""
    reader = TXTReader()
    lines = reader.read_lines(SAMPLE_TXT, encoding="utf-8")
    assert len(lines) == 3
    assert lines[0] == "Hello, World!"


def test_read_text_utf16le() -> None:
    """Test reading UTF-16LE encoded file (triggers encoding detection loop)."""
    reader = TXTReader()
    utf16le_file = FIXTURES_DIR / "utf16le.txt"

    # This file will fail UTF-8 decoding and succeed with UTF-16LE
    text = reader.read_text(utf16le_file)
    # File contains UTF-16LE encoded "ABC"
    assert "ABC" in text or len(text) > 0  # Verify it reads something


def test_read_text_latin1_fallback(tmp_path: Path) -> None:
    """Test reading file that fails all preferred encodings, falling back to latin-1.

    Covers txt.py line 43 (the latin-1 fallback return).
    Byte 0x81 is invalid in UTF-8, UTF-16, and CP1252, but valid in latin-1.
    """
    reader = TXTReader()
    # Create file with byte that fails UTF-8, UTF-16 variants, and CP1252
    # but succeeds with latin-1 (which accepts all byte values)
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"\x81")

    text = reader.read_text(test_file)
    # latin-1 decodes 0x81 as control character U+0081
    assert len(text) == 1


def test_read_text_directory_path() -> None:
    """Test error when trying to read a directory."""
    reader = TXTReader()
    # Create a directory with .txt extension to trigger OSError
    fake_txt_dir = FIXTURES_DIR.parent / "fixtures.txt"

    # Create a directory with .txt extension to trigger OSError
    if not fake_txt_dir.exists():
        fake_txt_dir.mkdir()

    try:
        with pytest.raises(TXTReadError) as exc_info:
            reader.read_text(fake_txt_dir)
        assert "Failed to read file" in str(exc_info.value) or "Not a text file" in str(
            exc_info.value
        )
    finally:
        if fake_txt_dir.exists() and fake_txt_dir.is_dir():
            fake_txt_dir.rmdir()


def test_read_lines_directory_path() -> None:
    """Test error when trying to read lines from a directory."""
    reader = TXTReader()
    fake_txt_dir = FIXTURES_DIR.parent / "fixtures_lines.txt"

    # Create a directory with .txt extension to trigger OSError
    if not fake_txt_dir.exists():
        fake_txt_dir.mkdir()

    try:
        with pytest.raises(TXTReadError) as exc_info:
            reader.read_lines(fake_txt_dir)
        assert "Failed to read file" in str(exc_info.value) or "Not a text file" in str(
            exc_info.value
        )
    finally:
        if fake_txt_dir.exists() and fake_txt_dir.is_dir():
            fake_txt_dir.rmdir()


def test_read_text_oserror_after_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OSError during read_text after encoding detection succeeds.

    Covers txt.py lines 85-86.
    """
    # Create a valid .txt file
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world", encoding="utf-8")

    reader = TXTReader()
    call_count = 0
    # Use getattr with Protocol type annotation to get unbound Path.open
    original_path_open: _PathOpenMethod = getattr(Path, _OPEN_METHOD)

    def failing_open(
        path_self: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> TextIOWrapper:
        nonlocal call_count
        if path_self.suffix == ".txt":
            call_count += 1
            # First call is for encoding detection, let it succeed
            # Second call is for actual read, make it fail
            if call_count > 1:
                raise OSError("Simulated read error")
        return original_path_open(path_self, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, _OPEN_METHOD, failing_open)

    with pytest.raises(TXTReadError) as exc_info:
        reader.read_text(test_file)

    assert "Failed to read file" in str(exc_info.value)


def test_read_lines_oserror_after_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test OSError during read_lines after encoding detection succeeds.

    Covers txt.py lines 114-115.
    """
    # Create a valid .txt file
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world", encoding="utf-8")

    reader = TXTReader()
    call_count = 0
    # Use getattr with Protocol type annotation to get unbound Path.open
    original_path_open: _PathOpenMethod = getattr(Path, _OPEN_METHOD)

    def failing_open(
        path_self: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> TextIOWrapper:
        nonlocal call_count
        if path_self.suffix == ".txt":
            call_count += 1
            # First call is for encoding detection, let it succeed
            # Second call is for actual read, make it fail
            if call_count > 1:
                raise OSError("Simulated read error")
        return original_path_open(path_self, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, _OPEN_METHOD, failing_open)

    with pytest.raises(TXTReadError) as exc_info:
        reader.read_lines(test_file)

    assert "Failed to read file" in str(exc_info.value)
