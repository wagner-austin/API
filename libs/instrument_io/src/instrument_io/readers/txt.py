"""Plain text file reader implementation.

Provides typed reading of plain text files with encoding detection.
All methods raise exceptions on failure - no recovery or fallbacks.
"""

from __future__ import annotations

from pathlib import Path

from instrument_io._exceptions import TXTReadError
from instrument_io.testing import hooks


def _is_txt_file(path: Path) -> bool:
    """Check if path is a text file."""
    return path.is_file() and path.suffix.lower() == ".txt"


def _detect_encoding(path: Path) -> str:
    """Detect file encoding by trying common encodings via hook.

    Args:
        path: Path to text file.

    Returns:
        Detected encoding name. Always succeeds because latin-1 accepts all bytes.

    Note:
        latin-1 accepts all byte values (0x00-0xFF), making it the guaranteed fallback.
    """
    return hooks.txt_detect_encoding(path)


class TXTReader:
    """Reader for plain text files.

    Provides typed access to text file content with automatic encoding detection.
    All methods raise exceptions on failure - no recovery or fallbacks.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if path is a text file.

        Args:
            path: Path to check.

        Returns:
            True if path is a text file (.txt).
        """
        return _is_txt_file(path)

    def read_text(self, path: Path, encoding: str | None = None) -> str:
        """Read full text content from file.

        Args:
            path: Path to text file.
            encoding: Optional encoding name. If None, auto-detects encoding.

        Returns:
            Full text content.

        Raises:
            TXTReadError: If reading fails.
        """
        if not path.exists():
            raise TXTReadError(str(path), "File does not exist")

        if not _is_txt_file(path):
            raise TXTReadError(str(path), "Not a text file")

        detected_encoding = encoding if encoding else _detect_encoding(path)
        return hooks.txt_read_text(path, detected_encoding)

    def read_lines(self, path: Path, encoding: str | None = None) -> list[str]:
        """Read text file as list of lines.

        Newline characters are stripped from line endings.

        Args:
            path: Path to text file.
            encoding: Optional encoding name. If None, auto-detects encoding.

        Returns:
            List of text lines (newlines stripped).

        Raises:
            TXTReadError: If reading fails.
        """
        if not path.exists():
            raise TXTReadError(str(path), "File does not exist")

        if not _is_txt_file(path):
            raise TXTReadError(str(path), "Not a text file")

        detected_encoding = encoding if encoding else _detect_encoding(path)
        return hooks.txt_read_lines(path, detected_encoding)


__all__ = [
    "TXTReader",
]
