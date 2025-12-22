"""Encoding detection for dataset files.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal


def _has_continuation_bytes(raw: bytes, start: int, count: int) -> bool:
    """Check if raw has count continuation bytes (0x80-0xBF) starting at start.

    Args:
        raw: Raw bytes.
        start: Starting index.
        count: Number of continuation bytes expected.

    Returns:
        True if all continuation bytes are valid.
    """
    if start + count > len(raw):
        return False
    return all(0x80 <= raw[start + offset] <= 0xBF for offset in range(count))


def _is_valid_utf8(raw: bytes) -> bool:
    """Check if bytes form valid UTF-8 sequences.

    In valid UTF-8:
    - 0xC0-0xDF must be followed by 1 continuation byte (0x80-0xBF)
    - 0xE0-0xEF must be followed by 2 continuation bytes
    - 0xF0-0xF7 must be followed by 3 continuation bytes

    Args:
        raw: Raw bytes to validate.

    Returns:
        True if valid UTF-8, False otherwise.
    """
    i = 0
    while i < len(raw):
        byte = raw[i]
        if byte < 0x80:
            i += 1
        elif 0xC0 <= byte <= 0xDF:
            if not _has_continuation_bytes(raw, i + 1, 1):
                return False
            i += 2
        elif 0xE0 <= byte <= 0xEF:
            if not _has_continuation_bytes(raw, i + 1, 2):
                return False
            i += 3
        elif 0xF0 <= byte <= 0xF7:
            if not _has_continuation_bytes(raw, i + 1, 3):
                return False
            i += 4
        else:
            return False
    return True


def detect_encoding(path: Path) -> Literal["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
    """Detect file encoding by checking byte patterns.

    Args:
        path: Path to the file.

    Returns:
        Detected encoding.
    """
    with open(path, "rb") as f:
        raw = f.read(1024)

    # Check for UTF-8 BOM
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    # Check for UTF-16 BOM (would need conversion, treat as latin-1)
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "latin-1"

    # Check if all bytes are ASCII (< 0x80)
    if not any(byte >= 0x80 for byte in raw):
        return "utf-8"

    # Validate UTF-8 byte sequences
    if _is_valid_utf8(raw):
        return "utf-8"

    return "latin-1"


__all__ = [
    "detect_encoding",
]
