"""XOR key loading and table building utilities.

This module provides shared XOR decoding functions used by all tracker classes.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

from tankpit_bot import _test_hooks


def load_xor_static_key(cached: str | None) -> tuple[str | None, str | None]:
    """Load static XOR key from file, using cache if available.

    Args:
        cached: Previously cached static key, or None.

    Returns:
        Tuple of (key, cache_value) - if cached was not None, returns (cached, cached).
        Otherwise loads from file and returns (key, key) or (None, None).
    """
    if cached is not None:
        return cached, cached

    static_key_path = Path(__file__).parent.parent.parent.parent / "xor_static_key.txt"
    if _test_hooks.path_exists(static_key_path):
        key = _test_hooks.read_text(static_key_path).strip()
        return key, key
    return None, None


def build_xor_table(static_key: str, magic: str) -> bytes:
    """Build XOR table from static key and magic.

    Args:
        static_key: The static XOR key.
        magic: The magic key from session.

    Returns:
        XOR table bytes.
    """
    table = bytearray(len(static_key))
    for i in range(len(static_key)):
        table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
    return bytes(table)


def is_valid_base64(payload: str) -> bool:
    """Check if payload is valid base64.

    Args:
        payload: String to validate.

    Returns:
        True if valid base64, False otherwise.
    """
    if not payload:
        return False
    # Base64 uses A-Z, a-z, 0-9, +, /, and = for padding
    pattern = r"^[A-Za-z0-9+/]*={0,2}$"
    if not re.match(pattern, payload):
        return False
    # Length must be multiple of 4 (with padding)
    return len(payload) % 4 == 0


def decode_base64_safe(payload: str) -> bytes | None:
    """Validate and decode base64 payload.

    Args:
        payload: Base64-encoded string.

    Returns:
        Decoded bytes, or None if invalid.
    """
    if not is_valid_base64(payload):
        return None
    return base64.b64decode(payload)


def xor_decode_body(body: bytes, xor_table: bytes, offset: int = 0) -> bytes:
    """XOR decode a message body.

    Args:
        body: Raw message body bytes.
        xor_table: XOR table to use.
        offset: Starting offset in body (default 0).

    Returns:
        Decoded bytes.
    """
    decoded = bytearray(len(body) - offset)
    for i in range(len(decoded)):
        decoded[i] = body[i + offset] ^ xor_table[i]
    return bytes(decoded)


__all__ = [
    "build_xor_table",
    "decode_base64_safe",
    "is_valid_base64",
    "load_xor_static_key",
    "xor_decode_body",
]
