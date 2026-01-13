"""Shared fixtures and helpers for tracker tests."""

from __future__ import annotations

import base64


def make_payload(body: bytes) -> str:
    """Create a base64 payload with 2-byte length header.

    Args:
        body: Raw message body bytes.

    Returns:
        Base64-encoded payload with length header.
    """
    header = len(body).to_bytes(2, "little")
    return base64.b64encode(header + body).decode()


def build_test_xor_table(static_key: str, magic: str) -> bytes:
    """Build XOR table for testing.

    Args:
        static_key: Static key string.
        magic: Magic key string.

    Returns:
        XOR table bytes.
    """
    magic_bytes = magic.encode("utf-8")
    key_len = len(static_key)
    return bytes(ord(static_key[i]) ^ magic_bytes[i % len(magic_bytes)] for i in range(key_len))
