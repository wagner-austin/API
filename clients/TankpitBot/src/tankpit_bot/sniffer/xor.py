"""XOR encoding and decoding for WebSocket message processing.

This module provides XOR table building and message decoding functionality
for processing encrypted TankPit WebSocket messages.
"""

from __future__ import annotations

from tankpit_bot.capture.xor import build_xor_table, load_xor_static_key

# Module-level XOR table for unified decoder
_global_xor_table: bytes | None = None
_global_static_key: str | None = None


def build_global_xor_table(magic: str) -> None:
    """Build global XOR table from magic key.

    Args:
        magic: The session magic string for XOR encoding.
    """
    global _global_xor_table, _global_static_key
    static_key, _global_static_key = load_xor_static_key(_global_static_key)
    if static_key is None:
        return
    _global_xor_table = build_xor_table(static_key, magic)


def xor_decode(body: bytes) -> bytes:
    """XOR decode message body (skip first byte which is msg_type).

    Args:
        body: Raw message body bytes.

    Returns:
        XOR-decoded bytes (without the msg_type byte).
    """
    if _global_xor_table is None or len(body) < 2:
        return body[1:] if len(body) > 1 else b""
    decoded = bytearray(len(body) - 1)
    for i in range(len(decoded)):
        if i < len(_global_xor_table):
            decoded[i] = body[i + 1] ^ _global_xor_table[i]
        else:
            decoded[i] = body[i + 1]
    return bytes(decoded)


def get_global_xor_table() -> bytes | None:
    """Get the current global XOR table.

    Returns:
        The XOR table bytes, or None if not initialized.
    """
    return _global_xor_table


def reset_xor_state() -> None:
    """Reset XOR state for testing."""
    global _global_xor_table, _global_static_key
    _global_xor_table = None
    _global_static_key = None


__all__ = [
    "build_global_xor_table",
    "get_global_xor_table",
    "reset_xor_state",
    "xor_decode",
]
