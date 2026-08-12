"""Message signature extraction and identification.

This module provides utilities for extracting and identifying message
signatures from base64-encoded WebSocket payloads.
"""

from __future__ import annotations

import base64

from tankpit_bot.capture.xor import is_valid_base64
from tankpit_bot.container import (
    ContainerMessageType,
    DecodeLevel,
    get_decode_level,
    identify_container_type,
)


def extract_message_signature(payload_b64: str, xor_table: bytes) -> bytes | None:
    """Extract and decode message signature from base64 payload.

    Args:
        payload_b64: Base64 encoded payload.
        xor_table: XOR decryption table.

    Returns:
        Decoded bytes or None if invalid format.
    """
    if not is_valid_base64(payload_b64):
        return None

    payload = base64.b64decode(payload_b64)

    if b"." not in payload[:3]:
        return None

    # We verified dot exists in positions 0-2, so find() returns 0, 1, or 2
    dot_pos = payload.find(b".")
    start = dot_pos + 1
    decode_len = min(len(payload) - start, len(xor_table))
    decoded = bytes(payload[start + j] ^ xor_table[j] for j in range(decode_len))
    return decoded if decoded else None


def format_sig_key(sig: int) -> str:
    """Format signature as display key.

    Args:
        sig: The signature byte value.

    Returns:
        Formatted string like "0x41 'A'".
    """
    char = chr(sig) if 32 <= sig < 127 else "?"
    return f"0x{sig:02X} '{char}'"


def identify_message(data: bytes) -> tuple[str, DecodeLevel] | None:
    """Identify message type and decode level using the actual decoder.

    Uses identify_container_type from container_decoder to determine the
    message type, then looks up the decode level from the authoritative
    MESSAGE_TYPE_LEVELS registry.

    Args:
        data: XOR-decoded message bytes.

    Returns:
        Tuple of (name, level) if identified, None if UNKNOWN.
    """
    msg_type = identify_container_type(data)
    if msg_type == ContainerMessageType.UNKNOWN:
        return None
    level = get_decode_level(msg_type)
    name = msg_type.name.lower()
    return (name, level)


__all__ = [
    "extract_message_signature",
    "format_sig_key",
    "identify_message",
]
