"""Static XOR key discovery functions.

This module provides functions for extracting and managing the static
XOR key used in the TankPit protocol.
"""

from __future__ import annotations

import base64

from tankpit_bot import _test_hooks
from tankpit_bot.browser.types import (
    KNOWN_PROTOCOL_SIGNATURES,
    STATIC_KEY_LENGTH,
    TEXT_MESSAGE_TYPES,
)
from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
from tankpit_bot.types import CapturedMessage


def extract_xor_first_bytes(messages: list[CapturedMessage]) -> list[int]:
    """Extract first XOR-encoded bytes from binary messages.

    Parses received messages, skips text messages, and extracts the first
    XOR-encoded data byte from each binary message.

    Args:
        messages: List of captured WebSocket messages.

    Returns:
        List of first XOR-encoded bytes from binary messages.
    """
    raw_first_bytes: list[int] = []

    for msg in messages:
        if msg["direction"] != "received":
            continue

        payload_b64 = msg["payload"]
        payload = base64.b64decode(payload_b64)

        if len(payload) < 4:
            continue

        # First 2 bytes are length header, byte[2] is message type
        msg_type = payload[2]

        # Skip text messages
        if msg_type in TEXT_MESSAGE_TYPES:
            continue

        # Binary messages have XOR-encoded data starting at byte[3]
        raw_first_bytes.append(payload[3])

    return raw_first_bytes


def find_best_static_byte(raw_first_bytes: list[int], magic_first_byte: int) -> tuple[int, int]:
    """Find the static key's first byte that maximizes known signature matches.

    Brute-forces all 256 possible values to find which static[0] produces
    the most known protocol signatures when XOR'd with captured data.

    Args:
        raw_first_bytes: First XOR-encoded bytes from binary messages.
        magic_first_byte: ASCII value of magic key's first character.

    Returns:
        Tuple of (best_static_byte, match_count).
    """
    best_static_0 = 0
    best_coverage = 0

    for static_0 in range(256):
        table_0 = static_0 ^ magic_first_byte
        known_count = sum(
            1 for raw_0 in raw_first_bytes if (raw_0 ^ table_0) in KNOWN_PROTOCOL_SIGNATURES
        )
        if known_count > best_coverage:
            best_coverage = known_count
            best_static_0 = static_0

    return best_static_0, best_coverage


def load_static_key() -> str:
    """Load the static XOR key from file.

    Returns:
        The 1000-character static key.

    Raises:
        FileNotFoundError: If key file does not exist.
        ValueError: If key is not exactly 1000 characters.
    """
    content = _test_hooks.read_text(DEFAULT_STATIC_KEY_PATH)
    key = content.strip()
    if len(key) != STATIC_KEY_LENGTH:
        raise ValueError(f"Static key has {len(key)} chars, expected {STATIC_KEY_LENGTH}")
    return key


def save_static_key(key: str) -> None:
    """Save the static XOR key to file.

    Args:
        key: The 1000-character static key.

    Raises:
        ValueError: If key is not exactly 1000 characters.
    """
    if len(key) != STATIC_KEY_LENGTH:
        raise ValueError(f"Static key has {len(key)} chars, expected {STATIC_KEY_LENGTH}")
    _test_hooks.write_text(DEFAULT_STATIC_KEY_PATH, key + "\n")


__all__ = [
    "extract_xor_first_bytes",
    "find_best_static_byte",
    "load_static_key",
    "save_static_key",
]
