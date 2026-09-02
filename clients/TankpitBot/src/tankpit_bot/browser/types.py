"""Browser module types and constants.

This module contains constants, error types, and shared values for the
browser module.
"""

from __future__ import annotations

# Known binary message signatures from TankPit protocol.
# These are the first decoded byte of binary messages after XOR decoding.
KNOWN_PROTOCOL_SIGNATURES: frozenset[int] = frozenset(
    {
        0x21,
        0x28,
        0x29,
        0x2B,
        0x2E,
        0x2F,
        0x3D,
        0x3E,
        0x3F,
        0x41,
        0x43,
        0x45,
        0x46,
        0x47,
        0x49,
        0x4A,
        0x4B,
        0x4C,
        0x4D,
        0x4F,
        0x52,
        0x53,
        0x54,
        0x56,
        0x58,
        0x5A,
        0x64,
        0x67,
        0x74,
    }
)

# Text message type bytes that should be skipped during XOR analysis.
TEXT_MESSAGE_TYPES: frozenset[int] = frozenset({0x2B, 0x2D, 0x3D, 0x25, 0x2A, 0x7E})

# The static-key path is NOT redeclared here. It was a private copy of
# the same __file__-relative expression that protocol/codec.py owns, so
# a move of either file silently pointed the two at different places.
# Consumers import static_key_file_path from its owner directly —
# no alias in between ([[session-state-deglobalisation]]).

# Expected length of the static XOR key.
STATIC_KEY_LENGTH: int = 1000


class BrowserError(Exception):
    """Base error for browser operations."""


class PlaywrightNotInstalledError(BrowserError):
    """Raised when Playwright hook is not installed."""


class GameNotJoinedError(BrowserError):
    """Raised when game doesn't load properly."""


__all__ = [
    "STATIC_KEY_LENGTH",
    "TEXT_MESSAGE_TYPES",
    "BrowserError",
    "GameNotJoinedError",
    "PlaywrightNotInstalledError",
]
