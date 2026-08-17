"""Static XOR key load/save.

The static key is captured from the game client's own JS
(``lifecycle._capture_static_key``) and persisted here. The
brute-force first-byte discovery that once lived in this module
(``extract_xor_first_bytes`` / ``find_best_static_byte``) was removed
2026-08-17: its only callers were its own tests, and its
``_test_hooks`` override slot was read by nothing — dead machinery
from the pre-capture key-cracking days, exactly the shape the
2026-08-17 nullable-hook sweep hunts.
"""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.browser.types import STATIC_KEY_LENGTH
from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH


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
    "load_static_key",
    "save_static_key",
]
