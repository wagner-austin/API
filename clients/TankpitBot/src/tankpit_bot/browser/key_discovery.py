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
from tankpit_bot.resources import static_key_file_path


def load_static_key() -> str:
    """Load the static XOR key from file.

    Returns:
        The 1000-character static key.

    Raises:
        FileNotFoundError: If key file does not exist.
        ValueError: If key is not exactly 1000 characters.
    """
    content = _test_hooks.read_text(static_key_file_path())
    key = content.strip()
    if len(key) != STATIC_KEY_LENGTH:
        raise ValueError(f"Static key has {len(key)} chars, expected {STATIC_KEY_LENGTH}")
    return key


# There is deliberately no writer here. The key is package data
# (:mod:`tankpit_bot.data`), addressed so the question "where is this asset"
# has one answer under a checkout, a pip install, a container and a cluster
# image alike ([[packaged-data-assets]]) — and package data is read, never
# written. The writer that used to live here fired on every session and
# succeeded only where the install happened to be writable, so it worked in a
# checkout and killed every containerized fleet bot four seconds into the
# game. Key rotation is now reported as drift by
# ``lifecycle.py::_check_shipped_static_key``; regenerating the bundled asset
# is a deliberate act, not something a running bot does to its own
# distribution.

__all__ = [
    "load_static_key",
]
