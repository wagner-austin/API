"""Shared fixtures and helpers for tracker tests."""

from __future__ import annotations

import base64
from typing import Protocol

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.capture.xor import XorStaticKeyUnavailableError, reset_static_key_cache
from tests.conftest import FakeFileSystem


class SetMagicTracker(Protocol):
    """The slice of a tracker the static-key contract covers.

    Attributes:
        _xor_table: The tracker's session table, None until built.
    """

    _xor_table: bytes | None

    def set_magic(self, magic: str) -> None:
        """Build this tracker's XOR table from the session magic."""
        ...


def assert_set_magic_requires_static_key(tracker: SetMagicTracker) -> None:
    """Assert ``set_magic`` is fatal without a static key.

    Every tracker used to hand-roll load-key-then-build-table and
    ``return`` silently when the key was missing, leaving ``_xor_table``
    None so every later decode ran against no cipher. Eleven copies of
    that branch, and eleven copies of this test — so the assertion
    lives here once ([[session-state-deglobalisation]]).

    Args:
        tracker: A freshly constructed tracker.
    """
    fs = FakeFileSystem()
    _test_hooks.path_exists = fs.path_exists
    _test_hooks.read_text = fs.read_text
    reset_static_key_cache()

    with pytest.raises(XorStaticKeyUnavailableError, match="static XOR key unavailable"):
        tracker.set_magic("testmagic")
    assert tracker._xor_table is None


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
