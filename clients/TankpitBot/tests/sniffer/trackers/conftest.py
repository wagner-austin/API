"""Shared fixtures and helpers for tracker tests."""

from __future__ import annotations

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


# ``make_payload`` and ``build_test_xor_table`` used to live here. The
# first was one of eleven copies of "length header + base64" and is now
# ``tests.wire_builders.frame_payload``; the second was a third
# implementation of the table math and is now
# ``protocol.codec.build_xor_table``, the production one
# ([[session-state-deglobalisation]]).
