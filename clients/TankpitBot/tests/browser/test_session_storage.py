"""Tests for :mod:`tankpit_bot.browser.session_storage`.

Covers ``load_storage_state`` (missing / empty / corrupt / valid) and
``save_storage_state`` (writes the JSON-serialised snapshot via
:mod:`tankpit_bot._test_hooks.fs`). No mocks — the fake browser
context returns a real (empty) storage-state dict, and the fake file
system captures the written text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot.browser.session_storage import (
    StorageStateCacheError,
    load_storage_state,
    resolve_storage_state_path,
    save_storage_state,
)
from tests.conftest import FakeEnv, FakeFileSystem
from tests.fakes.base import FakeBrowserContext

_STORAGE_PATH = Path("runs/state/tankpit.guest.storage.json")


class TestLoadStorageState:
    """Contract for ``load_storage_state``."""

    def test_missing_file_returns_none(self, fake_fs: FakeFileSystem) -> None:
        """A never-cached storage-state file resolves to ``None``."""
        _ = fake_fs
        assert load_storage_state(_STORAGE_PATH) is None

    def test_valid_json_returns_stringified_path(self, fake_fs: FakeFileSystem) -> None:
        """A parseable cache file yields its string path for Playwright."""
        fake_fs.write_text(_STORAGE_PATH, '{"cookies": [], "origins": []}')

        result = load_storage_state(_STORAGE_PATH)

        assert result == str(_STORAGE_PATH)

    def test_empty_file_raises(self, fake_fs: FakeFileSystem) -> None:
        """An empty cache file surfaces as :class:`StorageStateCacheError`."""
        fake_fs.write_text(_STORAGE_PATH, "")

        with pytest.raises(StorageStateCacheError, match="empty"):
            load_storage_state(_STORAGE_PATH)

    def test_whitespace_only_file_raises(self, fake_fs: FakeFileSystem) -> None:
        """A whitespace-only cache file is treated as empty and rejected."""
        fake_fs.write_text(_STORAGE_PATH, "   \n\t  \n")

        with pytest.raises(StorageStateCacheError, match="empty"):
            load_storage_state(_STORAGE_PATH)

    def test_corrupt_json_raises(self, fake_fs: FakeFileSystem) -> None:
        """A malformed JSON body surfaces as :class:`StorageStateCacheError`."""
        fake_fs.write_text(_STORAGE_PATH, "{not-valid-json")

        with pytest.raises(StorageStateCacheError, match="not valid JSON"):
            load_storage_state(_STORAGE_PATH)

    def test_storage_state_cache_error_is_value_error(self) -> None:
        """The cache error is a :class:`ValueError` for catch-any callers."""
        assert issubclass(StorageStateCacheError, ValueError)


class TestSaveStorageState:
    """Contract for ``save_storage_state``."""

    def test_writes_serialised_snapshot(self, fake_fs: FakeFileSystem) -> None:
        """The context's storage-state dict is JSON-serialised and written."""
        context = FakeBrowserContext()

        save_storage_state(context, _STORAGE_PATH)

        written = fake_fs.get_written_files()
        assert str(_STORAGE_PATH) in written
        # The fake context returns the canonical empty-state shape;
        # confirming both keys are present is enough — the JSON
        # formatter is pinned to indent=2 so we don't hard-code the
        # exact byte-for-byte body here.
        payload = written[str(_STORAGE_PATH)]
        assert '"cookies"' in payload
        assert '"origins"' in payload

    def test_round_trip_load_after_save(self, fake_fs: FakeFileSystem) -> None:
        """A saved snapshot survives a subsequent load without raising."""
        context = FakeBrowserContext()
        save_storage_state(context, _STORAGE_PATH)

        result = load_storage_state(_STORAGE_PATH)

        assert result == str(_STORAGE_PATH)


class TestResolveStorageStatePath:
    """Contract for ``resolve_storage_state_path``.

    The cache is keyed by LOGIN IDENTITY — the 2026-08-13 incident
    where a fleet child spawned as Arterial resumed the shared jar's
    Artax session (no login flow ran at all) is the failure this
    keying prevents. Guest sessions share the ``guest`` identity.
    """

    def test_guest_preference_never_consults_accounts(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """``prefer_account=False`` is always the guest identity.

        Even with a selector set: a guest session carries no account
        cookies, so account resolution must not run (a bad selector
        would otherwise fail a guest launch for no reason).
        """
        _ = fake_fs
        fake_env.set("TANKPIT_ACCOUNT", "NoSuchAccount")

        assert resolve_storage_state_path(False) == Path("runs/state/tankpit.guest.storage.json")

    def test_no_accounts_configured_is_guest(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """Account preference without any configured account is guest."""
        _ = fake_fs, fake_env

        assert resolve_storage_state_path(True) == Path("runs/state/tankpit.guest.storage.json")

    def test_resolved_account_keys_the_path(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """The selected account's username names the cache, sanitized.

        Two different selections yield two different files — the
        property the shared jar violated.
        """
        _ = fake_fs
        fake_env.set("TANKPIT_USERNAME", "Artax")
        fake_env.set("TANKPIT_PASSWORD", "secret")

        assert resolve_storage_state_path(True) == Path("runs/state/tankpit.artax.storage.json")

        fake_env.set("TANKPIT_USERNAME", "O'Brien 2")

        assert resolve_storage_state_path(True) == Path("runs/state/tankpit.o-brien-2.storage.json")

    def test_unsanitizable_username_is_guest(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """A username that sanitizes to nothing falls to the guest identity."""
        _ = fake_fs
        fake_env.set("TANKPIT_USERNAME", "")
        fake_env.set("TANKPIT_PASSWORD", "secret")

        assert resolve_storage_state_path(True) == Path("runs/state/tankpit.guest.storage.json")
