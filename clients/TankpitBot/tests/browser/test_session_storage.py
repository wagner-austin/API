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
    STORAGE_STATE_PATH,
    StorageStateCacheError,
    load_storage_state,
    save_storage_state,
)
from tests.conftest import FakeFileSystem
from tests.fakes.base import FakeBrowserContext

_STORAGE_PATH = Path("runs/state/tankpit.storage.json")


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


class TestStorageStatePathDefault:
    """Contract for the ``STORAGE_STATE_PATH`` constant."""

    def test_default_points_at_runs_state_directory(self) -> None:
        """The canonical location lives under ``runs/state/`` next to run logs."""
        assert Path("runs/state/tankpit.storage.json") == STORAGE_STATE_PATH
