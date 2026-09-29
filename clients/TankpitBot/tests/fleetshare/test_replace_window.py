"""Tests for one read of a file another process replaces."""

from __future__ import annotations

import errno
from collections.abc import Generator
from pathlib import Path

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.fleetshare.replace_window import ReplaceWindow, WindowedReadDict, read_once
from tests.conftest import FakeFileSystem

_PATH = Path("runs/bot/artax/knowledge.json")


@pytest.fixture(autouse=True)
def _restore_read_text() -> Generator[None, None, None]:
    """Put back the real reader a test replaced.

    Yields:
        None, with the original reader restored after.
    """
    real_read = _test_hooks.read_text
    yield
    _test_hooks.read_text = real_read


def _raising(error: OSError) -> _test_hooks.ReadTextProtocol:
    """A reader that fails every read with ``error``."""

    def read(path: Path) -> str:
        raise error

    return read


def test_a_present_file_is_read(fake_fs: FakeFileSystem) -> None:
    """A file outside the window comes back whole, with no window named."""
    fake_fs.write_text(_PATH, '{"instance": "artax"}')

    assert read_once(_PATH) == WindowedReadDict(text='{"instance": "artax"}', window=None)


def test_an_empty_file_is_the_empty_window(fake_fs: FakeFileSystem) -> None:
    """A zero-length read is the swap caught before content landed."""
    fake_fs.write_text(_PATH, "")

    assert read_once(_PATH) == WindowedReadDict(text="", window=ReplaceWindow.EMPTY)


def test_a_missing_file_is_the_missing_window(fake_fs: FakeFileSystem) -> None:
    """The fake filesystem's own missing-file error is the missing window."""
    _ = fake_fs

    assert read_once(_PATH) == WindowedReadDict(text="", window=ReplaceWindow.MISSING)


@pytest.mark.parametrize(
    ("error", "window"),
    [
        (PermissionError(errno.EACCES, "Permission denied"), ReplaceWindow.PERMISSION_DENIED),
        (OSError(errno.ENODATA, "No data available"), ReplaceWindow.NO_DATA),
    ],
)
def test_each_raised_form_is_named(error: OSError, window: ReplaceWindow) -> None:
    """The Windows-host refusal and the bind mount's ENODATA are both named."""
    _test_hooks.read_text = _raising(error)

    assert read_once(_PATH) == WindowedReadDict(text="", window=window)


def test_any_other_os_error_raises() -> None:
    """An I/O error is not a replace window and is never absorbed."""
    _test_hooks.read_text = _raising(OSError(errno.EIO, "Input/output error"))

    with pytest.raises(OSError, match="Input/output error"):
        read_once(_PATH)
