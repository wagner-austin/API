"""Tests for scripts/_test_hooks.py."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from scripts import _test_hooks


def test_real_is_dir_returns_true_for_existing_dir(tmp_path: Path) -> None:
    """_real_is_dir should return True for existing directories."""
    result = _test_hooks._real_is_dir(tmp_path)
    assert result is True


def test_real_is_dir_returns_false_for_nonexistent_path(tmp_path: Path) -> None:
    """_real_is_dir should return False for nonexistent paths."""
    nonexistent = tmp_path / "does_not_exist"
    result = _test_hooks._real_is_dir(nonexistent)
    assert result is False


def test_real_is_dir_returns_false_for_file(tmp_path: Path) -> None:
    """_real_is_dir should return False for files."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("test")
    result = _test_hooks._real_is_dir(file_path)
    assert result is False


def test_is_dir_hook_is_real_is_dir_by_default() -> None:
    """is_dir hook should be _real_is_dir by default."""
    assert _test_hooks.is_dir is _test_hooks._real_is_dir


def test_is_dir_protocol_callable(tmp_path: Path) -> None:
    """is_dir should be callable via the protocol."""
    result = _test_hooks.is_dir(tmp_path)
    assert result is True


def test_get_script_path_hook_is_real_by_default() -> None:
    """get_script_path hook should be _real_get_script_path by default."""
    assert _test_hooks.get_script_path is _test_hooks._real_get_script_path


def test_set_script_path_and_get_script_path(tmp_path: Path) -> None:
    """set_script_path should store the path, get_script_path should retrieve it."""
    original = _test_hooks._SCRIPT_PATH
    try:
        fake_path = tmp_path / "scripts" / "guard.py"
        _test_hooks.set_script_path(fake_path)
        result = _test_hooks._real_get_script_path()
        assert result == fake_path
    finally:
        _test_hooks._SCRIPT_PATH = original


@pytest.fixture()
def _clear_script_path() -> Generator[None, None, None]:
    """Temporarily clear _SCRIPT_PATH to test error path."""
    original = _test_hooks._SCRIPT_PATH
    _test_hooks._SCRIPT_PATH = None
    try:
        yield
    finally:
        _test_hooks._SCRIPT_PATH = original


def test_real_get_script_path_raises_when_not_set(_clear_script_path: None) -> None:
    """_real_get_script_path should raise RuntimeError when path not set."""
    del _clear_script_path
    with pytest.raises(RuntimeError, match="Script path not set"):
        _test_hooks._real_get_script_path()


def test_all_exports() -> None:
    """__all__ should export the expected symbols."""
    expected = {
        "GetScriptPathProtocol",
        "IsDirProtocol",
        "get_script_path",
        "is_dir",
        "set_script_path",
    }
    assert set(_test_hooks.__all__) == expected
