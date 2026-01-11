"""Tests for scripts/_test_hooks.py."""

from __future__ import annotations

from pathlib import Path

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


def test_all_exports() -> None:
    """__all__ should export the expected symbols."""
    assert "IsDirProtocol" in _test_hooks.__all__
    assert "is_dir" in _test_hooks.__all__
    assert len(_test_hooks.__all__) == 2
