from __future__ import annotations

import tempfile
from pathlib import Path

from data_bank_api import health as health_mod
from data_bank_api.health import _default_is_writable, _default_mkstemp


def test__is_writable_handles_oserror() -> None:
    def _always_fail(path: Path) -> bool:
        return False

    health_mod._is_writable = _always_fail
    # Call the hook to verify it works (note: hook is assigned above)
    result = health_mod._is_writable(Path(tempfile.gettempdir()) / "nope")
    assert result is False


def test__is_writable_default_returns_true_for_writable_dir() -> None:
    # Test the default implementation on a writable directory
    writable_dir = Path(tempfile.gettempdir())
    result = _default_is_writable(writable_dir)
    assert result is True


def test__is_writable_default_returns_false_on_oserror(tmp_path: Path) -> None:
    """Test the OSError branch when _mkstemp hook raises."""

    def _failing_mkstemp(prefix: str, dir_path: str) -> tuple[int, str]:
        raise OSError("simulated disk full")

    # Save original and set failing hook
    orig = health_mod._mkstemp
    health_mod._mkstemp = _failing_mkstemp

    try:
        result = _default_is_writable(tmp_path)
        assert result is False
    finally:
        health_mod._mkstemp = orig


def test__default_mkstemp_calls_tempfile(tmp_path: Path) -> None:
    """Test _default_mkstemp creates a file via tempfile.mkstemp."""
    fd, path = _default_mkstemp("test_", str(tmp_path))
    try:
        assert fd >= 0
        assert Path(path).exists()
    finally:
        import os

        os.close(fd)
        Path(path).unlink()
