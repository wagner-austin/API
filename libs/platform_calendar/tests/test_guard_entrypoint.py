"""Tests for scripts/guard.py entrypoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

# Import the module functions directly for testing
import scripts.guard as guard_module


def test_guard_entrypoint_runs_as_main() -> None:
    """Ensure guard script can be run as module."""
    # Ensure a clean module state to avoid runpy runtime warning
    if "scripts.guard" in sys.modules:
        del sys.modules["scripts.guard"]
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.guard", run_name="__main__")
    code = exc.value.code if isinstance(exc.value.code, int) else 0
    # Exit code 0 = success, 2 = guard violations found (both are valid runs)
    assert code in (0, 2)


def test_main_with_verbose_flag() -> None:
    """Test running with verbose flag."""
    result = guard_module.main(["--verbose"])
    # Should succeed (0) or find violations (2)
    assert result in (0, 2)


def test_main_with_v_flag() -> None:
    """Test running with -v flag."""
    result = guard_module.main(["-v"])
    assert result in (0, 2)


def test_main_with_root_override(tmp_path: Path) -> None:
    """Test running with --root override."""
    # Create a fake project structure
    (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'test'\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    # Run with root override - this will work because it finds the real monorepo root
    result = guard_module.main(["--root", str(tmp_path)])
    assert result in (0, 2)


def test_find_monorepo_root_not_found() -> None:
    """Test error when monorepo root is not found."""
    # Save original _is_dir
    original_is_dir = guard_module._is_dir

    def fake_is_dir(p: Path) -> bool:
        # Always return False - no libs directory exists
        return False

    guard_module._is_dir = fake_is_dir
    try:
        with pytest.raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
            guard_module._find_monorepo_root(Path("/some/path"))
    finally:
        guard_module._is_dir = original_is_dir


def test_default_is_dir() -> None:
    """Test default _is_dir function."""
    # Test with real paths
    current_dir = Path.cwd()
    result = guard_module._default_is_dir(current_dir)
    assert result is True

    nonexistent = Path("/nonexistent/path/12345")
    result = guard_module._default_is_dir(nonexistent)
    assert result is False


def test_main_with_unknown_args() -> None:
    """Test that unknown args are ignored."""
    result = guard_module.main(["--unknown-flag", "some-value"])
    assert result in (0, 2)
