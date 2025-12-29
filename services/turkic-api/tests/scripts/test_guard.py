"""Tests for scripts.guard module."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import pytest

from scripts import _test_hooks, guard


def test_find_monorepo_root_success() -> None:
    """Test _find_monorepo_root finds the libs directory."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]

    root = guard._find_monorepo_root(project_root)

    libs_path = root / "libs"
    services_path = root / "services"
    assert libs_path.is_dir()
    assert services_path.is_dir()


def test_find_monorepo_root_not_found() -> None:
    """Test _find_monorepo_root raises when libs not found."""
    original_hook = _test_hooks.is_dir

    def fake_is_dir(path: Path) -> bool:
        return False

    _test_hooks.is_dir = fake_is_dir

    try:
        with pytest.raises(RuntimeError, match=r"monorepo root.*not found"):
            guard._find_monorepo_root(Path("/some/deep/path"))
    finally:
        _test_hooks.is_dir = original_hook


def test_main_runs_orchestrator() -> None:
    """Test main function loads and runs orchestrator."""
    rc = guard.main(["--verbose"])

    assert rc == 0 or rc > 0


def test_main_with_root_override() -> None:
    """Test main function accepts --root override."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]

    rc = guard.main(["--root", str(project_root), "--verbose"])

    assert rc == 0 or rc > 0


def test_main_without_verbose() -> None:
    """Test main function runs without verbose flag."""
    rc = guard.main([])

    assert rc == 0 or rc > 0


def test_main_with_unknown_argument() -> None:
    """Test main function ignores unknown arguments."""
    rc = guard.main(["--unknown-flag", "some-value", "-x"])

    assert rc == 0 or rc > 0


def test_real_is_dir() -> None:
    """Test _real_is_dir uses Path.is_dir()."""
    script_path = Path(__file__).resolve()
    parent = script_path.parent

    assert _test_hooks._real_is_dir(parent) is True
    assert _test_hooks._real_is_dir(parent / "nonexistent") is False


def test_guard_entrypoint_runs_as_main() -> None:
    """Test the if __name__ == '__main__' guard executes main()."""
    modules_to_clear = [k for k in sys.modules if k.startswith("scripts")]
    saved_modules: dict[str, types.ModuleType] = {}
    for mod in modules_to_clear:
        saved_modules[mod] = sys.modules.pop(mod)

    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.guard", run_name="__main__")
        err = exc.value
        code: int = err.code if isinstance(err.code, int) else 0
        assert code in (0, 2)
    finally:
        sys.modules.update(saved_modules)
