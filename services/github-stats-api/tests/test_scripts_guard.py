from __future__ import annotations

from pathlib import Path

import pytest
from scripts import _test_hooks
from scripts import guard as guard_mod


def test_guard_main_runs_successfully() -> None:
    """Test that guard main function runs and returns valid exit code."""
    rc = guard_mod.main(None)
    assert rc == 0


def test_guard_main_with_verbose_flag() -> None:
    """Test guard main with verbose flag exercises that code path."""
    project_root = Path(__file__).resolve().parents[1]
    rc = guard_mod.main(["--root", str(project_root), "--verbose"])
    assert rc == 0


def test_guard_main_with_unknown_flag() -> None:
    """Test guard main with unknown flag to exercise else branch."""
    rc = guard_mod.main(["--unknown-flag"])
    assert rc == 0


def test_guard_find_monorepo_root_with_hook() -> None:
    """Test _find_monorepo_root with injected is_dir hook."""
    original_hook = _test_hooks.is_dir
    fake_root = Path("/fake/monorepo")

    def fake_is_dir(path: Path) -> bool:
        return path == fake_root / "libs"

    try:
        _test_hooks.is_dir = fake_is_dir
        start = fake_root / "services" / "github-stats-api"
        result = guard_mod._find_monorepo_root(start)
        assert result == fake_root, f"Expected {fake_root}, got {result}"
    finally:
        _test_hooks.is_dir = original_hook


def test_guard_find_monorepo_root_raises_when_not_found() -> None:
    """Test _find_monorepo_root raises when libs not found."""
    original_hook = _test_hooks.is_dir

    def fake_is_dir_never(path: Path) -> bool:
        return False

    try:
        _test_hooks.is_dir = fake_is_dir_never
        with pytest.raises(RuntimeError, match=r"monorepo root.*not found"):
            guard_mod._find_monorepo_root(Path("/some/random/path"))
    finally:
        _test_hooks.is_dir = original_hook


def test_guard_main_block_execution() -> None:
    """Test guard __main__ block via compile+exec."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "guard.py"
    code = script_path.read_text(encoding="utf-8")
    globals_dict = {"__name__": "__main__", "__file__": str(script_path)}
    with pytest.raises(SystemExit):
        exec(compile(code, str(script_path), "exec"), globals_dict, globals_dict)
