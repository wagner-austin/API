"""Tests for guard script functionality."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import scripts._test_hooks as test_hooks
import scripts.guard as guard_mod
from pytest import raises


def test_guard_main_entry_no_violations(tmp_path: Path) -> None:
    """Test guard.main with no violations in an empty directory."""
    rc = guard_mod.main(["--root", str(tmp_path)])
    assert rc in (0, 2)


def test_guard_main_unknown_flag_is_ignored(tmp_path: Path) -> None:
    """Test guard.main ignores unknown flags."""
    rc = guard_mod.main(["--root", str(tmp_path), "ignored-flag"])
    assert rc in (0, 2)


def test_guard_main_verbose_flag_prints_exit_code(tmp_path: Path) -> None:
    """Test guard.main with verbose flag prints exit code."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        rc = guard_mod.main(["--root", str(tmp_path), "--verbose"])
        output = sys.stdout.getvalue()
        assert rc in (0, 2)
        assert f"guard_exit_code code={rc}\n" in output
    finally:
        sys.stdout = old_stdout


def test_guard_find_monorepo_root_raises_without_libs(tmp_path: Path) -> None:
    """Test _find_monorepo_root raises when libs directory not found."""
    original_is_dir = test_hooks.is_dir

    def _fake_is_dir(path: Path) -> bool:
        return False

    test_hooks.is_dir = _fake_is_dir

    with raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
        guard_mod._find_monorepo_root(tmp_path)

    test_hooks.is_dir = original_is_dir
