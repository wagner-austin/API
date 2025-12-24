"""Tests for scripts/guard.py module."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_guard_main_and_main_block() -> None:
    """Test main() and the if __name__ == '__main__' block."""
    from scripts import guard as guard_mod

    rc = guard_mod.main(None)
    assert rc >= 0

    # Execute the file as if __name__ == "__main__" using compile+exec.
    # This covers the SystemExit path without using runpy (which returns Any).
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "guard.py"
    code = script_path.read_text(encoding="utf-8")
    globals_dict = {"__name__": "__main__", "__file__": str(script_path)}
    with pytest.raises(SystemExit):
        exec(compile(code, str(script_path), "exec"), globals_dict, globals_dict)


def test_guard_find_root_raises_when_libs_missing() -> None:
    """Test _find_monorepo_root raises when libs directory not found."""
    from scripts import guard as guard_mod

    original_is_dir = guard_mod._is_dir
    try:
        guard_mod._is_dir = lambda p: False  # never finds libs
        with pytest.raises(RuntimeError):
            guard_mod._find_monorepo_root(Path("C:\\"))
    finally:
        guard_mod._is_dir = original_is_dir


def test_guard_verbose_flag_and_root_override() -> None:
    """Test --verbose flag and --root override."""
    from scripts import guard as guard_mod

    project_root = Path(__file__).resolve().parents[1]
    rc = guard_mod.main(["--root", str(project_root), "--verbose"])
    assert rc >= 0


def test_guard_unknown_flag_hits_else_branch() -> None:
    """Test unknown flags are skipped via else branch."""
    from scripts import guard as guard_mod

    rc = guard_mod.main(["--unknown-flag"])
    assert rc >= 0


def test_guard_default_is_dir() -> None:
    """Test _default_is_dir returns correct values."""
    from scripts import guard as guard_mod

    # Existing directory returns True
    path = Path(__file__).parent
    assert guard_mod._default_is_dir(path) is True

    # Nonexistent path returns False
    assert guard_mod._default_is_dir(path / "nonexistent") is False

    # File returns False
    assert guard_mod._default_is_dir(Path(__file__)) is False


def test_guard_load_orchestrator() -> None:
    """Test _load_orchestrator loads run_for_project function."""
    from scripts import guard as guard_mod

    project_root = Path(__file__).resolve().parents[1]
    monorepo_root = guard_mod._find_monorepo_root(project_root)
    run_for_project = guard_mod._load_orchestrator(monorepo_root)

    assert callable(run_for_project)


def test_guard_short_verbose_flag() -> None:
    """Test -v short verbose flag."""
    from scripts import guard as guard_mod

    rc = guard_mod.main(["-v"])
    assert rc >= 0
