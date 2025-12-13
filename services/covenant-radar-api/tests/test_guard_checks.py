"""Tests for guard script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pytest import raises

from scripts import guard as guard_mod


def _project_root() -> Path:
    # tests/ -> covenant-radar-api/
    return Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_guard_detects_violations(tmp_path: Path) -> None:
    """Test guard detects typing and suppression violations."""
    root = tmp_path
    src = root / "src"
    bad = src / "bad.py"

    any_kw = "An" + "y"
    ti = "# " + "type" + ": " + "ignore"
    code = (
        f"from typing import {any_kw}\n"
        f"x: {any_kw} = 1  {ti}\n"
        "from typing import cast\n"
        "y = cast(int, 1)\n"
        "import contextlib\n"
        "with contextlib.suppress(Exception):\n"
        "    pass\n"
        "try:\n"
        "    1/0\n"
        "except Exception as exc:\n"
        "    raise RuntimeError('fail') from exc\n"
    )
    _write(bad, code)

    project_root = _project_root()
    guard_path = project_root / "scripts" / "guard.py"

    result = subprocess.run(
        [sys.executable, str(guard_path), "--root", str(root)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    out = result.stdout + result.stderr

    assert result.returncode != 0
    assert "Guard rule summary" in out
    assert "Guard checks failed" in out


def test_guard_main_entry_no_violations(tmp_path: Path) -> None:
    """Test guard passes on empty directory."""
    project_root = _project_root()
    guard_path = project_root / "scripts" / "guard.py"

    result = subprocess.run(
        [sys.executable, str(guard_path), "--root", str(tmp_path)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_guard_find_monorepo_root_raises_without_libs(tmp_path: Path) -> None:
    """Test _find_monorepo_root raises when libs directory not found."""

    # Override hook to always return False for libs directory checks
    def _always_false(path: Path) -> bool:
        return False

    guard_mod._is_dir = _always_false

    with raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
        guard_mod._find_monorepo_root(tmp_path)


def test_guard_verbose_flag(tmp_path: Path) -> None:
    """Test guard --verbose flag outputs exit code."""
    project_root = _project_root()
    guard_path = project_root / "scripts" / "guard.py"

    result = subprocess.run(
        [sys.executable, str(guard_path), "--root", str(tmp_path), "--verbose"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    out = result.stdout

    assert result.returncode == 0
    assert "guard_exit_code" in out


def test_guard_short_verbose_flag(tmp_path: Path) -> None:
    """Test guard -v flag outputs exit code."""
    project_root = _project_root()
    guard_path = project_root / "scripts" / "guard.py"

    result = subprocess.run(
        [sys.executable, str(guard_path), "--root", str(tmp_path), "-v"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    out = result.stdout

    assert result.returncode == 0
    assert "guard_exit_code" in out


def test_guard_main_direct_call(tmp_path: Path) -> None:
    """Test calling main() directly for coverage."""
    rc = guard_mod.main(["--root", str(tmp_path)])
    assert rc == 0


def test_guard_main_verbose_direct_call(tmp_path: Path) -> None:
    """Test calling main() with verbose flag for coverage."""
    rc = guard_mod.main(["--root", str(tmp_path), "--verbose"])
    assert rc == 0


def test_guard_main_short_verbose_direct_call(tmp_path: Path) -> None:
    """Test calling main() with -v flag for coverage."""
    rc = guard_mod.main(["--root", str(tmp_path), "-v"])
    assert rc == 0


def test_guard_default_is_dir_covers_production_path() -> None:
    """Test _default_is_dir returns True for existing directory."""
    project_root = _project_root()
    result = guard_mod._default_is_dir(project_root)
    assert result is True


def test_guard_default_is_dir_returns_false_for_nonexistent() -> None:
    """Test _default_is_dir returns False for nonexistent path."""
    result = guard_mod._default_is_dir(Path("/nonexistent/path/that/does/not/exist"))
    assert result is False


def test_guard_find_monorepo_root_finds_libs_directory() -> None:
    """Test _find_monorepo_root finds the monorepo root."""
    project_root = _project_root()
    # Reset hook to default before test
    guard_mod._is_dir = guard_mod._default_is_dir
    root = guard_mod._find_monorepo_root(project_root)
    # Should find the root with libs directory
    assert (root / "libs").is_dir()


def test_guard_load_orchestrator_returns_callable() -> None:
    """Test _load_orchestrator returns a callable function."""
    project_root = _project_root()
    guard_mod._is_dir = guard_mod._default_is_dir
    monorepo_root = guard_mod._find_monorepo_root(project_root)
    run_for_project = guard_mod._load_orchestrator(monorepo_root)
    assert callable(run_for_project)


def test_guard_main_with_unknown_arg(tmp_path: Path) -> None:
    """Test calling main() with an unknown argument (covers line 61)."""
    rc = guard_mod.main(["--root", str(tmp_path), "--unknown-arg"])
    assert rc == 0


def test_guard_name_main_block() -> None:
    """Test the if __name__ == '__main__' block (covers line 71)."""
    import runpy

    # Run guard.py as __main__ - the SystemExit will be raised
    project_root = _project_root()
    guard_path = str(project_root / "scripts" / "guard.py")

    # Create a temporary empty directory to test against
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Modify sys.argv to pass --root to guard
        import sys

        old_argv = sys.argv
        sys.argv = ["guard.py", "--root", tmp_dir]

        with raises(SystemExit) as exc_info:
            runpy.run_path(guard_path, run_name="__main__")

        sys.argv = old_argv

        # Should exit with 0 (no violations in empty dir)
        assert exc_info.value.code == 0
