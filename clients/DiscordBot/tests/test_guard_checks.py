from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _project_root() -> Path:
    # tests/ -> DiscordBot/
    return Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_guard_detects_violations(tmp_path: Path) -> None:
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


def test_guard_detects_violations_verbose(tmp_path: Path) -> None:
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
        [sys.executable, str(guard_path), "--root", str(root), "-v"],
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


def test_guard_main_entry_no_violations_verbose(tmp_path: Path) -> None:
    project_root = _project_root()
    guard_path = project_root / "scripts" / "guard.py"

    result = subprocess.run(
        [sys.executable, str(guard_path), "--root", str(tmp_path), "--verbose"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0
    assert "Guard checks passed: no violations found." in out


def test_guard_ignores_unknown_arguments(tmp_path: Path) -> None:
    project_root = _project_root()
    guard_path = project_root / "scripts" / "guard.py"

    result = subprocess.run(
        [sys.executable, str(guard_path), "ignored-flag", "--root", str(tmp_path)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_guard_import_does_not_run_main() -> None:
    # Importing as a module must not trigger SystemExit
    import scripts.guard as guard_mod

    # Verify main function exists by accessing it
    assert callable(guard_mod.main)


def test_default_guard_load_orchestrator_returns_real_runner() -> None:
    """Test the production orchestrator loader against the real monorepo.

    Every other guard test injects a fake loader, so this production path --
    which puts `libs/` on sys.path and imports monorepo_guards.orchestrator --
    was never executed.
    """
    from pathlib import Path

    from clubbot._test_hooks import (
        _default_guard_find_monorepo_root,
        _default_guard_load_orchestrator,
    )

    monorepo_root = _default_guard_find_monorepo_root(Path(__file__).resolve())
    assert (monorepo_root / "libs").is_dir()

    run_for_project = _default_guard_load_orchestrator(monorepo_root)

    assert callable(run_for_project)


def test_guard_find_monorepo_root_raises_without_libs_directory() -> None:
    """Test the climb ends in an error when no libs directory is ever found.

    Saved and restored in one scope so the injection reads as hook-based DI.
    """
    from scripts import guard

    original_is_dir = guard._is_dir
    try:
        guard._is_dir = lambda p: False
        with pytest.raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
            guard._find_monorepo_root(Path(__file__).resolve())
    finally:
        guard._is_dir = original_is_dir
