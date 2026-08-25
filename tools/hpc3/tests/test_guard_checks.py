"""Tests for the guard script.

The guard is what enforces the rest of the standards, so it is exercised
both end-to-end -- a subprocess run against a tree with real violations --
and in-process against fakes for every branch of its own plumbing.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from scripts import _test_hooks
from scripts.guard import _find_monorepo_root
from scripts.guard import main as guard_main


def _write(path: Path, text: str) -> None:
    """Write text, creating parent directories.

    Args:
        path: File to write.
        text: Contents to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _project_root() -> Path:
    """Locate this project's root.

    Returns:
        The ``hpc3`` tool directory, two levels up from this file.
    """
    return Path(__file__).resolve().parents[1]


class _FakeIsDir:
    def __call__(self, path: Path) -> bool:
        return True


class _FakeLoader:
    def __call__(self, monorepo_root: Path) -> _test_hooks.RunForProjectProtocol:
        def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
            return 0

        return _run_for_project


def _with_fakes(args: list[str] | None) -> int:
    """Run the guard's main with its plumbing faked.

    Args:
        args: Arguments to pass, or None to read the process arguments.

    Returns:
        The guard's exit code.
    """
    original_is_dir = _test_hooks.is_dir
    original_load_orchestrator = _test_hooks.load_orchestrator
    _test_hooks.is_dir = _FakeIsDir()
    _test_hooks.load_orchestrator = _FakeLoader()
    try:
        return guard_main(args)
    finally:
        _test_hooks.is_dir = original_is_dir
        _test_hooks.load_orchestrator = original_load_orchestrator


def test_guard_detects_violations(tmp_path: Path) -> None:
    src = tmp_path / "src" / "hpc3"
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
        "except Exception:\n"
        "    a = 1\n"
    )
    _write(bad, code)

    scripts_errors = tmp_path / "scripts" / "errors.py"
    tests_errors = tmp_path / "tests" / "errors.py"
    _write(scripts_errors, "class AppError(Exception):\n    ...\n")
    _write(tests_errors, "class ErrorCode(Exception):\n    ...\n")

    project_root = _project_root()
    result = subprocess.run(
        [sys.executable, "-m", "scripts.guard", "--root", str(tmp_path)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    out = result.stdout + result.stderr

    assert result.returncode != 0
    assert "Guard checks failed" in out or "Guard rule summary" in out
    assert "local-errors-module" in out


def test_guard_main_entry_no_violations(tmp_path: Path) -> None:
    project_root = _project_root()
    result = subprocess.run(
        [sys.executable, "-m", "scripts.guard", "--root", str(tmp_path)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert _with_fakes(["--root", str(tmp_path)]) == 0


def test_main_reads_the_process_arguments_when_given_none(tmp_path: Path) -> None:
    original = list(sys.argv)
    sys.argv[:] = ["prog", "--root", str(tmp_path)]
    try:
        assert _with_fakes(None) == 0
    finally:
        sys.argv[:] = original


def test_main_reports_the_exit_code_when_verbose(tmp_path: Path) -> None:
    assert _with_fakes(["--root", str(tmp_path), "--verbose"]) == 0


def test_main_skips_tokens_it_does_not_know(tmp_path: Path) -> None:
    assert _with_fakes(["bogus", "--root", str(tmp_path)]) == 0


def test_main_ignores_a_root_flag_with_no_value(tmp_path: Path) -> None:
    assert _with_fakes(["--root"]) == 0


def test_find_monorepo_root_raises_when_no_ancestor_has_libs() -> None:
    class _NeverDir:
        def __call__(self, path: Path) -> bool:
            return False

    original = _test_hooks.is_dir
    _test_hooks.is_dir = _NeverDir()
    try:
        with pytest.raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
            _find_monorepo_root(Path("somewhere"))
    finally:
        _test_hooks.is_dir = original


def test_real_is_dir_distinguishes_directories_from_files(tmp_path: Path) -> None:
    file_path = tmp_path / "f.txt"
    file_path.write_text("x", encoding="utf-8")
    assert _test_hooks.is_dir(tmp_path) is True
    assert _test_hooks.is_dir(file_path) is False


def test_real_get_script_path_round_trips_and_rejects_the_unset_state(tmp_path: Path) -> None:
    original = _test_hooks._SCRIPT_PATH
    _test_hooks._SCRIPT_PATH = None
    try:
        with pytest.raises(RuntimeError, match="Script path not set"):
            _test_hooks.get_script_path()
    finally:
        if original is not None:
            _test_hooks.set_script_path(original)
    marker = tmp_path / "guard.py"
    _test_hooks.set_script_path(marker)
    try:
        assert _test_hooks.get_script_path() == marker
    finally:
        if original is not None:
            _test_hooks.set_script_path(original)


def test_real_load_orchestrator_runs_the_monorepo_guards(tmp_path: Path) -> None:
    monorepo_root = _find_monorepo_root(_project_root())
    run_for_project = _test_hooks.load_orchestrator(monorepo_root)
    assert run_for_project(monorepo_root=monorepo_root, project_root=tmp_path) == 0


def test_the_guard_runs_as_a_module(tmp_path: Path) -> None:
    """This is how the Makefile invokes it, so the build's path is tested."""
    original = list(sys.argv)
    sys.argv[:] = ["prog", "--root", str(tmp_path)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("scripts.guard", run_name="__main__")
        assert excinfo.value.code == 0
    finally:
        sys.argv[:] = original
