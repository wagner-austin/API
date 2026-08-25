from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.guard import main as guard_main

from scripts import _test_hooks


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _project_root() -> Path:
    # tests/ -> Model-Trainer/ -> services/ -> repo root
    # The Model-Trainer project root is the service directory.
    return Path(__file__).resolve().parents[1]


def test_guard_detects_violations(tmp_path: Path) -> None:
    # Arrange: create files with clear violations in a temporary tree under src/
    src = tmp_path / "src" / "model_trainer"
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

    # Add local error modules across directories to ensure guard covers src, scripts, and tests.
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
    # Verify centralized error guard scanned scripts and tests directories.
    assert "local-errors-module" in out


def test_guard_main_entry_no_violations(tmp_path: Path) -> None:
    # Invoked as `-m` from the project root, which is what the Makefile does.
    # Running the file BY PATH instead puts scripts/ on sys.path[0] rather than
    # the project root, so `from scripts import _test_hooks` inside guard.py
    # could only resolve against an INSTALLED top-level `scripts` package. That
    # package was shipped by every one of the 40 pyprojects, so the copy it
    # found was whichever one installed last -- and on a real wheel install it
    # was the wrong one. `-m` reproduces the only invocation anything uses.
    project_root = _project_root()

    result = subprocess.run(
        [sys.executable, "-m", "scripts.guard", "--root", str(tmp_path)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0

    # Cover the main() entry point by calling it directly with args instead of
    # using runpy.run_path with sys.argv patching. Fakes are installed by
    # rebinding the guard hooks and restored afterwards.
    class _FakeIsDir:
        def __call__(self, path: Path) -> bool:
            return True

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> _test_hooks.RunForProjectProtocol:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 0

            return _run_for_project

    original_is_dir = _test_hooks.is_dir
    original_load_orchestrator = _test_hooks.load_orchestrator
    _test_hooks.is_dir = _FakeIsDir()
    _test_hooks.load_orchestrator = _FakeLoader()
    try:
        code = guard_main(["--root", str(tmp_path)])
    finally:
        _test_hooks.is_dir = original_is_dir
        _test_hooks.load_orchestrator = original_load_orchestrator
    assert code == 0
