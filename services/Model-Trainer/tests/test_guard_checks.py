from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.guard import main as guard_main

from model_trainer.core import _test_hooks


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
    guard_path = project_root / "scripts" / "guard.py"

    result = subprocess.run(
        [sys.executable, str(guard_path), "--root", str(tmp_path)],
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

    # Cover __main__ entry point by calling main() directly with args
    # instead of using runpy.run_path with sys.argv patching
    # Set up hooks to make guard use tmp_path as root
    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> _test_hooks.RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 0

            return _run_for_project

    _test_hooks.guard_find_monorepo_root = _FakeFindRoot()
    _test_hooks.guard_load_orchestrator = _FakeLoader()

    code = guard_main(["--root", str(tmp_path)])
    assert code == 0
