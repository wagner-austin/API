from __future__ import annotations

import logging
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_guard_main_entry_no_violations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    # Cover __main__ entry point by executing the script directly.
    # Monkeypatch sys.argv to use tmp_path as root (avoids full project scan timeout)
    monkeypatch.setattr("sys.argv", ["guard.py", "--root", str(tmp_path)])
    try:
        runpy.run_path(str(guard_path), run_name="__main__")
    except SystemExit as exc:
        logging.getLogger("model_trainer.tests").info("guard_main_exit code=%s", exc.code)
        code = exc.code
        assert code == 0 or code == "0"
