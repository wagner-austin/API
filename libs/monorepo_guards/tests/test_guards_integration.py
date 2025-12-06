from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest


def _project_root() -> Path:
    # tests/ -> monorepo_guards/
    return Path(__file__).resolve().parents[1]


# Ensure the project root (which contains the local "scripts" package) is importable
_ROOT = _project_root()
sys.path.insert(0, str(_ROOT))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_project_guard_detects_violations(tmp_path: Path) -> None:
    root = tmp_path
    pkg = root / "src" / "monorepo_guards"
    bad = pkg / "bad.py"

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


def test_project_guard_verbose_success(tmp_path: Path) -> None:
    project_root = _project_root()
    guard_path = project_root / "scripts" / "guard.py"

    result = subprocess.run(
        [sys.executable, str(guard_path), "--root", str(tmp_path), "-v"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0
    assert "Guard checks passed: no violations found." in out
    assert "guard_exit_code code=0" in out


def test_guard_import_main_no_violations(tmp_path: Path) -> None:
    # Importing the guard and calling main should succeed and exercise script lines.
    from scripts.guard import main as guard_main

    rc = guard_main(["--root", str(tmp_path)])
    assert rc == 0


def test_guard_import_main_verbose(tmp_path: Path) -> None:
    from scripts.guard import main as guard_main

    rc = guard_main(["--root", str(tmp_path), "--verbose"])  # covers verbose branch
    assert rc == 0


def test_guard_import_main_handles_unknown_flag(tmp_path: Path) -> None:
    from scripts.guard import main as guard_main

    rc = guard_main(["--root", str(tmp_path), "--unknown-flag", "--verbose"])
    assert rc == 0


def test_guard_find_monorepo_root_raises_without_libs(tmp_path: Path) -> None:
    from scripts import guard

    with pytest.raises(RuntimeError, match="monorepo root"):
        guard._find_monorepo_root(tmp_path)


def test_guard_module_main_entry_via_runpy() -> None:
    libs_dir = _project_root().parent
    libs_str = str(libs_dir)
    sys.path = [p for p in sys.path if p != libs_str]

    # Ensure a clean module state so run_module executes from scratch.
    sys.modules.pop("scripts.guard", None)
    sys.modules.pop("scripts", None)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("scripts.guard", run_name="__main__")
    assert excinfo.value.code == 0
