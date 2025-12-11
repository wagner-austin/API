from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pytest import raises


def _project_root() -> Path:
    # tests/ -> data-bank-api/
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


def test_guard_find_monorepo_root_raises_without_libs(tmp_path: Path) -> None:
    from scripts import guard as guard_mod

    # Save original hook
    original_is_dir = guard_mod._is_dir

    # Override hook to always return False for libs directory checks
    def _always_false(path: Path) -> bool:
        return False

    guard_mod._is_dir = _always_false

    try:
        with raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
            guard_mod._find_monorepo_root(tmp_path)
    finally:
        # Restore original hook
        guard_mod._is_dir = original_is_dir
