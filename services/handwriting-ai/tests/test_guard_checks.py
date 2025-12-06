from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

from platform_core.logging import get_logger


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _project_root() -> Path:
    # tests/ -> handwriting-ai/ -> services/ -> repo root
    # Project root is the parent of "tests".
    return Path(__file__).resolve().parents[1]


def test_guard_detects_violations(tmp_path: Path) -> None:
    src = tmp_path / "src" / "pkg"
    bad_typing = src / "t.py"
    bad_logging = src / "l.py"
    bad_exceptions = src / "e.py"

    any_kw = "An" + "y"
    ti = "# " + "type" + ": " + "ignore"
    typing_code = (
        f"from typing import {any_kw}\n"
        f"x: {any_kw} = 1  {ti}\n"
        "from typing import cast\n"
        "y = cast(int, 1)\n"
    )
    _write(bad_typing, typing_code)

    log_mod = "log" + "ging"
    basic_cfg = ".".join([log_mod, "basic" + "Config"])
    pr = "pri" + "nt"
    logging_code = f"import {log_mod}\n{pr}('x')\n{basic_cfg}(level=10)\n"
    _write(bad_logging, logging_code)

    exceptions_code = (
        "import contextlib\n"
        "with contextlib.suppress(Exception):\n"
        "    pass\n\n"
        "try:\n"
        "    1/0\n"
        "except Exception as exc:\n"
        "    raise RuntimeError('fail') from exc\n"
    )
    _write(bad_exceptions, exceptions_code)

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

    # Cover __main__ entry point
    sys.modules.pop("scripts.guard", None)
    sys.path.insert(0, str(project_root / "scripts"))
    try:
        runpy.run_module("scripts.guard", run_name="__main__")
    except SystemExit as exc:
        get_logger("handwriting_ai.tests").info("guard_main_exit code=%s", exc.code)
        code = exc.code
        assert code == 0 or code == "0"
