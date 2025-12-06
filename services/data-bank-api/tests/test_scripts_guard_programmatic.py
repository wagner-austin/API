from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from scripts.guard import main as guard_main


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_guard_main_ok_programmatic(tmp_path: Path) -> None:
    # No files present -> no violations
    code = guard_main(["--root", str(tmp_path)])
    assert code == 0


def test_guard_main_detects_violation_programmatic(tmp_path: Path) -> None:
    # Create a file with disallowed typing/exception patterns
    bad = tmp_path / "src" / "bad.py"
    code = (
        "from typing import Any\n"
        "x: Any = 1  # type: ignore\n"
        "from typing import cast\n"
        "y = cast(int, 1)\n"
        "try:\n"
        "    1/0\n"
        "except Exception as exc:\n"
        "    raise RuntimeError('fail') from exc\n"
    )
    _write(bad, code)

    rc = guard_main(["--root", str(tmp_path)])
    assert rc != 0


def test_guard_main_entry_runs_as_script() -> None:
    # Executing the module as a script should exit with code 0 for this repo
    import sys

    sys.modules.pop("scripts.guard", None)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.guard", run_name="__main__")
    code_obj = exc.value.code
    assert code_obj == 0


def test_guard_main_skips_unknown_args() -> None:
    # Provide unknown flags to cover the non --root branch in the parser loop
    rc = guard_main(["--bogus", "foo"])
    assert rc == 0


def test_guard_main_verbose_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Test both -v and --verbose flags
    rc1 = guard_main(["-v", "--root", str(tmp_path)])
    assert rc1 == 0
    out1 = capsys.readouterr()
    # Verbose mode produces output; verify non-empty stdout
    assert out1.out

    rc2 = guard_main(["--verbose", "--root", str(tmp_path)])
    assert rc2 == 0
    out2 = capsys.readouterr()
    # Verbose mode produces output; verify non-empty stdout
    assert out2.out
