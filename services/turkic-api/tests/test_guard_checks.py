"""Tests for guard script violations detection.

These tests exercise the guard violation detection through the guard.main()
function rather than subprocess to ensure proper module resolution.
"""

from __future__ import annotations

from pathlib import Path

from scripts import guard


def _write(path: Path, text: str) -> None:
    """Write text to a file, creating parent directories if needed.

    Args:
        path: File path to write to.
        text: Text content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_guard_detects_violations(tmp_path: Path) -> None:
    """Test guard.main detects violations and returns non-zero exit code."""
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

    rc = guard.main(["--root", str(root)])
    assert rc != 0


def test_guard_main_entry_no_violations(tmp_path: Path) -> None:
    """Test guard.main returns 0 when no violations found."""
    rc = guard.main(["--root", str(tmp_path)])
    assert rc == 0
