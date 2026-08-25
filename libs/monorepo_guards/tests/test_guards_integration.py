"""End-to-end: the guard as a real process, against a real tree.

Everything else about the shim is covered without spawning anything --
``test_shim.py`` for the argument handling, ``test_guard_shim.py`` for this
package's own bootstrap. What only a subprocess can establish is that
``python -m scripts.guard`` works from a clean interpreter, which is the
exact invocation every Makefile in the monorepo uses.

That distinction matters here more than it looks. Running the shim BY PATH
instead of with ``-m`` puts ``scripts/`` on ``sys.path[0]`` rather than the
package root, and the import inside it then resolves against an installed
top-level ``scripts`` -- which is how a test in this repo passed for months
while depending on whichever of 40 packages had installed last.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    """Locate this package's root.

    Returns:
        The ``monorepo_guards`` directory, one level above ``tests``.
    """
    return Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> None:
    """Write text, creating parent directories.

    Args:
        path: File to write.
        text: Contents to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_guard(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the guard the way the Makefile does.

    Args:
        *args: Arguments to pass after the module name.

    Returns:
        The completed process, with output captured.
    """
    return subprocess.run(
        [sys.executable, "-m", "scripts.guard", *args],
        cwd=str(_project_root()),
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_tree_with_violations_fails_the_process(tmp_path: Path) -> None:
    banned = "An" + "y"
    ignore = "# " + "type" + ": " + "ignore"
    _write(
        tmp_path / "src" / "monorepo_guards" / "bad.py",
        f"from typing import {banned}\n"
        f"x: {banned} = 1  {ignore}\n"
        "from typing import cast\n"
        "y = cast(int, 1)\n"
        "import contextlib\n"
        "with contextlib.suppress(Exception):\n"
        "    pass\n"
        "try:\n"
        "    1/0\n"
        "except Exception as exc:\n"
        "    raise RuntimeError('fail') from exc\n",
    )

    result = _run_guard("--root", str(tmp_path))
    out = result.stdout + result.stderr

    assert result.returncode == 2
    assert "Guard rule summary" in out
    assert "Guard checks failed" in out


def test_a_clean_tree_passes_the_process_and_reports_verbosely(tmp_path: Path) -> None:
    result = _run_guard("--root", str(tmp_path), "-v")
    out = result.stdout + result.stderr

    assert result.returncode == 0
    assert "Guard checks passed: no violations found." in out
    assert "guard_exit_code code=0" in out


__all__ = [
    "test_a_clean_tree_passes_the_process_and_reports_verbosely",
    "test_a_tree_with_violations_fails_the_process",
]
