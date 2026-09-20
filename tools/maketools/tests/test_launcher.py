"""``scripts/run.py``: the path it inserts and the exit code it forwards."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts import run

from maketools.cli import repository_root


def test_the_source_roots_are_this_package_and_platform_core() -> None:
    root = repository_root()
    assert (
        root / "tools" / "maketools" / "src",
        root / "libs" / "platform_core" / "src",
    ) == run.SOURCE_ROOTS


def test_the_launcher_puts_both_roots_on_the_path_when_they_are_absent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The suite already has both roots on the path; the launcher's whole job
    # is the case where they are not, so that case is arranged and restored.
    roots = [str(root) for root in run.SOURCE_ROOTS]
    saved = list(sys.path)
    sys.path[:] = [entry for entry in sys.path if entry not in roots]
    try:
        assert run.main(["no-such-command"]) == 1
        assert sys.path[:2] == roots
    finally:
        sys.path[:] = saved
    assert capsys.readouterr().err.startswith("MAKETOOLS_USAGE: unknown command")


def test_the_main_guard_executes_in_process(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    original = list(sys.argv)
    sys.argv[:] = ["run.py", "guard"]
    before = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        with pytest.raises(SystemExit) as exited:
            runpy.run_path(str(run.__file__), run_name="__main__")
    finally:
        os.chdir(before)
        sys.argv[:] = original
    assert exited.value.code == 0
    assert capsys.readouterr().out.startswith("guard: not applicable")
