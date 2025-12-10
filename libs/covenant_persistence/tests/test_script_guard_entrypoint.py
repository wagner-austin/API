"""Tests for scripts.guard entrypoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.guard import _find_monorepo_root, main


def test_guard_entrypoint_runs_as_main() -> None:
    """Guard module can be run as __main__."""
    # Ensure a clean module state to avoid runpy runtime warning
    if "scripts.guard" in sys.modules:
        del sys.modules["scripts.guard"]
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.guard", run_name="__main__")
    code = exc.value.code if isinstance(exc.value.code, int) else 0
    assert code in (0, 2)


def test_main_with_verbose_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """Guard main runs with verbose flag."""
    rc = main(["--verbose"])
    captured = capsys.readouterr()
    # Verbose mode produces output ending with exit code message
    assert captured.out.endswith(f"guard_exit_code code={rc}\n")
    assert rc in (0, 2)


def test_main_with_root_override() -> None:
    """Guard main runs with root override."""
    # Use current project root as override target
    project_root = Path(__file__).resolve().parents[1]
    rc = main(["--root", str(project_root)])
    assert rc in (0, 2)


def test_main_with_short_verbose_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """Guard main runs with short verbose flag."""
    rc = main(["-v"])
    captured = capsys.readouterr()
    # Short verbose flag produces output ending with exit code message
    assert captured.out.endswith(f"guard_exit_code code={rc}\n")
    assert rc in (0, 2)


def test_main_with_unknown_arg() -> None:
    """Guard main ignores unknown arguments."""
    # Unknown args are ignored
    rc = main(["--unknown-flag"])
    assert rc in (0, 2)


def test_find_monorepo_root_raises_when_not_found(tmp_path: Path) -> None:
    """_find_monorepo_root raises RuntimeError when root not found."""
    # Create a path that has no parent with 'libs' directory
    with pytest.raises(RuntimeError) as exc_info:
        _find_monorepo_root(tmp_path)
    assert "monorepo root with 'libs' directory not found" in str(exc_info.value)
