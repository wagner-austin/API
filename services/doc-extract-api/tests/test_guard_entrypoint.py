"""Tests for scripts.guard entrypoint."""

from __future__ import annotations

import runpy
import sys

import pytest


def test_guard_entrypoint_runs_as_main() -> None:
    """Running as a module should exit with 0 or 2 depending on checks."""
    module_name = "scripts.guard"
    saved_module = sys.modules.pop(module_name, None)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module(module_name, run_name="__main__", alter_sys=False)

    if saved_module is not None:
        sys.modules[module_name] = saved_module

    err = exc.value
    code: int = err.code if isinstance(err.code, int) else 0
    assert code in (0, 2)
