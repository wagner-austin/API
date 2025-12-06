from __future__ import annotations

import runpy
import sys

import pytest


def test_guard_entrypoint_runs_as_main() -> None:
    # Ensure a clean module state to avoid runpy runtime warning
    if "scripts.guard" in sys.modules:
        del sys.modules["scripts.guard"]
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.guard", run_name="__main__")
    code = exc.value.code if isinstance(exc.value.code, int) else 0
    assert code in (0, 2)
