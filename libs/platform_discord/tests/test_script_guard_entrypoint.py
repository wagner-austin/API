from __future__ import annotations

import runpy
import sys

import pytest


def test_guard_entrypoint_runs_as_main() -> None:
    if "scripts.guard" in sys.modules:
        del sys.modules["scripts.guard"]
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.guard", run_name="__main__")
    assert exc.value.code == 0
