from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


def test_guard_script_runs_as_main() -> None:
    # Stub orchestrator so the script's dynamic import resolves to our stub
    orch = ModuleType("monorepo_guards.orchestrator")

    def run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        return 0

    object.__setattr__(orch, "run_for_project", run_for_project)

    pkg = ModuleType("monorepo_guards")
    object.__setattr__(pkg, "orchestrator", orch)
    sys.modules["monorepo_guards"] = pkg
    sys.modules["monorepo_guards.orchestrator"] = orch

    # Ensure a clean import state to avoid runpy warning about preloaded modules
    sys.modules.pop("scripts.guard", None)
    sys.modules.pop("scripts", None)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.guard", run_name="__main__")
    assert exc.value.code == 0
