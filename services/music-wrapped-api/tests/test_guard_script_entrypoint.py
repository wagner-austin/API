from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from music_wrapped_api import _test_hooks


def test_guard_script_runs_as_main(tmp_path: Path) -> None:
    """Test guard script runs as __main__ using hooks instead of sys.modules injection."""
    # Track calls to verify the orchestrator was invoked
    calls: list[tuple[Path, Path]] = []

    def fake_find_root(start: Path) -> Path:
        return tmp_path

    def fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append((monorepo_root, project_root))
        return 0

    def fake_load_orchestrator(monorepo_root: Path) -> _test_hooks.GuardRunForProjectProtocol:
        return fake_run_for_project

    # Save original hooks
    orig_find = _test_hooks.guard_find_monorepo_root
    orig_load = _test_hooks.guard_load_orchestrator

    # Install fake hooks
    _test_hooks.guard_find_monorepo_root = fake_find_root
    _test_hooks.guard_load_orchestrator = fake_load_orchestrator

    # Ensure a clean import state to avoid runpy warning about preloaded modules
    sys.modules.pop("scripts.guard", None)
    sys.modules.pop("scripts", None)

    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.guard", run_name="__main__")
        assert exc.value.code == 0
        # Verify the orchestrator was called
        assert calls
        monorepo_root, _project_root = calls[0]
        assert monorepo_root == tmp_path
    finally:
        # Restore original hooks
        _test_hooks.guard_find_monorepo_root = orig_find
        _test_hooks.guard_load_orchestrator = orig_load
