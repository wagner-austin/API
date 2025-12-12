from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pytest
import scripts.guard as guard

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import GuardRunForProjectProtocol


class _RunCall(TypedDict):
    monorepo_root: Path
    project_root: Path


def test_default_find_monorepo_root_locates_libs(tmp_path: Path) -> None:
    """Test that the default implementation finds monorepo root with libs dir."""
    root = tmp_path / "repo"
    libs = root / "libs"
    libs.mkdir(parents=True, exist_ok=True)
    start = libs / "some" / "deep"
    start.mkdir(parents=True, exist_ok=True)

    found = _test_hooks._default_guard_find_monorepo_root(start)
    assert found == root


def test_default_find_monorepo_root_raises_when_missing(tmp_path: Path) -> None:
    """Test that the default implementation raises when no libs dir found."""
    start = tmp_path / "no_libs"
    start.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError):
        _ = _test_hooks._default_guard_find_monorepo_root(start)


def test_main_invokes_run_and_supports_flags(tmp_path: Path) -> None:
    """Test that main() invokes the run_for_project hook with correct args."""
    calls: list[_RunCall] = []

    def _fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append({"monorepo_root": monorepo_root, "project_root": project_root})
        return 3

    def _fake_find_monorepo_root(start: Path) -> Path:
        return start

    def _fake_load_orchestrator(monorepo_root: Path) -> GuardRunForProjectProtocol:
        _ = monorepo_root  # Unused in fake
        return _fake_run_for_project

    _test_hooks.guard_find_monorepo_root = _fake_find_monorepo_root
    _test_hooks.guard_load_orchestrator = _fake_load_orchestrator

    project_root = Path(__file__).resolve().parents[1]

    rc = guard.main(["--root", str(tmp_path), "-v"])
    assert rc == 3
    assert len(calls) == 1
    assert calls[0]["monorepo_root"] == project_root
    assert calls[0]["project_root"] == tmp_path


def test_main_uses_default_args_when_none(tmp_path: Path) -> None:
    """Test that main() uses defaults when argv is None."""
    calls: list[_RunCall] = []

    def _fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append({"monorepo_root": monorepo_root, "project_root": project_root})
        return 0

    def _fake_find(start: Path) -> Path:
        return start

    def _fake_load(monorepo_root: Path) -> GuardRunForProjectProtocol:
        _ = monorepo_root  # Unused in fake
        return _fake_run_for_project

    _test_hooks.guard_find_monorepo_root = _fake_find
    _test_hooks.guard_load_orchestrator = _fake_load

    rc = guard.main(None)
    assert rc == 0
    assert calls[0]["project_root"].name == "handwriting-ai"


def test_main_skips_unknown_flags(tmp_path: Path) -> None:
    """Test that main() ignores unknown flags."""
    calls: list[_RunCall] = []

    def _fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append({"monorepo_root": monorepo_root, "project_root": project_root})
        return 7

    def _fake_find(start: Path) -> Path:
        return start

    def _fake_load(monorepo_root: Path) -> GuardRunForProjectProtocol:
        _ = monorepo_root  # Unused in fake
        return _fake_run_for_project

    _test_hooks.guard_find_monorepo_root = _fake_find
    _test_hooks.guard_load_orchestrator = _fake_load

    rc = guard.main(["--unknown", "--root", str(tmp_path)])
    assert rc == 7
    assert len(calls) == 1
    assert calls[0]["project_root"] == tmp_path
