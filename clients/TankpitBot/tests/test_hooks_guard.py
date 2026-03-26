"""Tests for tankpit_bot._hooks_guard module."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from tankpit_bot import _hooks_guard


@pytest.fixture(autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    _hooks_guard.guard_find_monorepo_root = None
    _hooks_guard.guard_load_orchestrator = None


def test_guard_find_monorepo_root_initially_none() -> None:
    """guard_find_monorepo_root starts as None."""
    assert _hooks_guard.guard_find_monorepo_root is None


def test_guard_load_orchestrator_initially_none() -> None:
    """guard_load_orchestrator starts as None."""
    assert _hooks_guard.guard_load_orchestrator is None


def test_set_and_call_find_monorepo_root(tmp_path: Path) -> None:
    """Can set and call guard_find_monorepo_root hook."""

    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    fake = _FakeFindRoot()
    _hooks_guard.guard_find_monorepo_root = fake
    result = fake(tmp_path / "nested")
    assert result == tmp_path


def test_set_and_call_load_orchestrator(tmp_path: Path) -> None:
    """Can set and call guard_load_orchestrator hook."""

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> _hooks_guard.RunForProjectProto:
            def _run(*, monorepo_root: Path, project_root: Path) -> int:
                return 42

            return _run

    fake = _FakeLoader()
    _hooks_guard.guard_load_orchestrator = fake
    run_fn = fake(tmp_path)
    result = run_fn(monorepo_root=tmp_path, project_root=tmp_path)
    assert result == 42
