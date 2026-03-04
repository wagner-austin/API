"""Internal test hooks for covenant_ml - allows injecting test dependencies.

Production code sets hooks to real implementations at startup.
Tests set hooks to fakes before running.

Usage in tests:
    from covenant_ml import _test_hooks
    _test_hooks.guard_find_monorepo_root = FakeFindRoot()
    _test_hooks.guard_load_orchestrator = FakeLoader()
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class FindMonorepoRootProto(Protocol):
    """Protocol for _find_monorepo_root hook."""

    def __call__(self, start: Path) -> Path:
        """Find monorepo root starting from given path.

        Args:
            start: Starting path to search from.

        Returns:
            Path to monorepo root.
        """
        ...


class RunForProjectProto(Protocol):
    """Protocol for run_for_project hook."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guard checks for a project.

        Args:
            monorepo_root: Path to monorepo root.
            project_root: Path to project root.

        Returns:
            Exit code from guard checks.
        """
        ...


class LoadOrchestratorProto(Protocol):
    """Protocol for _load_orchestrator hook."""

    def __call__(self, monorepo_root: Path) -> RunForProjectProto:
        """Load the guard orchestrator.

        Args:
            monorepo_root: Path to monorepo root.

        Returns:
            run_for_project function.
        """
        ...


# Guard hooks - None means use default behavior (production implementation)
guard_find_monorepo_root: FindMonorepoRootProto | None = None
guard_load_orchestrator: LoadOrchestratorProto | None = None


__all__ = [
    "FindMonorepoRootProto",
    "LoadOrchestratorProto",
    "RunForProjectProto",
    "guard_find_monorepo_root",
    "guard_load_orchestrator",
]
