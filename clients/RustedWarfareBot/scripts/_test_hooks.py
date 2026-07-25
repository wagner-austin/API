"""Dependency-injection hooks for the repo-local scripts.

Same discipline as ``rw_bot.harness._test_hooks``: every symbol is bound to its
real implementation at import time and called unconditionally. Tests rebind and
restore. Keeping the guard's filesystem walk and dynamic import behind hooks is
what lets the guard be tested without scanning the real monorepo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol


class RunForProjectProto(Protocol):
    """The orchestrator entry point exported by ``monorepo_guards``."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run every guard rule against one project.

        Args:
            monorepo_root: Repository root containing ``libs/``.
            project_root: The project being checked.

        Returns:
            ``0`` when no rule fired, non-zero otherwise.
        """
        ...


class FindMonorepoRootProto(Protocol):
    """Locate the monorepo root from a starting directory."""

    def __call__(self, start: Path) -> Path:
        """Walk upward until a directory containing ``libs/`` is found.

        Args:
            start: Directory to begin from.

        Returns:
            The monorepo root.

        Raises:
            RuntimeError: When no ancestor contains ``libs/``.
        """
        ...


class LoadOrchestratorProto(Protocol):
    """Import the shared guard orchestrator from ``libs/``."""

    def __call__(self, monorepo_root: Path) -> RunForProjectProto:
        """Load the orchestrator.

        Args:
            monorepo_root: Repository root containing ``libs/``.

        Returns:
            The orchestrator's ``run_for_project`` callable.
        """
        ...


def _find_monorepo_root_impl(start: Path) -> Path:
    """Production implementation of :class:`FindMonorepoRootProto`.

    Args:
        start: Directory to begin from.

    Returns:
        The first ancestor (inclusive) containing a ``libs`` directory.

    Raises:
        RuntimeError: When the filesystem root is reached without a match.
    """
    current = start
    while True:
        if (current / "libs").is_dir():
            return current
        if current.parent == current:
            raise RuntimeError("monorepo root with 'libs' directory not found")
        current = current.parent


def _load_orchestrator_impl(monorepo_root: Path) -> RunForProjectProto:
    """Production implementation of :class:`LoadOrchestratorProto`.

    The orchestrator lives in ``libs/monorepo_guards`` and is imported
    dynamically because it is a path dependency rather than a published
    package. The attribute is bound straight to the Protocol type at
    assignment so the dynamic import never leaks an untyped value.

    Args:
        monorepo_root: Repository root containing ``libs/``.

    Returns:
        The orchestrator's ``run_for_project`` callable.
    """
    libs_path = monorepo_root / "libs"
    sys.path.insert(0, str(libs_path / "monorepo_guards" / "src"))
    sys.path.insert(0, str(libs_path))
    module = __import__("monorepo_guards.orchestrator", fromlist=["run_for_project"])
    run_for_project: RunForProjectProto = module.run_for_project
    return run_for_project


find_monorepo_root: FindMonorepoRootProto = _find_monorepo_root_impl
load_orchestrator: LoadOrchestratorProto = _load_orchestrator_impl


__all__ = [
    "FindMonorepoRootProto",
    "LoadOrchestratorProto",
    "RunForProjectProto",
    "find_monorepo_root",
    "load_orchestrator",
]
