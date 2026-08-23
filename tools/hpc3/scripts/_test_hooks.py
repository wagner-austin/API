"""Test hooks for the guard script.

Every symbol here is bound to its real implementation at import time, so
``scripts.guard`` calls the hook directly and production and test code paths
are byte-identical in shape. Tests rebind a symbol to a fake and restore it
afterwards; nothing branches on whether a hook is "set".

This module is private to the package; consumers outside it must not import
from here.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol


class RunForProjectProtocol(Protocol):
    """Protocol for the orchestrator's ``run_for_project`` function."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guards for a project.

        Args:
            monorepo_root: Root of the monorepo.
            project_root: Root of the project to check.

        Returns:
            Exit code, 0 when no violations were found.
        """
        ...


class IsDirProtocol(Protocol):
    """Protocol for checking whether a path is a directory."""

    def __call__(self, path: Path) -> bool:
        """Check whether a path is a directory.

        Args:
            path: Path to check.

        Returns:
            True when the path is a directory, False otherwise.
        """
        ...


class GetScriptPathProtocol(Protocol):
    """Protocol for resolving the guard script's own path."""

    def __call__(self) -> Path:
        """Resolve the guard script path.

        Returns:
            Absolute path to the guard script.

        Raises:
            RuntimeError: When the script path has not been set.
        """
        ...


class LoadOrchestratorProtocol(Protocol):
    """Protocol for loading the monorepo guard orchestrator."""

    def __call__(self, monorepo_root: Path) -> RunForProjectProtocol:
        """Load the orchestrator.

        Args:
            monorepo_root: Root of the monorepo.

        Returns:
            The orchestrator's ``run_for_project`` function.
        """
        ...


def _real_is_dir(path: Path) -> bool:
    """Real implementation using Path.is_dir().

    Args:
        path: Path to check.

    Returns:
        True when the path is a directory, False otherwise.
    """
    return path.is_dir()


_SCRIPT_PATH: Path | None = None


def set_script_path(path: Path) -> None:
    """Record the guard script's path for production use.

    Args:
        path: Absolute path to the guard script.
    """
    global _SCRIPT_PATH
    _SCRIPT_PATH = path


def _real_get_script_path() -> Path:
    """Real implementation returning the recorded guard script path.

    Returns:
        Absolute path to the guard script.

    Raises:
        RuntimeError: When the script path has not been set.
    """
    if _SCRIPT_PATH is None:
        raise RuntimeError("Script path not set - call set_script_path first")
    return _SCRIPT_PATH


def _real_load_orchestrator(monorepo_root: Path) -> RunForProjectProtocol:
    """Real implementation importing the orchestrator from the monorepo.

    Args:
        monorepo_root: Root of the monorepo.

    Returns:
        The orchestrator's ``run_for_project`` function.
    """
    libs_path = monorepo_root / "libs"
    guards_src = libs_path / "monorepo_guards" / "src"
    sys.path.insert(0, str(guards_src))
    sys.path.insert(0, str(libs_path))
    mod = __import__("monorepo_guards.orchestrator", fromlist=["run_for_project"])
    run_for_project: RunForProjectProtocol = mod.run_for_project
    return run_for_project


is_dir: IsDirProtocol = _real_is_dir
get_script_path: GetScriptPathProtocol = _real_get_script_path
load_orchestrator: LoadOrchestratorProtocol = _real_load_orchestrator


__all__ = [
    "GetScriptPathProtocol",
    "IsDirProtocol",
    "LoadOrchestratorProtocol",
    "RunForProjectProtocol",
    "get_script_path",
    "is_dir",
    "load_orchestrator",
    "set_script_path",
]
