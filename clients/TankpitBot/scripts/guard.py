"""Guard script for monorepo compliance checks.

Loads and runs the monorepo-guards orchestrator to enforce code quality rules.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from tankpit_bot import _hooks_guard


class _RunForProject(Protocol):
    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int: ...


def _find_monorepo_root_impl(start: Path) -> Path:
    """Production implementation - find monorepo root by looking for libs dir.

    Args:
        start: Starting path to search from.

    Returns:
        Path to monorepo root.

    Raises:
        RuntimeError: If monorepo root with 'libs' directory not found.
    """
    current = start
    while True:
        if (current / "libs").is_dir():
            return current
        if current.parent == current:
            raise RuntimeError("monorepo root with 'libs' directory not found")
        current = current.parent


def _find_monorepo_root(start: Path) -> Path:
    """Find monorepo root, using hook if set.

    Args:
        start: Starting path to search from.

    Returns:
        Path to monorepo root.
    """
    if _hooks_guard.guard_find_monorepo_root is not None:
        return _hooks_guard.guard_find_monorepo_root(start)
    return _find_monorepo_root_impl(start)


def _load_orchestrator_impl(monorepo_root: Path) -> _RunForProject:
    """Production implementation - load orchestrator from libs.

    Args:
        monorepo_root: Path to monorepo root.

    Returns:
        The run_for_project function from orchestrator.
    """
    libs_path = monorepo_root / "libs"
    guards_src = libs_path / "monorepo_guards" / "src"
    sys.path.insert(0, str(guards_src))
    sys.path.insert(0, str(libs_path))
    mod = __import__("monorepo_guards.orchestrator", fromlist=["run_for_project"])
    run_for_project: _RunForProject = mod.run_for_project
    return run_for_project


def _load_orchestrator(monorepo_root: Path) -> _RunForProject:
    """Load orchestrator, using hook if set.

    Args:
        monorepo_root: Path to monorepo root.

    Returns:
        The run_for_project function from orchestrator.
    """
    if _hooks_guard.guard_load_orchestrator is not None:
        return _hooks_guard.guard_load_orchestrator(monorepo_root)
    return _load_orchestrator_impl(monorepo_root)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guard checks.

    Args:
        argv: Command line arguments. Uses sys.argv[1:] if None.

    Returns:
        Exit code (0 for success, non-zero for violations).
    """
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    monorepo_root = _find_monorepo_root(project_root)
    run_for_project = _load_orchestrator(monorepo_root)

    args = list(argv) if argv is not None else list(sys.argv[1:])
    root_override: Path | None = None
    verbose = False
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == "--root" and idx + 1 < len(args):
            root_override = Path(args[idx + 1]).resolve()
            idx += 2
        elif token in ("-v", "--verbose"):
            verbose = True
            idx += 1
        else:
            idx += 1

    target_root = root_override if root_override is not None else project_root
    rc = run_for_project(monorepo_root=monorepo_root, project_root=target_root)
    if verbose:
        sys.stdout.write(f"guard_exit_code code={rc}\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(None))
