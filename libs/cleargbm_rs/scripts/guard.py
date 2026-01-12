"""Guard script for cleargbm_rs.

Invokes monorepo_guards orchestrator for this project.
Uses _test_hooks for dependency injection in tests.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from scripts import _test_hooks

# Set script path at module load time for production
_test_hooks.set_script_path(Path(__file__).resolve())


class _RunForProject(Protocol):
    """Protocol for the orchestrator's run_for_project function."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guards for a project.

        Args:
            monorepo_root: Root of the monorepo.
            project_root: Root of the project to check.

        Returns:
            Exit code (0 for success).
        """
        ...


def _find_monorepo_root(start: Path) -> Path:
    """Find the monorepo root by looking for 'libs' directory.

    Args:
        start: Starting path to search from.

    Returns:
        Path to monorepo root.

    Raises:
        RuntimeError: If monorepo root not found.
    """
    current = start
    while True:
        if _test_hooks.is_dir(current / "libs"):
            return current
        if current.parent == current:
            raise RuntimeError("monorepo root with 'libs' directory not found")
        current = current.parent


def _load_orchestrator(monorepo_root: Path) -> _RunForProject:
    """Load the orchestrator module and return run_for_project.

    Args:
        monorepo_root: Root of the monorepo.

    Returns:
        The run_for_project function.
    """
    libs_path = monorepo_root / "libs"
    guards_src = libs_path / "monorepo_guards" / "src"
    sys.path.insert(0, str(guards_src))
    sys.path.insert(0, str(libs_path))
    mod = __import__("monorepo_guards.orchestrator", fromlist=["run_for_project"])
    run_for_project: _RunForProject = mod.run_for_project
    return run_for_project


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for guard script.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success).
    """
    script_path = _test_hooks.get_script_path()
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
