"""Guard script for platform_devpost."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol


class _RunForProject(Protocol):
    """Protocol for run_for_project function."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guards for a project.

        Args:
            monorepo_root: Path to monorepo root.
            project_root: Path to project root.

        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        ...


def _default_is_dir(p: Path) -> bool:
    """Production implementation - uses Path.is_dir().

    Args:
        p: Path to check.

    Returns:
        True if path is a directory, False otherwise.
    """
    return p.is_dir()


_is_dir: Callable[[Path], bool] = _default_is_dir


def _find_monorepo_root(start: Path) -> Path:
    """Find the monorepo root by looking for libs directory.

    Args:
        start: Starting path to search from.

    Returns:
        Path to monorepo root.

    Raises:
        RuntimeError: If monorepo root not found.
    """
    current = start
    while True:
        if _is_dir(current / "libs"):
            return current
        if current.parent == current:
            raise RuntimeError("monorepo root with 'libs' directory not found")
        current = current.parent


def _load_orchestrator(monorepo_root: Path) -> _RunForProject:
    """Load the orchestrator from monorepo_guards.

    Args:
        monorepo_root: Path to monorepo root.

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
    """Run guards for this project.

    Args:
        argv: Command line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success, non-zero for failure).
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
