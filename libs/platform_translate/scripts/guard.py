"""Guard script for platform_translate.

Invokes the monorepo_guards orchestrator for this project. Every non-pure
operation is reached through ``scripts._test_hooks``, which binds each symbol
to its real implementation at import time. The hooks are called directly, so
there is no conditional dispatch and no separate production code path.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from scripts import _test_hooks

# Record the script path at module load time for production use.
_test_hooks.set_script_path(Path(__file__).resolve())


def _find_monorepo_root(start: Path) -> Path:
    """Find the monorepo root by looking for a 'libs' directory.

    Args:
        start: Starting path to search upward from.

    Returns:
        Path to the monorepo root.

    Raises:
        RuntimeError: When no ancestor directory contains 'libs'.
    """
    current = start
    while True:
        if _test_hooks.is_dir(current / "libs"):
            return current
        if current.parent == current:
            raise RuntimeError("monorepo root with 'libs' directory not found")
        current = current.parent


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guard checks for this project.

    Args:
        argv: Command line arguments, excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code, 0 when no violations were found.
    """
    script_path = _test_hooks.get_script_path()
    project_root = script_path.parents[1]
    monorepo_root = _find_monorepo_root(project_root)
    run_for_project = _test_hooks.load_orchestrator(monorepo_root)

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
