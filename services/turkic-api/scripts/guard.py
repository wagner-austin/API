from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from turkic_api import _test_hooks
from turkic_api._test_hooks import GuardRunForProjectProtocol


def _find_monorepo_root(start: Path) -> Path:
    """Find monorepo root using hook for test injection."""
    return _test_hooks.guard_find_monorepo_root(start)


def _load_orchestrator(monorepo_root: Path) -> GuardRunForProjectProtocol:
    """Load orchestrator using hook for test injection."""
    return _test_hooks.guard_load_orchestrator(monorepo_root)


def main(argv: Sequence[str] | None = None) -> int:
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
