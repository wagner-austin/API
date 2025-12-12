from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from handwriting_ai import _test_hooks


def main(argv: Sequence[str] | None = None) -> int:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    monorepo_root = _test_hooks.guard_find_monorepo_root(project_root)
    run_for_project = _test_hooks.guard_load_orchestrator(monorepo_root)

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
