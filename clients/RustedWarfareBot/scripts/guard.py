"""Run the shared monorepo guard rules against this project.

Invoked by ``make lint`` as ``python -m scripts.guard``. The rule set itself
lives in ``libs/monorepo_guards`` and is shared with every other client — this
module only locates it and applies it to ``src``, ``tests``, and ``scripts``.
Project-specific rules, when they exist, are added to the shared library rather
than forked here.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from scripts import _test_hooks


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guard rules.

    Args:
        argv: Argument list excluding the program name. ``None`` reads
            ``sys.argv[1:]``. Supports ``--root <path>`` to check a directory
            other than this project.

    Returns:
        ``0`` when no rule fired, non-zero otherwise.

    Raises:
        RuntimeError: When the monorepo root cannot be located.
    """
    project_root = Path(__file__).resolve().parents[1]
    monorepo_root = _test_hooks.find_monorepo_root(project_root)
    run_for_project = _test_hooks.load_orchestrator(monorepo_root)

    args = list(argv) if argv is not None else list(sys.argv[1:])
    target_root = project_root
    index = 0
    while index < len(args):
        if args[index] == "--root" and index + 1 < len(args):
            target_root = Path(args[index + 1]).resolve()
            index += 2
        else:
            index += 1

    return run_for_project(monorepo_root=monorepo_root, project_root=target_root)


if __name__ == "__main__":
    raise SystemExit(main(None))
