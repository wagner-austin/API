"""Run this package's guard checks.

A bootstrap, not an implementation. Every guard rule -- and the argument
handling around it -- lives in ``libs/monorepo_guards``. This file exists
only because that package is a dependency of four of the forty-one packages
here and so cannot simply be imported by the other thirty-seven. It puts
``monorepo_guards`` on the path and hands over.

Invoked as ``python -m scripts.guard`` from the package directory, which is
the single form every Makefile uses. Running this file BY PATH instead puts
``scripts/`` on ``sys.path[0]`` rather than the package root, which is a
different program: it can only find an INSTALLED top-level ``scripts``.

Byte-identical in all forty-one packages, enforced by the
``guard-shim-not-canonical`` rule. Generated from
``monorepo_guards.guard_shim_template``; edit that, not this.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class _RunGuard(Protocol):
    def __call__(self, argv: Sequence[str] | None, *, project_root: Path) -> int: ...


def main(argv: Sequence[str] | None = None) -> int:
    """Run every guard rule against this package.

    Args:
        argv: Arguments excluding the program name, or None to read the
            process arguments.

    Returns:
        0 when no violations were found, 2 when there were.
    """
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[3] / "libs" / "monorepo_guards" / "src"))
    module = __import__("monorepo_guards.shim", fromlist=["run_guard"])
    run_guard: _RunGuard = module.run_guard
    return run_guard(argv, project_root=here.parents[1])


if __name__ == "__main__":
    raise SystemExit(main())
