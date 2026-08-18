"""Run the shared monorepo guard rules against this project.

Invoked by ``make lint`` as ``python -m scripts.guard``. The rule set itself
lives in ``libs/monorepo_guards`` and is shared with every other client; this
module only applies it to ``src``, ``tests``, and ``scripts``. Project-specific
rules, when they exist, are added to the shared library rather than forked
here.

``monorepo-guards`` is a develop path dependency declared in
``pyproject.toml``, so poetry installs a ``.pth`` into the venv and
``monorepo_guards`` imports directly. The other clients reach it through
``sys.path`` insertion plus ``__import__``, which predates that dependency and
is no longer load-bearing; a direct import needs no Protocol indirection and
cannot drift from the real signature.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from monorepo_guards.orchestrator import run_for_project


def parse_target_root(argv: Sequence[str], default: Path) -> Path:
    """Resolve the directory the guard rules should be applied to.

    Args:
        argv: Argument list excluding the program name. ``--root <path>``
            selects a directory other than the project root.
        default: Directory to use when ``--root`` is absent.

    Returns:
        The resolved directory to check.
    """
    index = 0
    target = default
    while index < len(argv):
        if argv[index] == "--root" and index + 1 < len(argv):
            target = Path(argv[index + 1]).resolve()
            index += 2
        else:
            index += 1
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guard rules over this project.

    Args:
        argv: Argument list excluding the program name. ``None`` reads
            ``sys.argv[1:]``.

    Returns:
        ``0`` when no rule fired, non-zero otherwise.
    """
    project_root = Path(__file__).resolve().parents[1]
    monorepo_root = project_root.parents[1]
    args = list(argv) if argv is not None else list(sys.argv[1:])
    target_root = parse_target_root(args, project_root)
    return run_for_project(monorepo_root=monorepo_root, project_root=target_root)


if __name__ == "__main__":
    raise SystemExit(main(None))
