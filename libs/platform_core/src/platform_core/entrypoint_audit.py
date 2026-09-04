"""Which modules in a command directory are commands.

THE DEFECT THIS DIRECTORY OF COMMANDS KEEPS PRODUCING. A module defining
``entrypoint`` with no ``if __name__ == "__main__"`` block is importable,
runnable, and inert: Python executes its top-level statements -- all of them
``def`` -- and stops. It exits 0 and prints nothing. The console script a
packager generates calls ``entrypoint`` directly and works, so the two
invocation forms disagree in silence and the broken one is indistinguishable
from a command that legitimately had nothing to say.

Exit 0 is success to every CI step and ``&&`` chain, which is what makes this
worse than a crash.

MEASURED COST ON THIS MONOREPO. Three hits in ``model_trainer``, the last
costing real time on 2026-08-27 while re-registering cloze floors. Then all
twelve commands in ``hpc3`` on 2026-08-28 -- including ``hpc3-preflight``,
whose entire job is answering "would this job start?". Then
``code_style_eval`` on 2026-09-04, where it discarded a scoring pass over 226
files an A30 had spent thirty-three minutes generating.

THIS MODULE NO LONGER DECIDES WHETHER A COMMAND IS GUARDED, and the two
functions that used to are deleted rather than deprecated. They searched for
the substrings ``__name__ == "__main__"`` and ``"    entrypoint()"``, which
was wrong in both directions and measurably so: a module whose DOCSTRING
mentioned the guard was reported as misguarded rather than unguarded, and --
the one that matters -- a module whose guard called ``main()`` while some
unrelated helper happened to call ``entrypoint()`` at four-space indent was
reported clean. That is a false negative on exactly the silent-exit-0 defect
above.

``monorepo_guards.entrypoint_rules.EntrypointRule`` decides it now, on the
AST, for every package in the repository including ones nobody has written
yet, and it runs inside every ``make lint``. A per-package test asserting the
same thing a second time is the fork this module's own docstring used to warn
against.

WHAT IS LEFT HERE IS THE HALF A GUARD CANNOT DO. A guard is handed files and
reports on them; it cannot hand a package's test suite the list of commands
to go and RUN. Enumeration stays importable so that
``test_cli_entrypoint_shape`` can parametrize a ``runpy`` pass over each
command -- which is the only check that proves the chain executes rather than
describing its shape.
"""

from __future__ import annotations

import ast
import pathlib


def defines_entrypoint(source: str) -> bool:
    """Whether a module's source defines a console-script entry point.

    Parsed rather than searched: a mention of ``entrypoint`` in a docstring
    or an ``__all__`` list is not a definition, and this predicate is about
    what a module DEFINES. Keying on the entry point is also what lets a
    helper module with no command in it fall out of scope by its own shape,
    instead of by being written into an exemption list somebody maintains.

    Args:
        source: The module source.

    Returns:
        True when it defines a top-level ``entrypoint`` function.
    """
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "entrypoint"
        for node in ast.parse(source).body
    )


def public_modules(cli_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Every public module in a command directory.

    Args:
        cli_dir: The directory holding a package's commands.

    Returns:
        The module paths, sorted, excluding dunder and private modules.
    """
    return tuple(sorted(p for p in cli_dir.glob("*.py") if not p.name.startswith("_")))


def command_modules(cli_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """The modules in a command directory that are actually commands.

    Args:
        cli_dir: The directory holding a package's commands.

    Returns:
        The paths defining an ``entrypoint``, sorted.
    """
    return tuple(
        p for p in public_modules(cli_dir) if defines_entrypoint(p.read_text(encoding="utf-8"))
    )


__all__ = [
    "command_modules",
    "defines_entrypoint",
    "public_modules",
]
