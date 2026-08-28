"""Whether a package's commands actually run when run as modules.

THE DEFECT THIS FINDS. A module defining ``entrypoint`` with no
``if __name__ == "__main__"`` block is importable, runnable, and inert:
Python executes its top-level statements -- all of them ``def`` -- and stops.
It exits 0 and prints nothing. The console script a packager generates calls
``entrypoint`` directly and works, so the two invocation forms disagree in
silence and the broken one is indistinguishable from a command that
legitimately had nothing to say.

Exit 0 is success to every CI step and ``&&`` chain, which is what makes this
worse than a crash.

MEASURED COST ON THIS MONOREPO. Three hits in ``model_trainer``, the last
costing real time on 2026-08-27 while re-registering cloze floors. Then all
twelve commands in ``hpc3`` on 2026-08-28 -- including ``hpc3-preflight``,
whose entire job is answering "would this job start?". A preflight that never
runs and exits 0 is not a missing check; it is a green light that checked
nothing, and four jobs were one step from being submitted on the strength of
it.

WHY THIS IS ONE MODULE AND NOT ONE PER PACKAGE. The rule lived in
``model_trainer``'s test suite, and its own docstring recorded that the same
shape "had already been hit twice in the sibling hpc3 package" -- where it
then sat unfixed for weeks. Two copies of a rule is how one gets fixed and the
other does not. Consumers assert against this; they do not re-derive it.

Separate from :mod:`platform_core.testing` by role rather than by preference:
that module holds fakes and sample values a test injects, this one reads
source and reports on it, and merging them put ``testing`` over the
600-line ceiling.
"""

from __future__ import annotations

import ast
import pathlib

#: The block that makes ``python -m <module>`` do something.
MAIN_GUARD = '__name__ == "__main__"'

#: What the guard must call. ``entrypoint`` raises SystemExit carrying the
#: command's status; a guard calling ``main()`` instead would return that
#: status into nothing and the process would exit 0 whatever the command
#: reported -- the same silent success the guard exists to prevent, one layer
#: in.
GUARD_CALL = "    entrypoint()"


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


def unguarded_commands(cli_dir: pathlib.Path) -> tuple[str, ...]:
    """Commands that would do NOTHING when run as ``python -m <module>``.

    Args:
        cli_dir: The directory holding a package's commands.

    Returns:
        The file names lacking a guard, sorted. Empty when every command runs.
    """
    return tuple(
        p.name for p in command_modules(cli_dir) if MAIN_GUARD not in p.read_text(encoding="utf-8")
    )


def misguarded_commands(cli_dir: pathlib.Path) -> tuple[str, ...]:
    """Commands whose guard calls something other than the entry point.

    Args:
        cli_dir: The directory holding a package's commands.

    Returns:
        The file names whose guard does not call ``entrypoint()``, sorted.
        Empty when every guard is right. See :data:`GUARD_CALL` for why
        calling ``main()`` there is the same bug one layer in.
    """
    return tuple(
        p.name
        for p in command_modules(cli_dir)
        if MAIN_GUARD in (source := p.read_text(encoding="utf-8")) and GUARD_CALL not in source
    )


__all__ = [
    "GUARD_CALL",
    "MAIN_GUARD",
    "command_modules",
    "defines_entrypoint",
    "misguarded_commands",
    "public_modules",
    "unguarded_commands",
]
