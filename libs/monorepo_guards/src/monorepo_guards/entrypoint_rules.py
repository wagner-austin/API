"""Rule refusing a command module that does nothing when run as a module.

A module under ``cli/`` that defines ``entrypoint`` and carries no
``if __name__ == "__main__"`` block is importable, runnable, and does
NOTHING:

    python -m code_style_eval.cli.evaluate --holdout H --generated-dir D ...
    (exits 0, writes no file)

That is worse than a crash. The console script for the same command works, so
the two invocation forms disagree, and the silent one is indistinguishable
from a run that legitimately produced no output.

THIS RULE EXISTS BECAUSE THE PER-PACKAGE TEST WAS MISSED THREE TIMES.
``hpc3`` had the defect in all twelve of its commands, including a preflight
that reported a clean green light without checking anything, and added a test.
``model_trainer`` added the same test and its docstring recorded that hpc3 had
already been hit twice. ``code_style_eval`` had neither, and on 2026-09-04 a
scoring run over 226 generated files -- thirty-three minutes of A30 time to
produce -- exited 0 having scored none of them.

A check each package must remember to add is a check that gets missed, and
the record shows exactly that. Here it applies to every package with a
``cli/`` directory, including packages nobody has written yet.

WHY THE PREDICATE IS "DEFINES ``entrypoint``" AND NOT A FILE LIST. Not every
module under ``cli/`` is a command: a module of shared report helpers has no
entry point and correctly needs no guard. Keying on the entry point lets such
a module fall out of scope by its own shape rather than by being written into
an exemption list somebody has to maintain.
"""

from __future__ import annotations

import ast
from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.util import parse_source

CLI_DIRECTORY = "cli"
"""The directory name a command module must live in to be checked.

Scoped rather than repository-wide: plenty of modules define a function
called ``entrypoint`` without being console scripts, and a rule that fired on
all of them would be answered by exemptions rather than by guards.
"""

ENTRYPOINT_NAME = "entrypoint"

_MAIN_TEST = "__name__"
_MAIN_VALUE = "__main__"


def defines_entrypoint(tree: ast.Module) -> bool:
    """Say whether a module defines a console-script entry point.

    Args:
        tree: The parsed module.

    Returns:
        Whether a top-level function named ``entrypoint`` exists.
    """
    return any(
        isinstance(node, ast.FunctionDef) and node.name == ENTRYPOINT_NAME for node in tree.body
    )


def _is_main_guard(node: ast.If) -> bool:
    """Say whether a conditional is an ``if __name__ == "__main__"`` block.

    Takes ``ast.If`` rather than ``ast.stmt`` so the caller's own isinstance
    narrows the list it builds; checking it twice would leave mypy unable to
    see that ``guards`` holds conditionals.

    Args:
        node: A module-level conditional.

    Returns:
        Whether it guards on being run as a module.
    """
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    if not isinstance(left, ast.Name) or left.id != _MAIN_TEST:
        return False
    return isinstance(right, ast.Constant) and right.value == _MAIN_VALUE


def _calls_entrypoint(node: ast.If) -> bool:
    """Say whether a main guard calls ``entrypoint`` rather than ``main``.

    Calling ``main()`` from the guard returns an exit code into nothing, so
    the process exits 0 whatever the command reported -- the same silent
    success by a different route.

    Args:
        node: The main-guard block.

    Returns:
        Whether its body calls ``entrypoint``.
    """
    return any(
        isinstance(statement, ast.Call)
        and isinstance(statement.func, ast.Name)
        and statement.func.id == ENTRYPOINT_NAME
        for statement in ast.walk(node)
    )


class EntrypointRule:
    """Every ``cli/`` command runs when run as a module, and runs its entry point."""

    name = "entrypoint"

    def _violations(self, path: Path) -> list[Violation]:
        """Check one file.

        Args:
            path: The file to check.

        Returns:
            Its violations, empty when the file is not a command or is
            correctly guarded.
        """
        if path.parent.name != CLI_DIRECTORY or path.name.startswith("_"):
            return []
        tree = parse_source(path)
        if not defines_entrypoint(tree):
            return []
        guards = [node for node in tree.body if isinstance(node, ast.If) and _is_main_guard(node)]
        if not guards:
            return [
                Violation(
                    file=path,
                    line_no=1,
                    kind="entrypoint-unguarded",
                    line=(
                        'defines entrypoint but has no `if __name__ == "__main__"` '
                        "block, so `python -m` on it exits 0 having done nothing"
                    ),
                )
            ]
        if not any(_calls_entrypoint(guard) for guard in guards):
            return [
                Violation(
                    file=path,
                    line_no=guards[0].lineno,
                    kind="entrypoint-misguarded",
                    line=(
                        "the __main__ guard must call entrypoint(); calling main() "
                        "returns an exit code into nothing and the process exits 0"
                    ),
                )
            ]
        return []

    def run(self, files: list[Path]) -> list[Violation]:
        """Check every file.

        Args:
            files: The files to check.

        Returns:
            Every violation found, in file order.
        """
        out: list[Violation] = []
        for path in files:
            out.extend(self._violations(path))
        return out


__all__ = ["CLI_DIRECTORY", "ENTRYPOINT_NAME", "EntrypointRule", "defines_entrypoint"]
