"""Every CLI that can be run as a module actually runs when you do.

THE DEFECT THIS ENCODES, HIT THREE TIMES BEFORE IT WAS FIXED. A module under
``cli/`` that defines ``entrypoint`` but has no ``if __name__ == "__main__"``
block is importable, runnable, and does NOTHING:

    python -m model_trainer.cli.score_baseline --model gpt2 ... --out x.json
    (exits 0, writes no file)

That is worse than a crash. The cluster invokes the same command through the
``modeltrainer-score-baseline`` console script, which works -- so the two
invocation forms disagree, and the broken one looks exactly like a scoring run
that legitimately produced nothing. It cost real time on 2026-08-27 while
re-registering the cloze floors, and the same shape had already been hit twice
in the sibling ``hpc3`` package.

WHY THE PREDICATE IS "DEFINES ``entrypoint``" AND NOT A FILE LIST. Not every
module under ``cli/`` is a command -- ``record_reports`` is a library of report
helpers with no ``main`` and no ``entrypoint``, and it correctly needs no
guard. Keying on the entry point means that module falls out of scope by its
own shape rather than by being written into an exemption list somebody has to
maintain.
"""

from __future__ import annotations

import ast
import pathlib

CLI_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "model_trainer" / "cli"

#: The block that makes `python -m <module>` do something.
MAIN_GUARD = '__name__ == "__main__"'


def _cli_modules() -> tuple[pathlib.Path, ...]:
    """Every public module under ``cli/``.

    Returns:
        The module paths, sorted, excluding dunder and private modules.
    """
    return tuple(sorted(p for p in CLI_DIR.glob("*.py") if not p.name.startswith("_")))


def _defines_entrypoint(path: pathlib.Path) -> bool:
    """Whether a module defines a console-script entry point.

    Parsed rather than grepped: a mention of ``entrypoint`` inside a docstring
    or an ``__all__`` list is not a definition, and this rule is about what the
    module DEFINES.

    Args:
        path: The module to inspect.

    Returns:
        True when the module defines a top-level ``entrypoint`` function.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "entrypoint" for node in tree.body
    )


def test_the_cli_directory_is_not_empty() -> None:
    # A rule that silently scans nothing passes forever. This is the check
    # that the check has subjects.
    assert len(_cli_modules()) >= 10


def test_at_least_one_module_is_a_library_rather_than_a_command() -> None:
    # The predicate's other half. If every module had an entry point, keying
    # on it would be indistinguishable from keying on the directory, and the
    # rule would quietly become a file list the first time a helper appeared.
    without = [p.name for p in _cli_modules() if not _defines_entrypoint(p)]

    assert without != []


def test_every_command_runs_when_run_as_a_module() -> None:
    missing = [
        p.name
        for p in _cli_modules()
        if _defines_entrypoint(p) and MAIN_GUARD not in p.read_text(encoding="utf-8")
    ]

    assert missing == []


def test_the_guard_calls_the_entry_point_rather_than_main() -> None:
    # `entrypoint` raises SystemExit(main()); calling `main()` from the guard
    # would return an exit code into nothing and the process would exit 0
    # whatever the command reported.
    wrong = [
        p.name
        for p in _cli_modules()
        if _defines_entrypoint(p)
        and MAIN_GUARD in p.read_text(encoding="utf-8")
        and "    entrypoint()" not in p.read_text(encoding="utf-8")
    ]

    assert wrong == []
