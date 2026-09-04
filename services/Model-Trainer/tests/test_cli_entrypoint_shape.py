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
re-registering the cloze floors.

WHERE THE SHAPE CHECK LIVES, AND WHY NOT HERE. It used to live in this file,
and its own docstring recorded that "the same shape had already been hit twice
in the sibling ``hpc3`` package" -- where it then sat unfixed until
2026-08-28, when all twelve hpc3 commands turned out to have it, including a
preflight that reported a clean green light without checking anything. It was
then lifted into a shared scan the packages CALLED, which moved the problem
rather than solving it: ``code_style_eval`` never called it, and on 2026-09-04
lost a scoring pass over 226 files to the same defect.

``monorepo_guards.entrypoint_rules.EntrypointRule`` decides it now, on the
AST, for every package in the repository, inside every ``make lint``. Nothing
has to remember to call it. This file no longer asserts it.

WHY THE PREDICATE IS "DEFINES ``entrypoint``" AND NOT A FILE LIST. Not every
module under ``cli/`` is a command -- ``record_reports`` is a library of report
helpers with no ``main`` and no ``entrypoint``, and it correctly needs no
guard. Keying on the entry point means that module falls out of scope by its
own shape rather than by being written into an exemption list somebody has to
maintain. This package is where that half of the predicate is exercised: hpc3
has no such library module, so only here can the scan be shown to select
something narrower than the directory.
"""

from __future__ import annotations

import pathlib

from platform_core.entrypoint_audit import command_modules, public_modules

CLI_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "model_trainer" / "cli"


def test_the_cli_directory_is_not_empty() -> None:
    # A rule that silently scans nothing passes forever. This is the check
    # that the check has subjects.
    assert len(public_modules(CLI_DIR)) >= 10


def test_at_least_one_module_is_a_library_rather_than_a_command() -> None:
    # The predicate's other half. If every module had an entry point, keying
    # on it would be indistinguishable from keying on the directory, and the
    # rule would quietly become a file list the first time a helper appeared.
    without = set(public_modules(CLI_DIR)) - set(command_modules(CLI_DIR))

    assert without != set()
