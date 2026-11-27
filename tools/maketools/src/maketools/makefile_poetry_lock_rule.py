"""The rule that no recipe runs ``poetry lock`` (MCPs board task 972ea464).

WHY. A check that re-resolves its package's lock pays for a full resolution
on every run and tests a lock other than the one committed:
``libs/covenant_nn``'s two resolutions took 292 s and 99 s of a 454 s check
on the GitHub runner (API CI run 46048435508), and its committed lock was
stale against ``libs/cleargbm_rs`` the whole time. A recipe verifies the
lock with ``run.py lock-check`` instead, and ``run.py lock`` is the one
writer (:mod:`maketools.poetry_lock`).

WHAT THE RULE HOLDS. A recipe line that runs a command
(:func:`maketools.makefile_grammar.command_lines`: neither a comment nor an
``echo``, which is help text naming the command) does not run
``poetry lock``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from maketools.makefile_grammar import Violation, command_lines, tracked_makefiles

#: ``poetry lock``, its subcommand a word of its own, after any of poetry's
#: global options: ``-C`` / ``--directory`` and ``-P`` / ``--project`` take
#: the next word as their value (``poetry -C lock sync`` syncs a directory
#: named ``lock``), and every other option is one word.
POETRY_LOCK: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[\s;&|(])poetry"
    r"(?:\s+(?:-C|-P|--directory|--project)\s+\S+|\s+-(?!C\s|P\s|-directory\s|-project\s)\S+)*"
    r"\s+lock\b"
)

#: The rule's name, as :func:`maketools.makefile_grammar.render_violation` prints it.
RULE_RECIPE_LOCK: Final[str] = (
    "a recipe verifies poetry.lock with run.py lock-check and never runs poetry lock"
)


def check_poetry_lock_rule(path: Path, text: str) -> list[Violation]:
    """Apply the rule to one Makefile.

    Args:
        path: The Makefile.
        text: Its contents.

    Returns:
        One violation per recipe line that runs ``poetry lock``, naming the line.
    """
    return [
        Violation(path=path, line_number=number, rule=RULE_RECIPE_LOCK, text=body)
        for number, body in command_lines(text)
        if POETRY_LOCK.search(body) is not None
    ]


def lint_poetry_locks(repo_root: Path) -> list[Violation]:
    """Apply the rule to every tracked Makefile.

    Args:
        repo_root: The repository root.

    Returns:
        Every violation across the tree, in git's order.
    """
    found: list[Violation] = []
    for makefile in tracked_makefiles(repo_root):
        found.extend(check_poetry_lock_rule(makefile, makefile.read_text(encoding="utf-8")))
    return found


__all__ = ["POETRY_LOCK", "RULE_RECIPE_LOCK", "check_poetry_lock_rule", "lint_poetry_locks"]
