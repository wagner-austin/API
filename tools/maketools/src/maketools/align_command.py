"""``align --repo PATH --commit SHA [--drop SHA,SHA]``: the recipe's door.

It leads into :mod:`maketools.align_checkout`. Held apart from
:mod:`maketools.cli`, as ``concurrent`` is, so the registry file owns only
routing and this file owns the one command's argument rules and its
printed report.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypedDict

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools import _test_hooks
from maketools.align_checkout import align_checkout, render_report

#: The checkout to move, relative to the recipe's directory. The recipe
#: always passes it: ``make align-commit`` defaults ``REPO`` to ``.``.
REPO_FLAG: Final[str] = "--repo"

#: The commit to move to.
COMMIT_FLAG: Final[str] = "--commit"

#: The local commits the caller has read and allows dropping, comma-separated.
DROP_FLAG: Final[str] = "--drop"

#: How the command is written, for a usage refusal.
USAGE: Final[str] = f"{REPO_FLAG} PATH {COMMIT_FLAG} SHA [{DROP_FLAG} SHA,SHA]"


class AlignArguments(TypedDict):
    """What one ``align`` was asked to do.

    Attributes:
        repo: The checkout, as the recipe named it.
        commit: The commit to move it to.
        drop: The shas to drop (abbreviations allowed, empty strings kept
            for the aligner to ignore).
    """

    repo: str
    commit: str
    drop: list[str]


def parse_align_arguments(arguments: Sequence[str]) -> AlignArguments:
    """Read the three flags.

    ``--drop`` followed by nothing, or by the next flag, is an empty DROP:
    that is what ``make align-commit`` passes when the variable is unset,
    since make expands an unset ``$(DROP)`` to no word at all.

    Args:
        arguments: The flags and their values, in any order.

    Returns:
        The checkout, the commit and the shas to drop.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` without ``--repo`` or ``--commit`` and
            its value, or with any other argument.
    """
    rest = list(arguments)
    repo = ""
    commit = ""
    drop: list[str] = []
    while rest:
        flag = rest.pop(0)
        if flag == REPO_FLAG and rest:
            repo = rest.pop(0)
        elif flag == COMMIT_FLAG and rest:
            commit = rest.pop(0)
        elif flag == DROP_FLAG:
            drop = rest.pop(0).split(",") if rest and not rest[0].startswith("--") else []
        else:
            raise AppError(MaketoolsErrorCode.USAGE, f"align takes {USAGE}, got {list(arguments)}")
    if repo == "" or commit == "":
        raise AppError(
            MaketoolsErrorCode.USAGE, f"align needs {REPO_FLAG} PATH and {COMMIT_FLAG} SHA"
        )
    return AlignArguments(repo=repo, commit=commit, drop=drop)


def command_align(arguments: Sequence[str]) -> int:
    """Move the named main checkout, and print what moved.

    Args:
        arguments: The flags and their values.

    Returns:
        0 once the checkout reads back as the commit; a refusal propagates
        as the ``MAKETOOLS_ALIGN`` error :mod:`maketools.cli` prints.
    """
    asked = parse_align_arguments(arguments)
    report = align_checkout(Path(asked["repo"]), asked["commit"], asked["drop"])
    for line in render_report(report):
        _test_hooks.write_line(line)
    return 0


__all__ = [
    "COMMIT_FLAG",
    "DROP_FLAG",
    "REPO_FLAG",
    "USAGE",
    "AlignArguments",
    "command_align",
    "parse_align_arguments",
]
