"""``align --commit SHA [--drop SHA,SHA]``: the recipe's door into :mod:`maketools.align_checkout`.

Held apart from :mod:`maketools.cli`, as ``concurrent`` is, so the
registry file owns only routing and this file owns the one command's
argument rules and its printed report.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools import _test_hooks
from maketools.align_checkout import align_checkout, render_report

#: The commit to move to.
COMMIT_FLAG: Final[str] = "--commit"

#: The local commits the caller has read and allows dropping, comma-separated.
DROP_FLAG: Final[str] = "--drop"


def parse_align_arguments(arguments: Sequence[str]) -> tuple[str, list[str]]:
    """Read the two flags.

    ``--drop`` followed by nothing, or by the next flag, is an empty DROP:
    that is what ``make align-commit`` passes when the variable is unset,
    since make expands an unset ``$(DROP)`` to no word at all.

    Args:
        arguments: The flags and their values, in any order.

    Returns:
        The commit and the shas to drop (abbreviations allowed, empty
        strings kept for the aligner to ignore).

    Raises:
        AppError: ``MAKETOOLS_USAGE`` without ``--commit`` and its value, or
            with any other argument.
    """
    rest = list(arguments)
    commit = ""
    drop: list[str] = []
    while rest:
        flag = rest.pop(0)
        if flag == COMMIT_FLAG and rest:
            commit = rest.pop(0)
        elif flag == DROP_FLAG:
            drop = rest.pop(0).split(",") if rest and not rest[0].startswith("--") else []
        else:
            raise AppError(
                MaketoolsErrorCode.USAGE,
                f"align takes {COMMIT_FLAG} SHA [{DROP_FLAG} SHA,SHA], got {list(arguments)}",
            )
    if commit == "":
        raise AppError(MaketoolsErrorCode.USAGE, f"align needs {COMMIT_FLAG} SHA")
    return commit, drop


def command_align(arguments: Sequence[str]) -> int:
    """Move the main checkout the recipe runs in, and print what moved.

    Args:
        arguments: The flags and their values.

    Returns:
        0 once the checkout reads back as the commit; a refusal propagates
        as the ``MAKETOOLS_ALIGN`` error :mod:`maketools.cli` prints.
    """
    commit, drop = parse_align_arguments(arguments)
    report = align_checkout(Path.cwd(), commit, drop)
    for line in render_report(report):
        _test_hooks.write_line(line)
    return 0


__all__ = ["COMMIT_FLAG", "DROP_FLAG", "command_align", "parse_align_arguments"]
