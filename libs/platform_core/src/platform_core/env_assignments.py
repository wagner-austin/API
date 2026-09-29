"""The untracked credentials file a scheduled task sources, read without PowerShell.

A machine's standing credentials for the board and the fleet queue live in
one untracked file, ``tools/hpc-wake/runs/env.ps1`` in the API checkout. It is
PowerShell so an interactive session can keep dot-sourcing it. A SCHEDULED
task cannot: ``powershell.exe`` run as a task action under ``LogonType=S4U``
on the hub stalls before its engine loads and never exits (measured
2026-09-09 by the Docker-autostart work, and again on 2026-09-28, when two
fleet ticks hung 16 and 7 minutes and blocked their nodes; MCPs board task
94ac1c4f). So every scheduled reader runs a native interpreter and parses
the file here.

ONE PARSER, LIFTED. The hpc-wake pump's ``scripts/run_cycle.py`` carried
this grammar first; the fleet ticks are its second reader, and a second copy
is how two readers come to disagree about one file. The file's bytes are
the caller's to read, through its own seam, the way every consumer of
:mod:`platform_core.mcp_client` passes its own poster.

STRICT BY DESIGN. The only lines accepted are blank ones, ``#`` comments,
and ``$env:NAME = 'value'``. Anything else is refused rather than skipped,
because a credential this parser silently passed over would surface later as
an unauthenticated run, far from its cause. A name assigned twice is refused
too: which one "wins" would otherwise depend on the reader. A refusal names
the line NUMBER and never repeats the line, since a mis-quoted credential is
exactly the line that would be echoed into a log.
"""

from __future__ import annotations

import re
from typing import Final

from platform_core.errors import AppError, ErrorCode

#: What a PowerShell editor writes before a UTF-8 file's first line.
BYTE_ORDER_MARK: Final = "\N{BYTE ORDER MARK}"

#: A plain single-quoted assignment, the only statement the file may hold.
_ASSIGNMENT: Final = re.compile(r"^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*'([^']*)'\s*$")


def parse_env_assignments(text: str, *, source: str) -> tuple[tuple[str, str], ...]:
    """Parse the credentials file's assignments.

    Args:
        text: The file's contents, already decoded (a leading byte-order
            mark, which PowerShell editors write, is ignored).
        source: What the text was read from, for the refusal message.

    Returns:
        ``(name, value)`` pairs in file order, each name once.

    Raises:
        AppError: ``CONFIG_ERROR`` naming the source and line number for a
            non-blank, uncommented line that is not a plain single-quoted
            assignment, or for a name assigned a second time.
    """
    assignments: list[tuple[str, str]] = []
    seen: set[str] = set()
    for number, raw in enumerate(text.removeprefix(BYTE_ORDER_MARK).splitlines(), start=1):
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        matched = _ASSIGNMENT.match(line)
        if matched is None:
            raise AppError(
                ErrorCode.CONFIG_ERROR,
                f"line {number} of {source} is not a plain $env:NAME = 'value' assignment; "
                f"only those, blank lines and # comments are read, and skipping one would "
                f"run with a credential missing",
            )
        name = matched.group(1)
        if name in seen:
            raise AppError(
                ErrorCode.CONFIG_ERROR,
                f"line {number} of {source} assigns {name} a second time; each name is set once",
            )
        seen.add(name)
        assignments.append((name, matched.group(2)))
    return tuple(assignments)


__all__ = ["BYTE_ORDER_MARK", "parse_env_assignments"]
