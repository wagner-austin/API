"""Which cache generation a check ran in, and where its time went.

MCPs board tasks abb0a10e and 74d10d85. Four checks overran their 300 s
budget on 2026-11-01 because each was its package's first on its node in a
cache generation without its cp311 wheels, and fetched them inside the
check's clock. The fix moved those fetches into an install phase named
``deps`` before the clock, and its proof was a check that began a fresh
generation with its fetch in ``deps`` and its ``check`` phase inside budget.
Both audits found that proof unreadable by any query: the queue row said
``budget=within:213s/300s`` and nothing else, the same as a warm run's, and
the generation and the phase seconds were in the transcript and a cache
directory's name, which a session read over ssh.

The phase lines are no help to the verdict either: it reads the transcript's
last :data:`~fleet.core.verdict.LOG_TAIL_LINES` lines, and a long suite pushes
the install phases far above them (covenant_ml's 10,865-line transcript puts
its two ``deps`` lines 10,800 lines up). So every build prints one line
last, after the recipe or a failed install step::

    FLEET_RUN: generation=1793525277 began=yes phases=deps:7s,deps:46s,check:214s

naming the generation it read (``$generation``, :mod:`fleet.core.tool_caches`),
whether it began it, and each phase's whole seconds in order
(``$phases``, :mod:`fleet.core.phase_markers`). The verdict copies the three
onto the queue row and the thread (:func:`fleet.core.verdict.judge`).
"""

from __future__ import annotations

import re
from typing import Final

from typing_extensions import TypedDict

#: The transcript line's opening token.
RUN_SUMMARY_MARKER: Final = "FLEET_RUN"

#: The line, as :func:`sh_run_summary_line` prints it.
RUN_SUMMARY: Final[re.Pattern[str]] = re.compile(
    rf"^{RUN_SUMMARY_MARKER}: generation=(\S+) began=(yes|no) phases=(\S+)$", re.MULTILINE
)

#: What a field reads when the tail holds no summary line.
UNREAD: Final = "unread"


class RunSummary(TypedDict):
    """What one build's summary line says.

    Attributes:
        generation: The cache generation the build read, or ``unread``.
        began: ``yes`` when this build began that generation, ``no`` when an
            earlier build had, or ``unread``.
        phases: ``<phase>:<seconds>s`` for each phase in order, comma-joined;
            ``none`` for a build that ran none; or ``unread``.
    """

    generation: str
    began: str
    phases: str


def sh_run_summary_line(log: str) -> str:
    """The ``sh`` line that prints the summary into the transcript.

    Args:
        log: The transcript's path, appended to.

    Returns:
        The line, run after the generation is read and after every phase.
    """
    return (
        f"printf '{RUN_SUMMARY_MARKER}: generation=%s began=%s phases=%s\\n' "
        f'"$generation" "$generation_began" "${{phases:-none}}" >> \'{log}\''
    )


def read_run_summary(tail: str) -> RunSummary:
    """Read the build's summary off its transcript tail.

    Args:
        tail: The transcript's last lines.

    Returns:
        The last summary line's three fields, or each :data:`UNREAD` for a
        build from before the line or one stopped before it printed.
    """
    last: re.Match[str] | None = None
    for match in RUN_SUMMARY.finditer(tail):
        last = match
    if last is None:
        return RunSummary(generation=UNREAD, began=UNREAD, phases=UNREAD)
    return RunSummary(
        generation=f"{last.group(1)}", began=f"{last.group(2)}", phases=f"{last.group(3)}"
    )


__all__ = [
    "RUN_SUMMARY",
    "RUN_SUMMARY_MARKER",
    "UNREAD",
    "RunSummary",
    "read_run_summary",
    "sh_run_summary_line",
]
