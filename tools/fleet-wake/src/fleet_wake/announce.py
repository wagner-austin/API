"""Turning newly terminal dispatches into board posts -- pure functions only.

ONE POST PER (agent, project) GROUP PER CYCLE, not one per dispatch. A session
that fans a suite across three nodes ends three dispatches within seconds of
each other, and three notes for one intent buries the feed the moment the
bridge works -- which is a notification system failing in the other direction.
Grouping keeps one fan-out's ending to one note, and the ``@mention`` still
lands on exactly the label that dispatched.

THE FIRST TOKEN OF EVERY BODY IS :data:`MARKER`, so the posts are findable by
``task_feed(query=...)`` -- the one board surface that searches post bodies --
without depending on any render grammar of the board's own.

THE OUTCOME TALLY LEADS, AND FAILURES ARE NOT SOFTENED. ``refused``,
``failed``, ``cancelled`` and ``lost`` read exactly as they are; a bridge that
reported only the passes would be the wedge detector's opposite. ``lost`` in
particular is the one nobody else can report: a dispatch whose lease expired
with no result cannot announce its own death, which is what being wedged
means.
"""

from __future__ import annotations

from collections.abc import Sequence

from fleet.contracts.ledger import NO_EXIT_CODE, LedgerEntry, is_live
from typing_extensions import TypedDict

#: First token of every announcement body; the machine-searchable marker.
MARKER = "DISPATCH-TERMINAL"

#: How many per-dispatch lines a post carries before summarising the rest.
LINE_CAP = 20


class Announcement(TypedDict):
    """One board post's worth of endings.

    Attributes:
        agent: The board label the post tags -- the session that dispatched.
            Never empty: ``fleet-run`` requires ``--agent`` and the ledger
            records it on every row, so unlike hpc-wake's submitter there is
            no untagged case to carry.
        project: The repo-relative project every dispatch in the post ran.
        body: The full post text, marker first, mention last.
    """

    agent: str
    project: str
    body: str


def _exit_phrase(entry: LedgerEntry) -> str:
    """Say what the recipe's exit status was, or that there was none.

    A refused dispatch never started and a lost one never reported, so
    neither has an exit code. Spelling that as ``exit -1`` would be
    arithmetic on a number that means something else, and every reader would
    have to know the convention -- which is exactly why the ledger keeps
    ``outcome`` and ``exit_code`` as separate fields.

    Args:
        entry: The dispatch's current ledger row.

    Returns:
        The phrase to put on the dispatch's line.
    """
    if entry["exit_code"] == NO_EXIT_CODE:
        return "no exit code"
    return f"exit {entry['exit_code']}"


def _body(project: str, agent: str, entries: Sequence[LedgerEntry]) -> str:
    """Render one group's post.

    Args:
        project: The group's project.
        agent: The label to mention.
        entries: The group's terminal rows, in ledger order.

    Returns:
        The post text.
    """
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["outcome"]] = counts.get(entry["outcome"], 0) + 1
    tally = ", ".join(f"{outcome} x{count}" for outcome, count in sorted(counts.items()))

    lines = [f"{MARKER} {project}: {len(entries)} dispatch(es) ended ({tally})"]
    for entry in entries[:LINE_CAP]:
        elapsed = entry["ended_unix"] - entry["started_unix"]
        lines.append(
            f"{entry['run_id']} {entry['node']} {entry['outcome']} {_exit_phrase(entry)} {elapsed}s"
        )
    if len(entries) > LINE_CAP:
        lines.append(f"+{len(entries) - LINE_CAP} more, all in the workspace ledger")
    lines.append(f"@{agent} your dispatch(es) reached terminal state")
    return "\n".join(lines)


def announcements(entries: Sequence[LedgerEntry]) -> list[Announcement]:
    """Group newly terminal dispatches into one post per (agent, project).

    Args:
        entries: The terminal rows this cycle observed, in ledger order.

    Returns:
        One announcement per group, ordered by (project, agent) so a cycle's
        output is deterministic and two runs over the same input post the
        same thing in the same order.
    """
    groups: dict[tuple[str, str], list[LedgerEntry]] = {}
    for entry in entries:
        groups.setdefault((entry["project"], entry["agent"]), []).append(entry)

    return [
        Announcement(agent=agent, project=project, body=_body(project, agent, grouped))
        for (project, agent), grouped in sorted(groups.items())
    ]


def terminal_unannounced(
    rows: Sequence[LedgerEntry], announced: frozenset[str]
) -> tuple[LedgerEntry, ...]:
    """Select the dispatches this cycle should announce.

    TERMINALITY IS ``fleet.contracts.ledger.is_live``'s ANSWER, INVERTED, and
    is not re-derived here. That function owns which outcomes mean a dispatch
    still holds resources, the capacity check already depends on it, and a
    second list of terminal outcomes in this package would be the fork that
    goes stale the day a new outcome is added -- announcing nothing for it
    while the ledger quietly records it.

    Args:
        rows: The CURRENT row for each dispatch, from
            ``fleet.core.records.latest_rows``. Current rather than every
            row: the ledger is append-only, so a finished dispatch still has
            its ``running`` row in the file, and reading every row would
            announce each dispatch twice.
        announced: Run ids already posted about.

    Returns:
        The rows to announce, in ledger order.
    """
    return tuple(row for row in rows if not is_live(row) and row["run_id"] not in announced)


__all__ = [
    "LINE_CAP",
    "MARKER",
    "Announcement",
    "announcements",
    "terminal_unannounced",
]
