"""Turning newly terminal jobs into board posts -- pure functions only.

One post per (submitter, project) group per cycle, not one per job. The
ledger's heaviest cluster user submits 136-member arrays; a post per member
would bury the feed the moment the bridge worked, and burying the feed is a
notification system failing in the other direction. Grouping keeps one
sweep's ending to one note, and the ``@mention`` still lands on exactly the
label that was waiting.

The first token of every body is :data:`MARKER`, so the posts are findable
by ``task_feed(query=...)`` -- the one board surface that searches bodies --
without depending on any render grammar of the board's own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hpc3.contracts.closure import Closure
from hpc3.contracts.ledger import LedgerEntry
from platform_core.error_codes import HpcWakeErrorCode
from platform_core.errors import AppError
from typing_extensions import TypedDict

#: First token of every announcement body; the machine-searchable marker.
MARKER = "JOB-TERMINAL"

#: How many per-job lines a post carries before summarising the rest.
LINE_CAP = 20


class Announcement(TypedDict):
    """One board post's worth of endings.

    Attributes:
        submitter: The agent label the post tags, or ``""`` when the ledger
            recorded that no label was declared -- the post is still made,
            because the feed is still the record, but it mentions nobody.
        project: The hpc3 project every job in the post belongs to.
        body: The full post text, marker first, mention last.
    """

    submitter: str
    project: str
    body: str


def _entry_for(closure: Closure, entries_by_id: Mapping[str, LedgerEntry]) -> LedgerEntry:
    """Find the ledger entry a closure belongs to.

    Args:
        closure: The observed ending.
        entries_by_id: Every ledger entry, keyed by job id.

    Returns:
        The matching entry.

    Raises:
        AppError: ``JOB_UNKNOWN_TO_LEDGER`` when there is none. The
            accounting query is built FROM the ledger, so an unknown id here
            means an expansion or parsing defect, and a skipped announcement
            is that defect hidden behind a quiet cycle.
    """
    entry = entries_by_id.get(closure["job_id"])
    if entry is None:
        raise AppError(
            code=HpcWakeErrorCode.JOB_UNKNOWN_TO_LEDGER,
            message=(
                f"accounting reported terminal job {closure['job_id']!r} but the "
                "ledger holds no such id; the query is built from the ledger, so "
                "this is a defect, not a stray job"
            ),
        )
    return entry


def _body(project: str, submitter: str, items: Sequence[tuple[Closure, LedgerEntry]]) -> str:
    """Render one group's post.

    Args:
        project: The group's project.
        submitter: The label to mention, or ``""`` for nobody.
        items: The group's endings with their ledger entries, in the order
            accounting reported them.

    Returns:
        The post text.
    """
    counts: dict[str, int] = {}
    for closure, _entry in items:
        counts[closure["state"]] = counts.get(closure["state"], 0) + 1
    tally = ", ".join(f"{state} x{count}" for state, count in sorted(counts.items()))

    lines = [f"{MARKER} {project}: {len(items)} job(s) ended ({tally})"]
    for closure, entry in items[:LINE_CAP]:
        elapsed = closure["elapsed_seconds"]
        duration = "elapsed unrecorded" if elapsed is None else f"{elapsed}s"
        lines.append(f"{closure['job_id']} {entry['name']} {closure['state']} {duration}")
    if len(items) > LINE_CAP:
        lines.append(f"+{len(items) - LINE_CAP} more, all in the ledger's closure record")
    if submitter != "":
        lines.append(f"@{submitter} your job(s) reached terminal state")
    return "\n".join(lines)


def announcements(
    closures: Sequence[Closure], entries_by_id: Mapping[str, LedgerEntry]
) -> list[Announcement]:
    """Group newly terminal jobs into one post per (submitter, project).

    Args:
        closures: The endings this cycle observed, in accounting order.
        entries_by_id: Every ledger entry, keyed by job id.

    Returns:
        One announcement per group, ordered by (project, submitter) so a
        cycle's output is deterministic. A ledger row whose ``submitter`` is
        ``None`` -- written before the field existed -- groups with the
        declared-none rows: either way there is nobody to tag.

    Raises:
        AppError: ``JOB_UNKNOWN_TO_LEDGER`` via :func:`_entry_for`.
    """
    groups: dict[tuple[str, str], list[tuple[Closure, LedgerEntry]]] = {}
    for closure in closures:
        entry = _entry_for(closure, entries_by_id)
        recorded = entry["submitter"]
        submitter = "" if recorded is None else recorded
        groups.setdefault((entry["project"], submitter), []).append((closure, entry))

    return [
        Announcement(
            submitter=submitter,
            project=project,
            body=_body(project, submitter, items),
        )
        for (project, submitter), items in sorted(groups.items())
    ]


__all__ = ["LINE_CAP", "MARKER", "Announcement", "announcements"]
