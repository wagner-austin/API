"""Turning decided pushes into board posts -- pure functions only.

ONE POST PER (repository, pusher) PER CYCLE, not one per push and not one
per run. A session that lands three commits in five minutes gets one note,
and a repository with four workflows gets one note rather than four. Both
matter: three notes for one intent buries the feed the moment the bridge
works, which is a notification system failing in the other direction, and
four notes for one commit is the multi-workflow category error made into a
delivery mechanism.

THE FIRST TOKEN OF EVERY BODY IS :data:`MARKER`, so the posts are findable
by ``task_feed(query=...)`` -- the one board surface that searches post
bodies -- without depending on any render grammar of the board's own.

EVERY RUN LINE CARRIES WORKFLOW, OUTCOME, JOB COUNT AND URL, and that is the
whole argument of the ``mcps-codebase`` wiki page ``reading-ci-run-outcomes``
compressed into a line format:

* the WORKFLOW, because a repository can be green on one and red on another
  for one sha, and a verdict that names neither is about no workflow;
* the JOB COUNT, because both repositories narrow their matrix to the
  changed paths, so a green that examined one workspace and a green that
  examined forty-three are the same word;
* and for ``cancelled``, the count decides which of TWO DIFFERENT EVENTS it
  was -- a run superseded mid-flight ran something before it died, a run
  evicted from the concurrency queue never created a job at all. This is the
  one place where a bare conclusion is not merely thin but wrong, so it is
  the one place the rendering refuses to print the word alone.

NEITHER CANCELLATION IS RENDERED AS BENIGN, AND THAT IS A CORRECTION. This
module first printed "cancelled (superseded mid-flight)", taken from the wiki
page's framing that supersession is normally harmless -- "the older answer is
about stale code, so discarding it costs nothing". That reasoning assumes the
newer run re-checks the same code, and under a PATH-NARROWED matrix it does
not: a workflow that diffs ``event.before..sha`` gives the superseding push a
window starting at the superseded one, so the superseded push's changes fall
between windows and no later run ever covers them. Measured in ``wagner-austin/
API`` on 2026-09-09 by ``fable-brain-audit-0903``: a push creating a 28-file
package was superseded and got ZERO CI executions while the branch showed
green.

The phrase says "may" rather than "are" because that is the strongest claim
this bridge can support. Whether it bites depends on the repository's
concurrency policy -- ``wagner-austin/MCPs`` cancels nothing, so it is immune
-- and the Actions runs API does not expose a workflow's ``cancel-in-progress``
setting. Asserting the stronger form would be inventing a fact about a file
this package never reads.

FAILURES ARE NOT SOFTENED AND ARE NAMED. A post that said "failure" without
saying which jobs would send its reader to the run page to learn the thing
the bridge already knew.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from typing_extensions import TypedDict

from ci_wake.runs import JobTally, WorkflowRun, is_terminal
from ci_wake.verdicts import NO_RUN_SECONDS, STALL_SECONDS, PushVerdict

#: First token of every announcement body; the machine-searchable marker.
#: Deliberately the same shape as ``JOB-TERMINAL`` and ``DISPATCH-TERMINAL``,
#: so one query finds every wake bridge's output.
MARKER = "CI-TERMINAL"

#: How many pushes a post details before summarising the rest.
LINE_CAP: Final = 10

#: How many failed job names one run line lists before summarising.
FAILED_CAP: Final = 8

#: How short a sha is rendered. Seven is what git and GitHub both show.
_SHORT_SHA: Final = 7


class RunReport(TypedDict):
    """One run, joined to what its jobs came to.

    Attributes:
        run: The run as GitHub described it.
        tally: Its jobs. Fetched for unfinished runs as well as finished
            ones, so a stalled push can say how much work is hanging rather
            than only that something is.
    """

    run: WorkflowRun
    tally: JobTally


class PushReport(TypedDict):
    """One decided push and every run belonging to it.

    Attributes:
        verdict: The push and the state this cycle put it in.
        reports: Its runs, in GitHub's order. Empty when the push was
            abandoned, which is the state's whole definition.
    """

    verdict: PushVerdict
    reports: tuple[RunReport, ...]


class Announcement(TypedDict):
    """One board post's worth of verdicts.

    Attributes:
        repo: The repository every push in the post belongs to.
        agent: The board label the post tags, or the empty string when the
            pushing session exported none.
        body: The full post text, marker first, mention last.
    """

    repo: str
    agent: str
    body: str


def _outcome_phrase(report: RunReport) -> str:
    """Say what one run's outcome was, in words a reader can act on.

    Args:
        report: The run and its job tally.

    Returns:
        The phrase. ``cancelled`` is never returned bare -- see this
        module's docstring on why that one word is two different events.
    """
    run = report["run"]
    if not is_terminal(run):
        return f"STILL {run['status']}"
    if run["conclusion"] != "cancelled":
        return run["conclusion"]
    if report["tally"]["total"] == 0:
        return "cancelled (EVICTED FROM THE QUEUE, no job ever ran)"
    return "cancelled (SUPERSEDED -- its changes may be in no later run's diff window)"


def _jobs_phrase(tally: JobTally) -> str:
    """Say how many jobs a run had and which of them failed.

    Args:
        tally: The run's jobs.

    Returns:
        The phrase, always naming the total. The failed names follow when
        there are any, capped at :data:`FAILED_CAP` with the remainder
        counted rather than dropped.
    """
    parts = [f"{tally['total']} jobs"]
    if tally["listed"] < tally["total"]:
        parts.append(f"{tally['listed']} listed, so the failed list below is partial")
    if len(tally["failed"]) > 0:
        named = ", ".join(tally["failed"][:FAILED_CAP])
        overflow = len(tally["failed"]) - FAILED_CAP
        suffix = f" +{overflow} more" if overflow > 0 else ""
        parts.append(f"{len(tally['failed'])} failed: {named}{suffix}")
    return " -- ".join(parts)


def _push_lines(report: PushReport) -> list[str]:
    """Render one push and its runs.

    Args:
        report: The decided push and its runs.

    Returns:
        The lines, headed by the short sha and the ref.
    """
    attempt = report["verdict"]["attempt"]
    lines = [f"{attempt['sha'][:_SHORT_SHA]} {attempt['ref']}"]
    if report["verdict"]["state"] == "abandoned":
        lines.append(
            f"  NO RUN EVER APPEARED -- {NO_RUN_SECONDS // 60}m after the push, GitHub "
            "lists no workflow run for this sha. Either the push was refused, no "
            "workflow's path filters matched it, or Actions is starting no jobs."
        )
        return lines
    if report["verdict"]["state"] == "stalled":
        lines.append(
            f"  NO VERDICT AFTER {STALL_SECONDS // 3600}h -- this row is now closed and "
            "will not be announced again. Watch the runs below yourself."
        )
    for entry in report["reports"]:
        run = entry["run"]
        outcome = _outcome_phrase(entry)
        lines.append(f"  {run['workflow']} {outcome} -- {_jobs_phrase(entry['tally'])}")
        lines.append(f"  {run['html_url']}")
    return lines


def _tally_phrase(reports: Sequence[PushReport]) -> str:
    """Summarise a group's outcomes for the post's first line.

    Args:
        reports: The group's pushes.

    Returns:
        A sorted ``outcome xN`` list over every run in the group, plus the
        pushes that produced no run at all. Abandoned pushes are counted
        under ``no-run`` rather than omitted -- a header whose numbers did
        not add up to the lines beneath it would be the first thing a
        reader distrusted.
    """
    counts: dict[str, int] = {}
    for report in reports:
        if len(report["reports"]) == 0:
            counts["no-run"] = counts.get("no-run", 0) + 1
        for entry in report["reports"]:
            run = entry["run"]
            outcome = run["conclusion"] if is_terminal(run) else f"still-{run['status']}"
            counts[outcome] = counts.get(outcome, 0) + 1
    return ", ".join(f"{outcome} x{count}" for outcome, count in sorted(counts.items()))


def _mention_line(agent: str) -> str:
    """Address the post, or say plainly that it cannot be addressed.

    Args:
        agent: The pushing session's board label, or the empty string.

    Returns:
        The final line of the body.
    """
    if agent == "":
        return (
            "This push exported no BOARD_AGENT_LABEL, so there is nobody to tag. "
            "Export it before pushing and the next verdict finds you."
        )
    return f"@{agent} your push has its CI verdict"


def _body(repo: str, agent: str, reports: Sequence[PushReport]) -> str:
    """Render one group's post.

    Args:
        repo: The group's repository.
        agent: The label to mention, or the empty string.
        reports: The group's decided pushes, in enrolment order.

    Returns:
        The post text.
    """
    total_runs = sum(len(report["reports"]) for report in reports)
    lines = [
        f"{MARKER} {repo}: {len(reports)} push(es), {total_runs} run(s) ({_tally_phrase(reports)})"
    ]
    for report in reports[:LINE_CAP]:
        lines.extend(_push_lines(report))
    if len(reports) > LINE_CAP:
        lines.append(f"+{len(reports) - LINE_CAP} more, all in the enrolment record")
    lines.append(_mention_line(agent))
    return "\n".join(lines)


def announcements(reports: Sequence[PushReport]) -> list[Announcement]:
    """Group decided pushes into one post per (repository, pusher).

    Args:
        reports: The pushes this cycle decided to announce, in enrolment
            order.

    Returns:
        One announcement per group, ordered by (repo, agent) so a cycle's
        output is deterministic and two runs over the same input post the
        same thing in the same order.
    """
    groups: dict[tuple[str, str], list[PushReport]] = {}
    for report in reports:
        attempt = report["verdict"]["attempt"]
        groups.setdefault((attempt["repo"], attempt["agent"]), []).append(report)
    return [
        Announcement(repo=repo, agent=agent, body=_body(repo, agent, grouped))
        for (repo, agent), grouped in sorted(groups.items())
    ]


__all__ = [
    "FAILED_CAP",
    "LINE_CAP",
    "MARKER",
    "Announcement",
    "PushReport",
    "RunReport",
    "announcements",
]
