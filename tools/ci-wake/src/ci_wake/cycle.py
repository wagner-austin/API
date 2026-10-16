"""One poll: enrolment record to GitHub to board to position record.

THE ORDER IS THE DELIVERY GUARANTEE. Announcements POST before position rows
are WRITTEN, so a crash between the two repeats a post on the next cycle
rather than losing one. At-least-once, with the position file as the mark.
The alternative -- record first, then post -- turns any transport failure
into a push nobody is ever told about, which is precisely the silence this
bridge exists to remove. Same order as both sibling bridges, for the same
reason.

AND NOTHING IS CAUGHT. A refused post ends the cycle with a non-zero exit
for the scheduler to record, and the position rows are not written, so the
next cycle tries again. A ``gh`` that is not logged in ends the cycle the
same way. A bridge that swallowed either would write its position anyway and
never announce that work again -- reporting success while doing the opposite
of its job.

WHY THE GITHUB CALLS ARE SHAPED THE WAY THEY ARE. One run query per
OUTSTANDING push, not per enrolled push: a sha already announced is never
asked about again, so the cost of a cycle tracks what is in flight rather
than what has ever been pushed. The job query is then made only for pushes
that are about to be announced, which is what keeps a quiet cycle to one
cheap call per waiting push.

THE CLOCK IS READ ONCE. Classification and the position rows share it, so a
push cannot be decided against one instant and recorded against another --
which over a slow cycle would let a row be marked at a time that disagrees
with the reason it was marked.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

from board_watch.config import load_credentials
from platform_core.board import post_to_task, register_service_session

from ci_wake import _test_hooks
from ci_wake.announce import CoverReport, PushReport, RunReport, announcements
from ci_wake.coverage import BranchRun, branch_of, branch_runs, find_cover
from ci_wake.enrolment import PushAttempt, attempt_key, latest_attempts, read_attempts
from ci_wake.identity import HARNESS, IDENTITY, PURPOSE, load_task_id
from ci_wake.landing import UNLANDED, UnlandedPush, holds, landing_of
from ci_wake.position import AnnouncedPush, append_announced, position_path, read_announced
from ci_wake.runs import decode_jobs, decode_runs, gh_json, jobs_argv, runs_argv
from ci_wake.verdicts import (
    PushVerdict,
    awaits_cover,
    classify,
    is_announceable,
    is_evicted,
)


class BranchRuns:
    """A branch's completed runs, read once per cycle and shared by its pushes.

    One cycle may hold several superseded pushes to one branch; asking
    GitHub for the branch's runs once per push would multiply the one call
    that pages. The window starts at the earliest outstanding push, so every
    push's covering run is inside it.
    """

    def __init__(self, since_unix: int) -> None:
        """Start an empty cache.

        Args:
            since_unix: The earliest outstanding push's time.
        """
        self._since = since_unix
        self._runs: dict[tuple[str, str], tuple[BranchRun, ...]] = {}

    def of(self, repo: str, branch: str) -> tuple[BranchRun, ...]:
        """The branch's completed runs since the window opened.

        Args:
            repo: ``owner/name``.
            branch: The branch.

        Returns:
            Every completed run, read from GitHub on first use.
        """
        key = (repo, branch)
        if key not in self._runs:
            self._runs[key] = branch_runs(repo, branch, self._since)
        return self._runs[key]


def _covers(
    attempt: PushAttempt, own: Sequence[RunReport], branches: BranchRuns
) -> tuple[CoverReport, ...] | None:
    """Find the run that checked a superseded push, in each of its workflows.

    Args:
        attempt: The enrolment row.
        own: Its own runs that were cancelled before any job.
        branches: This cycle's branch runs.

    Returns:
        One covering run per own run, or None while any workflow has none
        yet, or when the pushed ref is not a branch.
    """
    branch = branch_of(attempt["ref"])
    if branch == "":
        return None
    runs = branches.of(attempt["repo"], branch)
    covers: list[CoverReport] = []
    for entry in own:
        cover = find_cover(
            attempt["repo"],
            attempt["sha"],
            entry["run"]["workflow"],
            attempt["attempted_unix"],
            runs,
        )
        if cover is None:
            return None
        tally = decode_jobs(gh_json(jobs_argv(attempt["repo"], cover["run"]["run_id"])))
        covers.append(CoverReport(cover=cover, tally=tally))
    return tuple(covers)


def _report(
    attempt: PushAttempt, now_epoch: int, branches: BranchRuns
) -> PushReport | UnlandedPush | None:
    """Ask GitHub about one outstanding push and decide what to say.

    A push with an own run cancelled before any job waits for the later run
    of that workflow that checked it (:mod:`ci_wake.coverage`) and is
    announced with that run's verdict beside its runs that executed; only
    past :data:`COVER_SECONDS` without one is it announced as its own runs.
    A push with no run whose branch never held it is closed unannounced.

    Args:
        attempt: The enrolment row.
        now_epoch: Unix seconds, shared by the whole cycle.
        branches: This cycle's branch runs, for a superseded push.

    Returns:
        The push and its runs when it is announceable, an
        :class:`UnlandedPush` when it never landed, or None when it is
        still worth waiting for. None rather than an empty report so the
        caller cannot accidentally post about a push that is merely quiet.

    Raises:
        AppError: ``GH_COMMAND_FAILED`` when ``gh`` refused.
        JSONTypeError: When a payload is not the shape this package reads.
    """
    runs = decode_runs(gh_json(runs_argv(attempt["repo"], attempt["sha"])))
    verdict = PushVerdict(attempt=attempt, runs=runs, state=classify(attempt, runs, now_epoch))
    if not is_announceable(verdict):
        return None
    # A push with no run is asked whether it ever landed (MCPs board task
    # be748588): the hooks enrol before git sends anything, so a refused push
    # leaves a row, and its silence means nothing. A tag has no branch to
    # ask about and keeps the announcement.
    branch = branch_of(attempt["ref"])
    if verdict["state"] == "abandoned" and branch != "":
        landing = landing_of(attempt["repo"], branch, attempt["sha"])
        if not holds(landing):
            return UnlandedPush(attempt=attempt, branch=branch, landing=landing)
    reports = tuple(
        RunReport(
            run=run,
            tally=decode_jobs(gh_json(jobs_argv(attempt["repo"], run["run_id"]))),
        )
        for run in runs
    )
    evicted = tuple(entry for entry in reports if is_evicted(entry["run"], entry["tally"]["total"]))
    if verdict["state"] == "ripe" and len(evicted) > 0:
        covers = _covers(attempt, evicted, branches)
        if covers is not None:
            return PushReport(
                verdict=PushVerdict(attempt=attempt, runs=runs, state="covered"),
                reports=reports,
                covers=covers,
            )
        if awaits_cover(attempt, now_epoch):
            return None
    return PushReport(verdict=verdict, reports=reports, covers=())


def run_cycle(enrolment: pathlib.Path) -> None:
    """Run one bridge cycle against one enrolment record.

    Args:
        enrolment: Path to the record the ``pre-push`` hooks append to.

    Raises:
        AppError: Configuration (missing credentials or task id), the
            ``gh`` boundary, a malformed enrolment field, or the board
            refusing a post.
        JSONTypeError: An enrolment row, position line, or GitHub payload
            that does not decode.
        OSError: A record that cannot be read or written.
    """
    credentials = load_credentials()
    task_id = load_task_id()

    attempts = latest_attempts(read_attempts(enrolment))
    if attempts == ():
        _test_hooks.emit("enrolment record is empty; nothing has been pushed from this machine")
        return

    marks = position_path(enrolment)
    announced = read_announced(marks)
    outstanding = tuple(
        attempt
        for attempt in attempts
        if attempt_key(attempt["repo"], attempt["sha"]) not in announced
    )
    if outstanding == ():
        _test_hooks.emit(f"{len(attempts)} push(es) enrolled, all already announced")
        return

    now_epoch = _test_hooks.now()
    branches = BranchRuns(min(attempt["attempted_unix"] for attempt in outstanding))
    decided = [
        report
        for report in (_report(attempt, now_epoch, branches) for attempt in outstanding)
        if report is not None
    ]
    if len(decided) == 0:
        _test_hooks.emit(f"{len(outstanding)} push(es) outstanding, none decided yet")
        return
    unlanded = [push for push in decided if isinstance(push, UnlandedPush)]
    reports = [report for report in decided if not isinstance(report, UnlandedPush)]

    if len(reports) > 0:
        # THE LEDGER GATE COMES FIRST, and it is why this bridge went silent
        # from 2026-09-16 to 2026-09-21: MCPs mig 514 refuses a write from a
        # session no ledger surface knows, a service session is exactly one,
        # and every tick since died on TASK_SESSION_UNLEDGERED with the
        # traceback going only to runs/cycle.log, which nothing reads. Placed
        # here rather than at the top of the cycle so a quiet tick stays
        # quiet: by this line there is an announcement to make, and before it
        # there was not.
        register_service_session(
            _test_hooks.http_post,
            credentials,
            IDENTITY,
            harness=HARNESS,
            purpose=PURPOSE,
        )
        for announcement in announcements(reports):
            post_to_task(
                _test_hooks.http_post,
                credentials,
                IDENTITY,
                task_id=task_id,
                kind="note",
                body=announcement["body"],
            )
            _test_hooks.emit(
                f"posted {announcement['repo']}: tagged @{announcement['agent']}"
                if announcement["agent"] != ""
                else f"posted {announcement['repo']}: unaddressed, no BOARD_AGENT_LABEL"
            )

    for report in reports:
        attempt = report["verdict"]["attempt"]
        append_announced(
            marks,
            AnnouncedPush(
                key=attempt_key(attempt["repo"], attempt["sha"]),
                state=report["verdict"]["state"],
                announced_unix=now_epoch,
            ),
        )
    for push in unlanded:
        append_announced(
            marks,
            AnnouncedPush(
                key=attempt_key(push.attempt["repo"], push.attempt["sha"]),
                state=UNLANDED,
                announced_unix=now_epoch,
            ),
        )
        _test_hooks.emit(push.line())
    _test_hooks.emit(
        f"cycle: {len(attempts)} enrolled, {len(outstanding)} outstanding, "
        f"{len(reports)} announced, {len(unlanded)} closed unannounced as never landed, "
        "positions recorded"
    )


__all__ = ["BranchRuns", "run_cycle"]
