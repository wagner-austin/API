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

from board_watch.config import load_credentials
from platform_core.board import post_to_task

from ci_wake import _test_hooks
from ci_wake.announce import PushReport, RunReport, announcements
from ci_wake.enrolment import PushAttempt, attempt_key, latest_attempts, read_attempts
from ci_wake.identity import IDENTITY, load_task_id
from ci_wake.position import AnnouncedPush, append_announced, position_path, read_announced
from ci_wake.runs import decode_jobs, decode_runs, gh_json, jobs_argv, runs_argv
from ci_wake.verdicts import PushVerdict, classify, is_announceable


def _report(attempt: PushAttempt, now_epoch: int) -> PushReport | None:
    """Ask GitHub about one outstanding push and decide what to say.

    Args:
        attempt: The enrolment row.
        now_epoch: Unix seconds, shared by the whole cycle.

    Returns:
        The push and its runs when it is announceable, or None when it is
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
    return PushReport(
        verdict=verdict,
        reports=tuple(
            RunReport(
                run=run,
                tally=decode_jobs(gh_json(jobs_argv(attempt["repo"], run["run_id"]))),
            )
            for run in runs
        ),
    )


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
    reports = [
        report
        for report in (_report(attempt, now_epoch) for attempt in outstanding)
        if report is not None
    ]
    if len(reports) == 0:
        _test_hooks.emit(f"{len(outstanding)} push(es) outstanding, none decided yet")
        return

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
    _test_hooks.emit(
        f"cycle: {len(attempts)} enrolled, {len(outstanding)} outstanding, "
        f"{len(reports)} announced, positions recorded"
    )


__all__ = ["run_cycle"]
