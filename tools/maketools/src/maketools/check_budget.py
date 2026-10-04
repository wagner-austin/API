"""The five-minute budget on one package's ``make check`` (board task 4080d695, A6).

The operator, 2026-10-04: "ideally make check should be like 5 min for every
repo." In the seven days before, the check lock's journal recorded 1,051
test runs on austinpc totalling 80.5 hours, with packages/db's passing runs
at a median of 936 s and its fleet rows at 1,281 s, and nothing anywhere
said that was too long. This module is the thing that says it: a check that
spends more than :data:`BUDGET_SECONDS` fails, naming its time.

ONE RULE, TWO WAYS OF TIMING A CHECK. A package whose ``make test`` goes
through maketools' check lock is timed from the moment make read its
Makefile (:mod:`maketools.check_clock`) to the moment its test step
returned, LESS the time spent queued on the lock: the queue is somebody
else's run, and a budget that charged it would fail the second session to
start a check for the first session's suite. A repository with no check
lock (board task 1b152218: MCPs' own scripts/ps-harness aside, the API
repository, corvis-stick and chat) has no queue to subtract, so its
``check:`` recipe runs ``check-budget``, which runs ``make``
:data:`UNBUDGETED_TARGET` (the old check's prerequisites, renamed) and
charges the whole of it. Both arrive at :func:`budgeted_exit_code` and print
through :func:`verdict_lines`, so the limit, the exit code and the words a
reader sees are the same wherever a check runs.

THIS FILE IS LIFTED, BYTE FOR BYTE, and so imports the standard library and
nothing else. API's tools/maketools carries a copy pinned by
tools/maketools/lift-lock.json, because the API repository is public and
its CI cannot read this private one; slime and idle lift all of
src/maketools. A change here reaches them when their lift is refreshed, and
a copy that differs from its pin fails their lint.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Final, NamedTuple

#: The budget, in seconds: the operator's five minutes.
BUDGET_SECONDS: Final[int] = 300

#: The exit code of a check whose suite passed but whose time did not. Distinct
#: from a failing suite's own code (vitest and pytest exit 1, make 2) so a
#: reader of the journal or a fleet verdict can tell the two apart.
OVER_BUDGET_EXIT_CODE: Final[int] = 3

#: The target a repository with no check lock moves its check's prerequisites
#: to, which ``check-budget`` runs and times. Named, never forwarded as a
#: command line, for the reason the check lock's ``_test-unlocked`` is.
UNBUDGETED_TARGET: Final[str] = "_check-unbudgeted"

#: What the verdict says was charged for a check timed whole.
WHOLE_CHECK_SPLIT: Final[str] = "the whole make check, run by check-budget with no lock to queue on"


class CheckSpend(NamedTuple):
    """How long one package's check took, split the way the verdict prints it.

    A NamedTuple rather than a dataclass because the API repository, which
    lifts this file, bans dataclasses by guard; it is immutable all the same.

    Attributes:
        before_lock: From make reading the Makefile to the lock being asked
            for: lint, and the migration ``make test`` runs first.
        queued: Waiting for another session's run of the same package.
            Reported, never charged.
        suite: The test step, run under the lock.
    """

    before_lock: timedelta
    queued: timedelta
    suite: timedelta

    @property
    def charged(self) -> timedelta:
        """The time the budget counts: everything but the queue.

        Returns:
            ``before_lock + suite``.
        """
        return self.before_lock + self.suite

    @property
    def over_budget(self) -> bool:
        """Whether the charged time exceeds :data:`BUDGET_SECONDS`.

        Returns:
            :func:`exceeds_budget` of :attr:`charged`.
        """
        return exceeds_budget(self.charged)

    @property
    def split(self) -> str:
        """The spans as the verdict prints them.

        Returns:
            Lint and setup, suite and queue, each in whole seconds.
        """
        return (
            f"lint and setup {int(self.before_lock.total_seconds())}s, "
            f"suite {int(self.suite.total_seconds())}s, "
            f"queued {int(self.queued.total_seconds())}s (not counted)"
        )


def exceeds_budget(charged: timedelta) -> bool:
    """Whether a check's charged time is past the budget.

    Args:
        charged: The time the budget counts.

    Returns:
        True past :data:`BUDGET_SECONDS`; exactly the budget is within it.
    """
    return charged > timedelta(seconds=BUDGET_SECONDS)


def spend(
    *, started: datetime, asked: datetime, acquired: datetime, finished: datetime
) -> CheckSpend:
    """Split a locked check's time at the lock.

    Args:
        started: When make read the Makefile.
        asked: When the check lock was asked for.
        acquired: When it was held.
        finished: When the test step returned.

    Returns:
        The three spans.
    """
    return CheckSpend(
        before_lock=asked - started,
        queued=acquired - asked,
        suite=finished - acquired,
    )


def budgeted_exit_code(exit_code: int, charged: timedelta) -> int:
    """The code a timed check exits with.

    A failing check keeps its own code, so the failure a session reads is
    the suite's; a passing check past the budget fails with
    :data:`OVER_BUDGET_EXIT_CODE`.

    Args:
        exit_code: The timed make's exit code.
        charged: The time the budget counts.

    Returns:
        The exit code.
    """
    if exit_code == 0 and exceeds_budget(charged):
        return OVER_BUDGET_EXIT_CODE
    return exit_code


def verdict_lines(package: str, charged: timedelta, split: str) -> list[str]:
    """What a check prints about its budget, pass or fail.

    Args:
        package: The package, as the repository names it.
        charged: The time the budget counts.
        split: How that time divides: :attr:`CheckSpend.split` for a locked
            check, :data:`WHOLE_CHECK_SPLIT` for one timed whole.

    Returns:
        One line within budget; a loud block naming the time past it.
    """
    seconds = int(charged.total_seconds())
    if not exceeds_budget(charged):
        return [f"CHECK BUDGET: {package} took {seconds}s of {BUDGET_SECONDS}s ({split})."]
    return [
        "",
        "=" * 72,
        f"CHECK OVER BUDGET: {package} took {seconds}s, over the {BUDGET_SECONDS}s budget "
        f"by {seconds - BUDGET_SECONDS}s.",
        f"  split : {split}",
        "  why   : the operator, 2026-10-04: make check should be about 5 minutes for "
        "every repo (board task 4080d695).",
        "  fix   : make the suite lighter; nothing raises the budget.",
        "=" * 72,
    ]
