"""The policy, over simulated time, with no clock and no fake anywhere.

Every function under test is pure and takes ``now_epoch``, so these tests
move time by choosing a number rather than by freezing anything. That is the
point of the split: a settling rule tested through a clock fake is a test of
the fake's arithmetic.

THE TABLE BELOW IS THE SPECIFICATION. Each row is a state the bridge can be
in, and three of the four are announced -- because this bridge's failure mode
is silence, and a push that never produces a verdict is exactly as silent as
a bridge that is switched off.
"""

from __future__ import annotations

from ci_wake.enrolment import PushAttempt
from ci_wake.runs import WorkflowRun
from ci_wake.verdicts import (
    CREATION_SETTLE_SECONDS,
    NO_RUN_SECONDS,
    STALL_SECONDS,
    PushVerdict,
    classify,
    is_announceable,
)
from tests.conftest import AGENT, REPO, SHA

PUSHED_AT = 1788700000


def _attempt() -> PushAttempt:
    """Build the one enrolment row every test here classifies.

    Returns:
        The row, pushed at :data:`PUSHED_AT`.
    """
    return PushAttempt(
        repo=REPO, sha=SHA, ref="refs/heads/main", agent=AGENT, attempted_unix=PUSHED_AT
    )


def _run(*, status: str = "completed", conclusion: str = "success") -> WorkflowRun:
    """Build one run in a given state.

    Args:
        status: The run's status.
        conclusion: Its conclusion.

    Returns:
        The run.
    """
    return WorkflowRun(
        run_id=1,
        workflow="Check",
        status=status,
        conclusion=conclusion,
        html_url="https://example.test/1",
    )


class TestNoRunsYet:
    def test_a_fresh_push_with_no_runs_is_held(self) -> None:
        """Runs are created seconds after a push, not instantly, and calling
        that gap "abandoned" would announce failure for every push."""
        assert classify(_attempt(), [], PUSHED_AT + 5) == "holding"

    def test_it_is_still_held_one_second_before_the_horizon(self) -> None:
        assert classify(_attempt(), [], PUSHED_AT + NO_RUN_SECONDS - 1) == "holding"

    def test_at_the_horizon_it_is_abandoned(self) -> None:
        """The horizon is inclusive so the state is reachable at exactly the
        boundary rather than one cycle later."""
        assert classify(_attempt(), [], PUSHED_AT + NO_RUN_SECONDS) == "abandoned"

    def test_a_long_dead_push_is_abandoned(self) -> None:
        """A push git refused as non-fast-forward was enrolled before the
        transfer was attempted, and no later hook could retract it."""
        assert classify(_attempt(), [], PUSHED_AT + 86400) == "abandoned"


class TestUnfinishedRuns:
    def test_a_running_run_is_held(self) -> None:
        assert classify(_attempt(), [_run(status="in_progress", conclusion="")], PUSHED_AT) == (
            "holding"
        )

    def test_one_unfinished_run_holds_the_whole_push(self) -> None:
        """A repository can be green on one workflow and red on another for
        one sha, so announcing on the finished half is a verdict about no
        workflow."""
        runs = [_run(), _run(status="queued", conclusion="")]

        assert classify(_attempt(), runs, PUSHED_AT + STALL_SECONDS - 1) == "holding"

    def test_past_the_stall_horizon_it_is_announced_as_stalled(self) -> None:
        """A wedged runner, a job the host ran out of memory under, a queue
        nobody is draining. Each is silence, and each has happened here."""
        runs = [_run(), _run(status="in_progress", conclusion="")]

        assert classify(_attempt(), runs, PUSHED_AT + STALL_SECONDS) == "stalled"


class TestEveryRunFinished:
    def test_a_verdict_arriving_inside_the_settle_window_is_held(self) -> None:
        """A very fast workflow can finish before a slower sibling has been
        CREATED. Announcing then reports a one-workflow verdict for a commit
        that had three."""
        assert classify(_attempt(), [_run()], PUSHED_AT + CREATION_SETTLE_SECONDS - 1) == "holding"

    def test_past_the_settle_window_it_is_ripe(self) -> None:
        assert classify(_attempt(), [_run()], PUSHED_AT + CREATION_SETTLE_SECONDS) == "ripe"

    def test_a_failure_is_as_ripe_as_a_success(self) -> None:
        runs = [_run(conclusion="failure")]

        assert classify(_attempt(), runs, PUSHED_AT + 3600) == "ripe"

    def test_a_cancelled_run_is_terminal_and_therefore_ripe(self) -> None:
        """``cancelled`` is a conclusion, not a state. Holding for one would
        wait forever for a run that is already over -- and the evicted kind
        is precisely the case somebody needs to hear about."""
        runs = [_run(conclusion="cancelled")]

        assert classify(_attempt(), runs, PUSHED_AT + 3600) == "ripe"


class TestHorizonsAreOrdered:
    def test_the_settle_window_is_shorter_than_both_horizons(self) -> None:
        """Not decoration. If the settle window outran the no-run horizon, a
        push whose runs finished quickly would be abandoned before it could
        ever be called ripe."""
        assert CREATION_SETTLE_SECONDS < NO_RUN_SECONDS < STALL_SECONDS


class TestIsAnnounceable:
    def test_ripe_wakes_somebody(self) -> None:
        assert is_announceable(PushVerdict(attempt=_attempt(), runs=(), state="ripe")) is True

    def test_abandoned_wakes_somebody(self) -> None:
        assert is_announceable(PushVerdict(attempt=_attempt(), runs=(), state="abandoned")) is True

    def test_stalled_wakes_somebody(self) -> None:
        assert is_announceable(PushVerdict(attempt=_attempt(), runs=(), state="stalled")) is True

    def test_holding_says_nothing(self) -> None:
        """The only silent state, and the predicate is written as a negation
        of it so a state added later is announced by default -- wrong in a
        way somebody notices, rather than dropped in a way nobody does."""
        assert is_announceable(PushVerdict(attempt=_attempt(), runs=(), state="holding")) is False
