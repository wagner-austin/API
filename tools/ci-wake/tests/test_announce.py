"""What a post says, and the three things a bare conclusion would not.

THESE ARE THE TESTS THAT PIN THE WIKI PAGE. ``reading-ci-run-outcomes``
(mcps-codebase) measures how a run list misleads a reader; the assertions
below are that argument turned into a rendering contract, so a later
simplification of the line format fails a test rather than quietly making
every announcement less true than it was.
"""

from __future__ import annotations

from typing import Literal

from ci_wake.announce import FAILED_CAP, LINE_CAP, MARKER, PushReport, RunReport, announcements
from ci_wake.enrolment import PushAttempt
from ci_wake.runs import JobTally, WorkflowRun
from ci_wake.verdicts import PushVerdict
from tests.conftest import AGENT, OTHER_SHA, REPO, SHA


def _attempt(*, sha: str = SHA, agent: str = AGENT, repo: str = REPO) -> PushAttempt:
    """Build one enrolment row.

    Args:
        sha: The commit sha.
        agent: The pushing session's label, or the empty string.
        repo: The repository.

    Returns:
        The row.
    """
    return PushAttempt(
        repo=repo, sha=sha, ref="refs/heads/main", agent=agent, attempted_unix=1788700000
    )


def _run(
    *,
    run_id: int = 34397357156,
    workflow: str = "Check",
    status: str = "completed",
    conclusion: str = "success",
) -> WorkflowRun:
    """Build one run.

    Args:
        run_id: The run's numeric id.
        workflow: The workflow's display name.
        status: The run's status.
        conclusion: Its conclusion.

    Returns:
        The run.
    """
    return WorkflowRun(
        run_id=run_id,
        workflow=workflow,
        status=status,
        conclusion=conclusion,
        html_url=f"https://github.com/{REPO}/actions/runs/{run_id}",
    )


def _tally(*, total: int = 52, listed: int | None = None, failed: tuple[str, ...] = ()) -> JobTally:
    """Build one job tally.

    Args:
        total: How many jobs the run had.
        listed: How many the page carried; defaults to ``total``.
        failed: The names of the jobs that did not succeed or skip.

    Returns:
        The tally.
    """
    return JobTally(total=total, listed=total if listed is None else listed, failed=failed)


def _report(
    *,
    attempt: PushAttempt | None = None,
    state: Literal["holding", "ripe", "abandoned", "stalled"] = "ripe",
    runs: tuple[tuple[WorkflowRun, JobTally], ...] = (),
) -> PushReport:
    """Build one decided push.

    Args:
        attempt: The enrolment row; a default one when omitted.
        state: The state this cycle put it in.
        runs: Each run paired with its job tally.

    Returns:
        The report.
    """
    row = _attempt() if attempt is None else attempt
    return PushReport(
        verdict=PushVerdict(attempt=row, runs=tuple(run for run, _ in runs), state=state),
        reports=tuple(RunReport(run=run, tally=tally) for run, tally in runs),
    )


def _only_body(reports: list[PushReport]) -> str:
    """Render one group and return its body.

    Args:
        reports: The pushes, all sharing one (repo, agent).

    Returns:
        The post text.
    """
    posts = announcements(reports)
    assert len(posts) == 1
    return posts[0]["body"]


class TestTheMarker:
    def test_every_body_leads_with_it(self) -> None:
        """``task_feed(query=...)`` is the one board surface that searches
        post bodies, and this is what makes these findable there without
        depending on any render grammar of the board's own."""
        body = _only_body([_report(runs=((_run(), _tally()),))])

        assert body.startswith(f"{MARKER} {REPO}:")

    def test_it_matches_the_other_bridges_shape(self) -> None:
        """``JOB-TERMINAL`` and ``DISPATCH-TERMINAL`` are the siblings, so
        one query finds every wake bridge's output."""
        assert MARKER.endswith("-TERMINAL")


class TestRunLines:
    def test_a_line_carries_workflow_conclusion_job_count_and_url(self) -> None:
        """All four, and each for its own reason -- see the module docstring
        of :mod:`ci_wake.announce`."""
        body = _only_body([_report(runs=((_run(), _tally(total=52)),))])

        assert "Check success -- 52 jobs" in body
        assert f"https://github.com/{REPO}/actions/runs/34397357156" in body

    def test_failed_jobs_are_named_not_just_counted(self) -> None:
        """A post that said "failure" without saying which jobs would send
        its reader to the run page to learn what the bridge already knew."""
        body = _only_body(
            [
                _report(
                    runs=(
                        (
                            _run(conclusion="failure"),
                            _tally(total=52, failed=("audit", "check (packages/db)")),
                        ),
                    )
                )
            ]
        )

        assert "2 failed: audit, check (packages/db)" in body

    def test_a_long_failed_list_is_capped_with_the_remainder_counted(self) -> None:
        """Capped, never truncated silently. A post naming eight of eleven
        broken jobs while implying eight is the total is worse than a post
        that says so."""
        failed = tuple(f"check (pkg{index})" for index in range(FAILED_CAP + 3))
        body = _only_body(
            [_report(runs=((_run(conclusion="failure"), _tally(total=40, failed=failed)),))]
        )

        assert f"{len(failed)} failed:" in body
        assert "+3 more" in body
        assert "check (pkg10)" not in body

    def test_a_partial_job_page_says_the_failed_list_is_partial(self) -> None:
        """``total_count`` and the returned array differ past one page, and a
        partial list presented as complete is the same lie as a silent cap."""
        body = _only_body(
            [
                _report(
                    runs=(
                        (_run(conclusion="failure"), _tally(total=137, listed=100, failed=("a",))),
                    )
                )
            ]
        )

        assert "137 jobs" in body
        assert "100 listed, so the failed list below is partial" in body

    def test_each_workflow_gets_its_own_line(self) -> None:
        """A repository can be green on one and red on another for one sha,
        and a post that collapsed them would be a verdict about neither."""
        body = _only_body(
            [
                _report(
                    runs=(
                        (_run(run_id=1, workflow="Check", conclusion="success"), _tally(total=52)),
                        (
                            _run(run_id=2, workflow="packages", conclusion="failure"),
                            _tally(total=41, failed=("check (services/Model-Trainer)",)),
                        ),
                    )
                )
            ]
        )

        assert "Check success -- 52 jobs" in body
        assert "packages failure -- 41 jobs -- 1 failed: check (services/Model-Trainer)" in body


class TestCancelledIsTwoDifferentEvents:
    def test_cancelled_with_no_jobs_is_named_as_an_eviction(self) -> None:
        """The concurrency group holds one running and one pending run, so a
        third arrival EVICTS the pending one. It never created a job. The
        wiki page measured three such runs inside fourteen hours in which
        nothing executed and nothing was reported."""
        body = _only_body([_report(runs=((_run(conclusion="cancelled"), _tally(total=0)),))])

        assert "cancelled (EVICTED FROM THE QUEUE, no job ever ran)" in body

    def test_cancelled_with_jobs_is_named_as_a_supersession(self) -> None:
        """A run superseded mid-flight ran something before it died, which is
        a different fact from an eviction."""
        body = _only_body([_report(runs=((_run(conclusion="cancelled"), _tally(total=2)),))])

        assert "cancelled (SUPERSEDED" in body

    def test_a_supersession_is_never_rendered_as_benign(self) -> None:
        """THE CORRECTION THIS TEST EXISTS TO PIN, and the wording it replaced
        came from a wiki page that was fact_checked the same day.

        That page frames supersession as harmless -- "the older answer is about
        stale code, so discarding it costs nothing" -- which assumes the newer
        run re-checks the same code. Under a matrix narrowed by
        ``event.before..sha`` it does not: the superseding push's diff window
        STARTS at the superseded push, so the superseded push's changes are in
        no run's window, ever. Measured in wagner-austin/API 2026-09-09, where
        a push creating a 28-file package got zero CI executions while the
        branch showed green.

        So the line must warn rather than reassure. Asserted as the ABSENCE of
        the old reassuring phrase as well as the presence of the new one,
        because a future edit that restored "mid-flight" alone would pass a
        presence-only test.
        """
        body = _only_body([_report(runs=((_run(conclusion="cancelled"), _tally(total=2)),))])

        assert "may be in no later run's diff window" in body
        assert "superseded mid-flight" not in body

    def test_the_supersession_claim_is_hedged_because_the_bridge_cannot_read_the_policy(
        self,
    ) -> None:
        """ "may", not "are". Whether the hole bites depends on the repository's
        ``cancel-in-progress`` setting -- wagner-austin/MCPs cancels nothing and
        is immune -- and the Actions runs API does not expose it. The stronger
        claim would be inventing a fact about a file this package never reads.
        """
        body = _only_body([_report(runs=((_run(conclusion="cancelled"), _tally(total=2)),))])

        assert "may be in no later run" in body
        assert "are in no later run" not in body

    def test_the_bare_word_never_appears_on_its_own(self) -> None:
        """The one place a conclusion alone is not merely thin but wrong."""
        for tally in (_tally(total=0), _tally(total=2)):
            body = _only_body([_report(runs=((_run(conclusion="cancelled"), tally),))])
            assert "Check cancelled --" not in body


class TestAbandoned:
    def test_it_says_no_run_ever_appeared_and_names_the_three_causes(self) -> None:
        """The third cause is the one that already cost this workspace
        hours: Actions starting no jobs while every local check stayed
        green."""
        body = _only_body([_report(state="abandoned")])

        assert "NO RUN EVER APPEARED" in body
        assert "push was refused" in body
        assert "path filters" in body
        assert "Actions is starting no jobs" in body

    def test_it_counts_in_the_header_rather_than_vanishing_from_it(self) -> None:
        """A header whose numbers did not add up to the lines beneath it is
        the first thing a reader distrusts."""
        body = _only_body([_report(state="abandoned")])

        assert "1 push(es), 0 run(s) (no-run x1)" in body


class TestStalled:
    def test_it_says_the_row_is_closed_and_shows_the_runs_to_watch(self) -> None:
        """The one honest trade in the design: the real verdict is never
        posted, so the post hands over the URLs instead."""
        body = _only_body(
            [
                _report(
                    state="stalled",
                    runs=((_run(status="in_progress", conclusion=""), _tally(total=52)),),
                )
            ]
        )

        assert "NO VERDICT AFTER 3h" in body
        assert "will not be announced again" in body
        assert "Check STILL in_progress -- 52 jobs" in body
        assert f"https://github.com/{REPO}/actions/runs/34397357156" in body

    def test_an_unfinished_run_is_tallied_by_status_not_by_conclusion(self) -> None:
        body = _only_body(
            [
                _report(
                    state="stalled",
                    runs=((_run(status="queued", conclusion=""), _tally(total=0)),),
                )
            ]
        )

        assert "still-queued x1" in body


class TestAddressing:
    def test_a_push_with_a_label_is_mentioned(self) -> None:
        body = _only_body([_report(runs=((_run(), _tally()),))])

        assert body.rstrip().endswith(f"@{AGENT} your push has its CI verdict")

    def test_a_push_without_one_says_so_rather_than_tagging_nobody(self) -> None:
        """A human pushing from a terminal has no board label. Their verdict
        is posted board-level and the post says how to be addressed next
        time."""
        body = _only_body([_report(attempt=_attempt(agent=""), runs=((_run(), _tally()),))])

        assert "exported no BOARD_AGENT_LABEL" in body
        assert "@" not in body.splitlines()[-1]


class TestGrouping:
    def test_one_post_per_repository_and_pusher(self) -> None:
        """Not one per push and not one per run. Three notes for one intent
        buries the feed the moment the bridge works."""
        posts = announcements(
            [
                _report(attempt=_attempt(sha=SHA), runs=((_run(run_id=1), _tally()),)),
                _report(attempt=_attempt(sha=OTHER_SHA), runs=((_run(run_id=2), _tally()),)),
            ]
        )

        assert len(posts) == 1
        assert "2 push(es), 2 run(s)" in posts[0]["body"]

    def test_two_pushers_in_one_repository_get_one_post_each(self) -> None:
        posts = announcements(
            [
                _report(attempt=_attempt(agent="opus-one-0909"), runs=((_run(), _tally()),)),
                _report(
                    attempt=_attempt(sha=OTHER_SHA, agent="opus-two-0909"),
                    runs=((_run(), _tally()),),
                ),
            ]
        )

        assert [post["agent"] for post in posts] == ["opus-one-0909", "opus-two-0909"]

    def test_two_repositories_get_one_post_each(self) -> None:
        posts = announcements(
            [
                _report(attempt=_attempt(repo="wagner-austin/API"), runs=((_run(), _tally()),)),
                _report(attempt=_attempt(repo=REPO), runs=((_run(), _tally()),)),
            ]
        )

        assert [post["repo"] for post in posts] == ["wagner-austin/API", REPO]

    def test_the_order_is_deterministic(self) -> None:
        """Two cycles over the same input post the same thing in the same
        order, so a reader comparing two runs is comparing the work rather
        than a dictionary's iteration."""
        reports = [
            _report(attempt=_attempt(repo="wagner-austin/zzz"), runs=((_run(), _tally()),)),
            _report(attempt=_attempt(repo="wagner-austin/aaa"), runs=((_run(), _tally()),)),
        ]

        assert [post["repo"] for post in announcements(reports)] == [
            post["repo"] for post in announcements(list(reversed(reports)))
        ]

    def test_no_pushes_produce_no_posts(self) -> None:
        assert announcements([]) == []


class TestTheHeader:
    def test_it_tallies_every_outcome_in_the_group(self) -> None:
        body = _only_body(
            [
                _report(attempt=_attempt(sha=SHA), runs=((_run(conclusion="success"), _tally()),)),
                _report(
                    attempt=_attempt(sha=OTHER_SHA),
                    runs=((_run(conclusion="failure"), _tally(failed=("audit",))),),
                ),
            ]
        )

        assert body.startswith(f"{MARKER} {REPO}: 2 push(es), 2 run(s) (failure x1, success x1)")

    def test_beyond_the_line_cap_the_remainder_is_counted_not_dropped(self) -> None:
        reports = [
            _report(attempt=_attempt(sha=f"{index:040x}"), runs=((_run(), _tally()),))
            for index in range(LINE_CAP + 2)
        ]

        body = _only_body(reports)

        assert f"{LINE_CAP + 2} push(es)" in body
        assert "+2 more, all in the enrolment record" in body
