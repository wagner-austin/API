"""What a post says, and the three things a bare conclusion would not.

THESE ARE THE TESTS THAT PIN THE WIKI PAGE. ``reading-ci-run-outcomes``
(mcps-codebase) measures how a run list misleads a reader; the assertions
below are that argument turned into a rendering contract, so a later
simplification of the line format fails a test rather than quietly making
every announcement less true than it was.
"""

from __future__ import annotations

from ci_wake.announce import FAILED_CAP, LINE_CAP, MARKER, announcements
from tests.conftest import (
    AGENT,
    OTHER_SHA,
    REPO,
    SHA,
    make_attempt,
    make_report,
    make_run,
    make_tally,
    only_body,
)


class TestTheMarker:
    def test_every_body_leads_with_it(self) -> None:
        """``task_feed(query=...)`` is the one board surface that searches
        post bodies, and this is what makes these findable there without
        depending on any render grammar of the board's own."""
        body = only_body([make_report(runs=((make_run(), make_tally()),))])

        assert body.startswith(f"{MARKER} {REPO}:")

    def test_it_matches_the_other_bridges_shape(self) -> None:
        """``JOB-TERMINAL`` and ``DISPATCH-TERMINAL`` are the siblings, so
        one query finds every wake bridge's output."""
        assert MARKER.endswith("-TERMINAL")


class TestRunLines:
    def test_a_line_carries_workflow_conclusion_job_count_and_url(self) -> None:
        """All four, and each for its own reason -- see the module docstring
        of :mod:`ci_wake.announce`."""
        body = only_body([make_report(runs=((make_run(), make_tally(total=52)),))])

        assert "Check success -- 52 jobs" in body
        assert f"https://github.com/{REPO}/actions/runs/34397357156" in body

    def test_failed_jobs_are_named_not_just_counted(self) -> None:
        """A post that said "failure" without saying which jobs would send
        its reader to the run page to learn what the bridge already knew."""
        body = only_body(
            [
                make_report(
                    runs=(
                        (
                            make_run(conclusion="failure"),
                            make_tally(total=52, failed=("audit", "check (packages/db)")),
                        ),
                    )
                )
            ]
        )

        assert (
            "2 failed: audit (step: Run cd audit && npm run check), "
            "check (packages/db) (step: Run cd check (packages/db) && npm run check)"
        ) in body

    def test_a_job_that_never_reached_a_failing_step_is_not_counted_as_failed(
        self,
    ) -> None:
        """THE FOURTH INSTANCE OF THIS BRIDGE'S RECURRING DEFECT, IN THE SHAPE
        THAT CAUSED IT.

        ``wagner-austin/MCPs`` run 34459514888, 2026-09-10: this bridge posted
        "5 failed" naming five packages. Only ``github-mcp`` had a step that
        concluded failure -- three died inside setup with their check step
        left PENDING and a fourth was killed mid-run, none of them executing a
        test, all four green locally. Four sessions read it as five broken
        packages, and two of them spent time investigating packages that were
        fine.

        So "N failed" must mean N packages whose own check failed, and the
        rest get a line that says what is actually known about them.
        """
        body = only_body(
            [
                make_report(
                    runs=(
                        (
                            make_run(conclusion="failure"),
                            make_tally(
                                total=56,
                                failed=("github-mcp",),
                                stopped=(
                                    "constituent-crm",
                                    "meetings",
                                    "netfile-mcp",
                                    "packages/wiki-search",
                                ),
                            ),
                        ),
                    )
                )
            ]
        )

        assert "1 failed: github-mcp (step: Run cd github-mcp && npm run check)" in body
        assert (
            "4 stopped without a failing step: constituent-crm, meetings, "
            "netfile-mcp, packages/wiki-search"
        ) in body
        # The old rendering's exact claim, asserted absent: it is what four
        # sessions read as five broken packages.
        assert "5 failed" not in body

    def test_the_stopped_line_states_the_fact_and_not_a_cause(self) -> None:
        """ "stopped without a failing step" is in the payload. WHY it stopped
        -- setup death, mid-run kill, a lost runner -- is not, and this bridge
        has been corrected three times for narrating causes the API does not
        expose."""
        body = only_body(
            [
                make_report(
                    runs=(
                        (
                            make_run(conclusion="failure"),
                            make_tally(total=2, stopped=("meetings",)),
                        ),
                    )
                )
            ]
        )

        assert "1 stopped without a failing step: meetings" in body
        for invented in ("setup", "died", "killed", "runner"):
            assert invented not in body

    def test_a_long_failed_list_is_capped_with_the_remainder_counted(self) -> None:
        """Capped, never truncated silently. A post naming eight of eleven
        broken jobs while implying eight is the total is worse than a post
        that says so."""
        failed = tuple(f"check (pkg{index})" for index in range(FAILED_CAP + 3))
        body = only_body(
            [
                make_report(
                    runs=((make_run(conclusion="failure"), make_tally(total=40, failed=failed)),)
                )
            ]
        )

        assert f"{len(failed)} failed:" in body
        assert "+3 more" in body
        assert "check (pkg10)" not in body

    def test_a_partial_job_page_says_the_failed_list_is_partial(self) -> None:
        """``total_count`` and the returned array differ past one page, and a
        partial list presented as complete is the same lie as a silent cap."""
        body = only_body(
            [
                make_report(
                    runs=(
                        (
                            make_run(conclusion="failure"),
                            make_tally(total=137, listed=100, failed=("a",)),
                        ),
                    )
                )
            ]
        )

        assert "137 jobs" in body
        assert "100 listed, so the failed list below is partial" in body

    def test_each_workflow_gets_its_own_line(self) -> None:
        """A repository can be green on one and red on another for one sha,
        and a post that collapsed them would be a verdict about neither."""
        body = only_body(
            [
                make_report(
                    runs=(
                        (
                            make_run(run_id=1, workflow="Check", conclusion="success"),
                            make_tally(total=52),
                        ),
                        (
                            make_run(run_id=2, workflow="packages", conclusion="failure"),
                            make_tally(total=41, failed=("check (services/Model-Trainer)",)),
                        ),
                    )
                )
            ]
        )

        assert "Check success -- 52 jobs" in body
        assert "packages failure -- 41 jobs -- 1 failed: check (services/Model-Trainer)" in body


class TestCancelledIsTwoDifferentEvents:
    def test_cancelled_with_no_jobs_says_nothing_ran(self) -> None:
        """A run that created no job at all is a different situation from one
        where most jobs finished, and the count is what separates them. The
        wiki page measured three such runs inside fourteen hours in which
        nothing executed and nothing was reported."""
        body = only_body(
            [make_report(runs=((make_run(conclusion="cancelled"), make_tally(total=0)),))]
        )

        assert "cancelled -- no job ever ran" in body

    def test_cancelled_with_jobs_counts_the_survivors(self) -> None:
        """How much ran is the part a reader can act on."""
        body = only_body(
            [
                make_report(
                    runs=(
                        (
                            make_run(conclusion="cancelled"),
                            make_tally(total=2, cancelled=("check (db)",)),
                        ),
                    )
                )
            ]
        )

        assert "cancelled -- 1 of 2 jobs completed" in body

    def test_no_cause_is_ever_asserted_for_a_cancellation(self) -> None:
        """THE THIRD CORRECTION TO THIS LINE, AND THE ONE THAT FINALLY STOPS
        IT INVENTING THINGS.

        The rendering said "SUPERSEDED" and "EVICTED FROM THE QUEUE". The
        Actions API exposes neither: a concurrency supersession, a manual
        ``gh run cancel`` and a force-close after a runner disappears all
        produce the identical conclusion. Caught 2026-09-10 by
        ``fable-brain-audit-0903``, who WAS the ground truth for their own
        case -- this bridge announced their ``runner-diag`` run as SUPERSEDED
        when they had cancelled it by hand, and no later run had entered its
        concurrency group at all.

        Asserted as an absence across BOTH cancelled arms, because the two
        invented causes lived in different branches and a test covering one
        would not have caught the other.
        """
        for tally in (make_tally(total=0), make_tally(total=2, cancelled=("check (db)",))):
            body = only_body([make_report(runs=((make_run(conclusion="cancelled"), tally),))])

            assert "SUPERSEDED" not in body
            assert "EVICTED" not in body

    def test_a_supersession_is_never_rendered_as_benign(self) -> None:
        """The wording it replaced came from a wiki page that frames
        supersession as harmless -- "the older answer is about stale code, so
        discarding it costs nothing" -- which assumes the newer run re-checks
        the same code. Under a matrix narrowed by ``event.before..sha`` a
        STOPPED job's changes can fall between windows instead.

        Asserted as the ABSENCE of the old reassuring phrase as well as the
        presence of the new one, because a future edit that restored
        "mid-flight" alone would pass a presence-only test.
        """
        body = only_body(
            [
                make_report(
                    runs=(
                        (
                            make_run(conclusion="cancelled"),
                            make_tally(total=2, cancelled=("check (db)",)),
                        ),
                    )
                )
            ]
        )

        assert "may be in no later run's diff window" in body
        assert "superseded mid-flight" not in body

    def test_the_caveat_attaches_to_the_cancelled_jobs_not_to_themake_run(self) -> None:
        """THE CORRECTION THIS TEST EXISTS TO PIN, and it is a correction to
        this package's own first fix rather than to someone else's code.

        That fix said the RUN's changes may be in no later window -- which
        repeated, one level up, the exact defect it was correcting. A run
        reading ``cancelled`` says only that SOMETHING in it stopped: run
        34418498808 in wagner-austin/API read ``cancelled`` while exactly 3 of
        its 42 jobs were cancelled and the rest ran to completion. Attaching
        the caveat to the run asserted about 42 jobs what was true of 3.

        The earlier justification also cited a measurement that was retracted
        the same night -- a "28-file package with zero CI executions" that had
        in fact been checked and was green, the empty result being an artifact
        of ``gh run list --commit`` silently requiring a full sha.

        So: the surviving jobs are counted, and the caveat names the stopped
        ones.
        """
        body = only_body(
            [
                make_report(
                    runs=(
                        (
                            make_run(conclusion="cancelled"),
                            make_tally(
                                total=42,
                                cancelled=("check (handwriting-ai)", "check (Model-Trainer)"),
                            ),
                        ),
                    )
                )
            ]
        )

        assert "40 of 42 jobs completed" in body
        assert "2 cancelled, whose packages may be in no later run's diff window" in body
        assert "check (handwriting-ai), check (Model-Trainer)" in body

    def test_a_cancelled_job_is_not_reported_as_a_failed_one(self) -> None:
        """They are different events and lumping them was the same aggregate
        mistake. A cancelled job did not fail; it was stopped."""
        body = only_body(
            [
                make_report(
                    runs=(
                        (
                            make_run(conclusion="cancelled"),
                            make_tally(total=3, failed=("audit",), cancelled=("check (db)",)),
                        ),
                    )
                )
            ]
        )

        assert "1 failed: audit" in body
        assert "1 cancelled, whose packages" in body

    def test_the_supersession_claim_is_hedged_because_the_bridge_cannot_read_the_policy(
        self,
    ) -> None:
        """ "may", not "are". Whether the hole bites depends on the repository's
        ``cancel-in-progress`` setting -- wagner-austin/MCPs cancels nothing and
        is immune -- and the Actions runs API does not expose it. The stronger
        claim would be inventing a fact about a file this package never reads.
        """
        body = only_body(
            [
                make_report(
                    runs=(
                        (
                            make_run(conclusion="cancelled"),
                            make_tally(total=2, cancelled=("check (db)",)),
                        ),
                    )
                )
            ]
        )

        assert "may be in no later run" in body
        assert "are in no later run" not in body

    def test_the_bare_word_never_appears_on_its_own(self) -> None:
        """The one place a conclusion alone is not merely thin but wrong.

        THIS ASSERTION WAS REWRITTEN BECAUSE ITS PREDICATE STOPPED MEANING
        WHAT IT SAID. It used to read ``"Check cancelled --" not in body``,
        which caught the bare word only while the qualifier was parenthesised
        (``cancelled (SUPERSEDED ...)``). When the cause words were removed
        and ``--`` became the separator, that string began appearing in
        CORRECT output, so the guard would have failed on every good render
        and passed on none -- a predicate that outlived the format it was
        written against.

        The intent is that the word always carries a qualifier, so that is
        what is asserted now: whichever arm produces the line, it says how
        much ran.
        """
        for tally in (make_tally(total=0), make_tally(total=2, cancelled=("check (db)",))):
            body = only_body([make_report(runs=((make_run(conclusion="cancelled"), tally),))])

            qualifiers = ("no job ever ran", "of 2 jobs completed")
            assert any(qualifier in body for qualifier in qualifiers)


class TestAbandoned:
    def test_it_says_no_run_ever_appeared_and_names_the_three_causes(self) -> None:
        """The third cause is the one that already cost this workspace
        hours: Actions starting no jobs while every local check stayed
        green."""
        body = only_body([make_report(state="abandoned")])

        assert "NO RUN EVER APPEARED" in body
        assert "push was refused" in body
        assert "path filters" in body
        assert "Actions is starting no jobs" in body

    def test_it_counts_in_the_header_rather_than_vanishing_from_it(self) -> None:
        """A header whose numbers did not add up to the lines beneath it is
        the first thing a reader distrusts."""
        body = only_body([make_report(state="abandoned")])

        assert "1 push(es), 0 run(s) (no-run x1)" in body


class TestStalled:
    def test_it_says_the_row_is_closed_and_shows_the_runs_to_watch(self) -> None:
        """The one honest trade in the design: the real verdict is never
        posted, so the post hands over the URLs instead."""
        body = only_body(
            [
                make_report(
                    state="stalled",
                    runs=((make_run(status="in_progress", conclusion=""), make_tally(total=52)),),
                )
            ]
        )

        assert "NO VERDICT AFTER 3h" in body
        assert "will not be announced again" in body
        assert "Check STILL in_progress -- 52 jobs" in body
        assert f"https://github.com/{REPO}/actions/runs/34397357156" in body

    def test_an_unfinished_run_is_tallied_by_status_not_by_conclusion(self) -> None:
        body = only_body(
            [
                make_report(
                    state="stalled",
                    runs=((make_run(status="queued", conclusion=""), make_tally(total=0)),),
                )
            ]
        )

        assert "still-queued x1" in body


class TestAddressing:
    def test_a_push_with_a_label_is_mentioned(self) -> None:
        body = only_body([make_report(runs=((make_run(), make_tally()),))])

        assert body.rstrip().endswith(f"@{AGENT} your push has its CI verdict")

    def test_a_push_without_one_says_so_rather_than_tagging_nobody(self) -> None:
        """A human pushing from a terminal has no board label. Their verdict
        is posted board-level and the post says how to be addressed next
        time."""
        body = only_body(
            [make_report(attempt=make_attempt(agent=""), runs=((make_run(), make_tally()),))]
        )

        assert "exported no BOARD_AGENT_LABEL" in body
        assert "@" not in body.splitlines()[-1]


class TestGrouping:
    def test_one_post_per_repository_and_pusher(self) -> None:
        """Not one per push and not one per run. Three notes for one intent
        buries the feed the moment the bridge works."""
        posts = announcements(
            [
                make_report(
                    attempt=make_attempt(sha=SHA), runs=((make_run(run_id=1), make_tally()),)
                ),
                make_report(
                    attempt=make_attempt(sha=OTHER_SHA), runs=((make_run(run_id=2), make_tally()),)
                ),
            ]
        )

        assert len(posts) == 1
        assert "2 push(es), 2 run(s)" in posts[0]["body"]

    def test_two_pushers_in_one_repository_get_one_post_each(self) -> None:
        posts = announcements(
            [
                make_report(
                    attempt=make_attempt(agent="opus-one-0909"), runs=((make_run(), make_tally()),)
                ),
                make_report(
                    attempt=make_attempt(sha=OTHER_SHA, agent="opus-two-0909"),
                    runs=((make_run(), make_tally()),),
                ),
            ]
        )

        assert [post["agent"] for post in posts] == ["opus-one-0909", "opus-two-0909"]

    def test_two_repositories_get_one_post_each(self) -> None:
        posts = announcements(
            [
                make_report(
                    attempt=make_attempt(repo="wagner-austin/API"),
                    runs=((make_run(), make_tally()),),
                ),
                make_report(attempt=make_attempt(repo=REPO), runs=((make_run(), make_tally()),)),
            ]
        )

        assert [post["repo"] for post in posts] == ["wagner-austin/API", REPO]

    def test_the_order_is_deterministic(self) -> None:
        """Two cycles over the same input post the same thing in the same
        order, so a reader comparing two runs is comparing the work rather
        than a dictionary's iteration."""
        reports = [
            make_report(
                attempt=make_attempt(repo="wagner-austin/zzz"), runs=((make_run(), make_tally()),)
            ),
            make_report(
                attempt=make_attempt(repo="wagner-austin/aaa"), runs=((make_run(), make_tally()),)
            ),
        ]

        assert [post["repo"] for post in announcements(reports)] == [
            post["repo"] for post in announcements(list(reversed(reports)))
        ]

    def test_no_pushes_produce_no_posts(self) -> None:
        assert announcements([]) == []


class TestTheHeader:
    def test_it_tallies_every_outcome_in_the_group(self) -> None:
        body = only_body(
            [
                make_report(
                    attempt=make_attempt(sha=SHA),
                    runs=((make_run(conclusion="success"), make_tally()),),
                ),
                make_report(
                    attempt=make_attempt(sha=OTHER_SHA),
                    runs=((make_run(conclusion="failure"), make_tally(failed=("audit",))),),
                ),
            ]
        )

        assert body.startswith(f"{MARKER} {REPO}: 2 push(es), 2 run(s) (failure x1, success x1)")

    def test_beyond_the_line_cap_the_remainder_is_counted_not_dropped(self) -> None:
        reports = [
            make_report(
                attempt=make_attempt(sha=f"{index:040x}"), runs=((make_run(), make_tally()),)
            )
            for index in range(LINE_CAP + 2)
        ]

        body = only_body(reports)

        assert f"{LINE_CAP + 2} push(es)" in body
        assert "+2 more, all in the enrolment record" in body
