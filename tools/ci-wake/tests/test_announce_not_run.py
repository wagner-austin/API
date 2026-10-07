"""A cancelled run is announced as NOT RUN, with its sha, to the session that pushed it.

MCPs board task c04519f9, A3. ``wagner-austin/MCPs`` gives every push to main
one shared pending slot, so each push of a burst but the newest gets a run
cancelled before any job starts. Its post used to say "cancelled -- no job
ever ran" under a seven-character sha and close with "your push has its CI
verdict", which is how 49b8cb6a1's never-run tests read as checked. These
cases pin the words, the full sha and the mention that replaced that.
"""

from __future__ import annotations

from ci_wake.announce import NOT_RUN, announcements
from tests.conftest import (
    AGENT,
    OTHER_SHA,
    SHA,
    make_attempt,
    make_report,
    make_run,
    make_tally,
    only_body,
)


class TestAnEvictedRun:
    def test_it_is_marked_not_run_and_names_the_full_sha(self) -> None:
        body = only_body(
            [make_report(runs=((make_run(conclusion="cancelled"), make_tally(total=0)),))]
        )

        assert NOT_RUN == "NOT RUN"
        assert "Check cancelled -- no job ever ran, NOT RUN -- 0 jobs" in body
        assert (
            f"  NOT RUN: CI cancelled work for {SHA}, so this sha has no complete CI "
            "verdict and a cancelled run is not a pass; check it with make check-fleet."
        ) in body

    def test_the_pusher_is_told_it_was_not_run_and_never_that_it_has_a_verdict(self) -> None:
        body = only_body(
            [make_report(runs=((make_run(conclusion="cancelled"), make_tally(total=0)),))]
        )

        assert body.splitlines()[-1] == (
            f"@{AGENT} 1 of your pushed sha(s) were not run by CI in full (NOT RUN), "
            "so they have no CI verdict"
        )
        assert "has its CI verdict" not in body


class TestAPartlyCancelledRun:
    def test_it_counts_the_jobs_not_run_and_still_names_the_sha(self) -> None:
        body = only_body(
            [
                make_report(
                    runs=(
                        (
                            make_run(conclusion="cancelled"),
                            make_tally(total=3, cancelled=("check (db)", "check (tenant)")),
                        ),
                    )
                )
            ]
        )

        assert "Check cancelled -- 1 of 3 jobs completed, 2 NOT RUN -- 3 jobs" in body
        assert f"NOT RUN: CI cancelled work for {SHA}" in body


class TestAGroupWithBoth:
    def test_only_the_cancelled_push_gets_the_line_and_the_mention_counts_it(self) -> None:
        cancelled = make_report(
            attempt=make_attempt(sha=SHA),
            runs=((make_run(conclusion="cancelled"), make_tally(total=0)),),
        )
        green = make_report(
            attempt=make_attempt(sha=OTHER_SHA),
            runs=((make_run(conclusion="success"), make_tally()),),
        )

        body = only_body([cancelled, green])

        assert f"NOT RUN: CI cancelled work for {SHA}" in body
        assert f"NOT RUN: CI cancelled work for {OTHER_SHA}" not in body
        assert body.splitlines()[-1].startswith(f"@{AGENT} 1 of your pushed sha(s) were not run")

    def test_a_group_with_nothing_cancelled_keeps_its_verdict_line(self) -> None:
        body = only_body([make_report(runs=((make_run(conclusion="failure"), make_tally()),))])

        assert NOT_RUN not in body
        assert body.splitlines()[-1] == f"@{AGENT} your push has its CI verdict"

    def test_an_unaddressed_cancelled_push_still_carries_the_words_and_the_sha(self) -> None:
        posts = announcements(
            [
                make_report(
                    attempt=make_attempt(agent=""),
                    runs=((make_run(conclusion="cancelled"), make_tally(total=0)),),
                )
            ]
        )

        assert f"NOT RUN: CI cancelled work for {SHA}" in posts[0]["body"]
        assert "exported no BOARD_AGENT_LABEL" in posts[0]["body"]
