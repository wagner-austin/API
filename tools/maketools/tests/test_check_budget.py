"""The lifted five-minute rule, as this package carries it (MCPs board task 1b152218).

``src/maketools/check_budget.py`` is MCPs' file byte for byte
(:mod:`maketools.lift`), and its home suite covers it there. This package
holds itself to 100 percent of what it ships, so the copy is exercised here
too: the split a locked check charges, the boundary of the budget, the
verdict lines and the exit code. Only ``check-budget``'s whole-check path
runs in this repository; the locked path is the home's.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from maketools.check_budget import (
    BUDGET_SECONDS,
    OVER_BUDGET_EXIT_CODE,
    WHOLE_CHECK_SPLIT,
    CheckSpend,
    budgeted_exit_code,
    exceeds_budget,
    spend,
    verdict_lines,
)

NOW = datetime(2026, 10, 4, 10, 0, 0, tzinfo=UTC)


def test_a_locked_check_is_charged_everything_but_its_queue() -> None:
    spent = spend(
        started=NOW,
        asked=NOW + timedelta(seconds=40),
        acquired=NOW + timedelta(seconds=940),
        finished=NOW + timedelta(seconds=1180),
    )
    assert spent == CheckSpend(
        before_lock=timedelta(seconds=40),
        queued=timedelta(seconds=900),
        suite=timedelta(seconds=240),
    )
    assert spent.charged == timedelta(seconds=280)
    assert not spent.over_budget
    assert spent.split == "lint and setup 40s, suite 240s, queued 900s (not counted)"


def test_exactly_the_budget_is_within_it_and_one_second_more_is_not() -> None:
    assert BUDGET_SECONDS == 300
    assert not exceeds_budget(timedelta(seconds=300))
    assert exceeds_budget(timedelta(seconds=301))


def test_the_verdict_is_one_line_within_the_budget_and_a_loud_block_past_it() -> None:
    assert verdict_lines("libs/procart", timedelta(seconds=90), WHOLE_CHECK_SPLIT) == [
        f"CHECK BUDGET: libs/procart took 90s of 300s ({WHOLE_CHECK_SPLIT})."
    ]
    assert verdict_lines("libs/covenant_ml", timedelta(seconds=744), WHOLE_CHECK_SPLIT) == [
        "",
        "=" * 72,
        "CHECK OVER BUDGET: libs/covenant_ml took 744s, over the 300s budget by 444s.",
        f"  split : {WHOLE_CHECK_SPLIT}",
        "  why   : the operator, 2026-10-04: make check should be about 5 minutes for "
        "every repo (board task 4080d695).",
        "  fix   : make the suite lighter; nothing raises the budget.",
        "=" * 72,
    ]


def test_a_failing_check_keeps_its_code_and_a_slow_passing_one_exits_three() -> None:
    assert budgeted_exit_code(0, timedelta(seconds=10)) == 0
    assert budgeted_exit_code(0, timedelta(seconds=301)) == OVER_BUDGET_EXIT_CODE == 3
    assert budgeted_exit_code(2, timedelta(seconds=301)) == 2
