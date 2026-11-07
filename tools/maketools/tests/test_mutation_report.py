"""Tests for :mod:`maketools.mutation.report` (MCPs board task e52e5148).

The lines are MCPs' steps' (``mcp_shared_py.mutation.report`` and
``mcp-shared/src/mutation/report.ts``), so these are MCPs' own cases for
its Python step (``mcp-shared-py/tests/test_mutation_report.py``, board
task fec5851c) with the import path made this package's: the same
expectations, written out whole, hold the same lines in both repositories.
"""

from __future__ import annotations

from maketools.mutation.report import (
    Tally,
    Uncaught,
    UncaughtMutant,
    mutant_line,
    render_verdict,
)

TALLY = Tally(files=2, base="abc123", by_tests=5, by_suite=2, by_none=1, declarations=4)

RAN = (
    "mutation: 8 mutant(s): 5 run by the tests covering their line, 2 by the whole suite "
    "(lines run only at import), 1 by no test; 4 more in annotations and parameter markers "
    "are not code"
)


def _mutant(path: str, line: int, column: int, why: Uncaught) -> UncaughtMutant:
    return UncaughtMutant(path, line, column, "core/ReplaceComparisonOperator_Eq_Lt", "x < y", why)


def test_a_mutant_line_names_its_file_line_one_based_column_operator_and_change() -> None:
    mutant = UncaughtMutant(
        "src/p/a.py", 12, 4, "core/NumberReplacer", 'print("a\\tb")', Uncaught.SURVIVED
    )
    assert mutant_line(mutant) == (
        'MUTANT SURVIVED src/p/a.py:12:5 NumberReplacer -> "print(\\"a\\\\tb\\")"'
    )


def test_a_mutant_no_test_ran_says_so() -> None:
    assert mutant_line(_mutant("src/p/a.py", 3, 0, Uncaught.NO_TEST)) == (
        'MUTANT SURVIVED, NO TEST RAN IT src/p/a.py:3:1 ReplaceComparisonOperator_Eq_Lt -> "x < y"'
    )


def test_an_operator_outside_cosmic_rays_core_keeps_its_whole_name() -> None:
    mutant = UncaughtMutant("src/a.py", 1, 0, "extra/Swap", "y", Uncaught.SURVIVED)
    assert mutant_line(mutant) == 'MUTANT SURVIVED src/a.py:1:1 extra/Swap -> "y"'


def test_a_run_with_no_survivor_passes_naming_how_many_were_made() -> None:
    assert render_verdict(TALLY, []) == [
        RAN,
        "mutation: passed: 0 of 8 mutant(s) made in 2 src file(s) changed against origin/main "
        "(merge base abc123) survived",
    ]


def test_survivors_fail_the_run_in_file_line_and_column_order() -> None:
    later_column = _mutant("src/a.py", 7, 9, Uncaught.SURVIVED)
    earlier_column = _mutant("src/a.py", 7, 2, Uncaught.SURVIVED)
    earlier_line = _mutant("src/a.py", 3, 40, Uncaught.NO_TEST)
    other_file = _mutant("src/b.py", 1, 0, Uncaught.SURVIVED)
    lines = render_verdict(TALLY, [other_file, later_column, earlier_line, earlier_column])
    assert lines == [
        RAN,
        mutant_line(earlier_line),
        mutant_line(earlier_column),
        mutant_line(later_column),
        mutant_line(other_file),
        "mutation: FAILED: 4 of 8 mutant(s) made in 2 src file(s) changed against origin/main "
        "(merge base abc123) survived. Kill each with a test, or rewrite the code so the "
        "mutation cannot change what it does.",
    ]
