"""The mutation step's span, held to its own bound beside the budget (MCPs board task e52e5148).

The span file's lines are MCPs' (``<whole seconds> <epoch seconds it
ended> <name>``); a line of any other shape, or naming a span with no
bound, is refused by code rather than read as nothing; overlapping spans
are taken out of the charge once; and a passing check fails exactly when a
span is past its bound.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from platform_core.error_codes_tooling import MutationErrorCode
from platform_core.errors import AppError

from maketools.bounded_spans import (
    MUTATION_BOUND_SECONDS,
    MUTATION_SPAN,
    BoundedSpan,
    bounded_exit_code,
    covered,
    parse_span,
    read_spans,
    record_span,
    span_line,
    span_lines,
    spans_split,
)
from maketools.check_budget import OVER_BUDGET_EXIT_CODE


def _span(seconds: int, ended: float = 1000.0) -> BoundedSpan:
    """A mutation span.

    Args:
        seconds: Its length.
        ended: When it ended.

    Returns:
        The span with the mutation bound.
    """
    return BoundedSpan(MUTATION_SPAN, seconds, MUTATION_BOUND_SECONDS, ended)


def test_the_bound_is_mcps_nine_hundred_seconds() -> None:
    assert MUTATION_BOUND_SECONDS == 900
    assert MUTATION_SPAN == "mutation testing"


def test_a_recorded_span_reads_back_as_written(tmp_path: Path) -> None:
    path = tmp_path / "spans"
    record_span(path, MUTATION_SPAN, 41, 1794087804.5)
    record_span(path, MUTATION_SPAN, 7, 1794087900.25)
    assert path.read_text(encoding="utf-8") == (
        "41 1794087804.500 mutation testing\n7 1794087900.250 mutation testing\n"
    )
    assert read_spans(path) == (_span(41, 1794087804.5), _span(7, 1794087900.25))


def test_a_check_no_span_was_recorded_in_reads_no_spans(tmp_path: Path) -> None:
    assert read_spans(tmp_path / "absent") == ()


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "spans"
    path.write_text("\n" + span_line(MUTATION_SPAN, 3, 10.0) + "  \n", encoding="utf-8")
    assert read_spans(path) == (_span(3, 10.0),)


@pytest.mark.parametrize(
    "line",
    [
        "3 10.000 deploy rehearsal",
        "3.5 10.000 mutation testing",
        "-3 10.000 mutation testing",
        "3 ten mutation testing",
        "3 10.000",
        "x3 10.000 mutation testing",
    ],
)
def test_a_line_of_another_shape_or_an_unbounded_name_is_refused(line: str) -> None:
    with pytest.raises(AppError, match=r"^span line .* is not '<whole seconds>") as refused:
        parse_span(line)
    assert refused.value.code is MutationErrorCode.SPAN_UNREADABLE


def test_an_integer_end_is_read() -> None:
    assert parse_span("5 12 mutation testing") == _span(5, 12.0)


def test_overlapping_spans_are_covered_once_and_apart_ones_add() -> None:
    assert covered([]) == timedelta(0)
    assert covered([_span(10, 100.0), _span(10, 105.0)]) == timedelta(seconds=15)
    assert covered([_span(10, 100.0), _span(4, 98.0)]) == timedelta(seconds=10)
    assert covered([_span(10, 100.0), _span(10, 200.0)]) == timedelta(seconds=20)


def test_the_split_names_each_span_and_its_bound() -> None:
    assert spans_split("whole", []) == "whole"
    assert spans_split("whole", [_span(12), _span(3)]) == (
        "whole; mutation testing 12s of its own 900s bound (not counted); "
        "mutation testing 3s of its own 900s bound (not counted)"
    )


def test_only_a_span_past_its_bound_prints_the_loud_block() -> None:
    assert span_lines("tools/fleet", [_span(900)]) == []
    assert span_lines("tools/fleet", [_span(900), _span(905)]) == [
        "",
        "=" * 72,
        "CHECK OVER BOUND: tools/fleet's mutation testing took 905s, over its 900s bound by 5s.",
        "  why   : a span outside the five-minute budget is held to its own bound "
        "(tools/maketools bounded_spans.py, MCPs board task e52e5148).",
        "  fix   : make the run lighter or the change smaller; nothing raises the bound.",
        "=" * 72,
    ]


def test_a_passing_check_fails_on_a_span_past_its_bound_and_a_failing_one_keeps_its_code() -> None:
    within = timedelta(seconds=100)
    assert bounded_exit_code(0, within, [_span(900)]) == 0
    assert bounded_exit_code(0, within, [_span(901)]) == OVER_BUDGET_EXIT_CODE
    assert bounded_exit_code(2, within, [_span(901)]) == 2
    assert bounded_exit_code(0, timedelta(seconds=301), []) == OVER_BUDGET_EXIT_CODE
