"""Turning idle observations into a credit rate.

The measurement this supports is blunt on purpose -- stand still, watch credits,
compare windows -- so what needs testing is the arithmetic and, more
importantly, the refusals. A window that cannot be trusted has to be dropped
rather than averaged, because a wrong income figure is worse than no income
figure: the reserve policy would be tuned against it.
"""

from __future__ import annotations

from rw_bot.mechanics.income import (
    Reading,
    format_rates,
    marginal,
    measure,
    payback_seconds,
)


def _reading(window: int, extractors: int, clock_ms: int, credits: int) -> Reading:
    return Reading(
        window=window,
        extractors=extractors,
        clock_ms=clock_ms,
        credits=credits,
    )


def test_a_window_measures_credits_per_second() -> None:
    """Two hundred credits over four seconds is fifty a second."""
    rates = measure(
        (
            _reading(0, 0, 1_000, 4_000),
            _reading(0, 0, 3_000, 4_100),
            _reading(0, 0, 5_000, 4_200),
        )
    )
    assert len(rates) == 1
    assert rates[0]["per_second"] == 50.0
    assert rates[0]["gained"] == 200
    assert rates[0]["span_ms"] == 4_000
    assert rates[0]["readings"] == 3


def test_windows_are_reported_in_order() -> None:
    rates = measure(
        (
            _reading(1, 1, 1_000, 100),
            _reading(1, 1, 2_000, 200),
            _reading(0, 0, 1_000, 100),
            _reading(0, 0, 2_000, 150),
        )
    )
    assert [rate["window"] for rate in rates] == [0, 1]
    assert [rate["extractors"] for rate in rates] == [0, 1]


def test_a_single_reading_has_no_slope() -> None:
    """One point is a position, not a rate, and reporting zero would lie."""
    assert measure((_reading(0, 0, 1_000, 4_000),)) == ()


def test_no_readings_at_all_measure_nothing() -> None:
    assert measure(()) == ()


def test_a_window_whose_clock_did_not_move_is_dropped() -> None:
    """Two samples inside one engine millisecond divide by nothing."""
    assert measure((_reading(0, 0, 1_000, 4_000), _reading(0, 0, 1_000, 4_050))) == ()


def test_a_window_where_an_extractor_finished_is_dropped() -> None:
    """The rate belongs to neither count, and averaging invents a third."""
    assert measure((_reading(0, 2, 1_000, 100), _reading(0, 3, 2_000, 200))) == ()


def test_a_window_where_credits_fell_is_dropped() -> None:
    """Something was bought or something was lost.

    Either way the slope is no longer income, and this is the guard that keeps
    a stray purchase from being reported as negative earnings.
    """
    assert measure((_reading(0, 1, 1_000, 900), _reading(0, 1, 2_000, 200))) == ()


def test_the_marginal_extractor_is_the_slope_across_the_range() -> None:
    """Endpoints rather than neighbours, so noise is spread over the widest base."""
    rates = measure(
        (
            _reading(0, 0, 0, 0),
            _reading(0, 0, 1_000, 10),
            _reading(1, 4, 0, 0),
            _reading(1, 4, 1_000, 50),
        )
    )
    # 10/s at none and 50/s at four: forty credits a second over four
    # extractors.
    assert marginal(rates) == 10.0


def test_an_extractor_lost_between_windows_still_gives_a_slope() -> None:
    """The later window can hold fewer, and the endpoints are not the order taken.

    An extractor is a building an opponent can destroy, so a probe run in a
    live skirmish can measure three and then two. Reading the endpoints as
    first-and-last rather than fewest-and-most would invert the slope and
    report an extractor as costing credits.
    """
    rates = measure(
        (
            _reading(0, 3, 0, 0),
            _reading(0, 3, 1_000, 100),
            _reading(1, 1, 0, 0),
            _reading(1, 1, 1_000, 40),
        )
    )
    # 40/s at one and 100/s at three: thirty a second for each of the two.
    assert marginal(rates) == 30.0


def test_one_extractor_count_cannot_give_a_slope() -> None:
    """Measuring the same world twice says nothing about adding to it."""
    rates = measure((_reading(0, 3, 0, 0), _reading(0, 3, 1_000, 30)))
    assert marginal(rates) is None


def test_nothing_measured_has_no_marginal_value() -> None:
    assert marginal(()) is None


def test_payback_is_price_over_rate() -> None:
    assert payback_seconds(700, 10.0) == 70.0


def test_something_that_earns_nothing_never_pays_back() -> None:
    """Reported as None rather than infinity, so the caller has to say it."""
    assert payback_seconds(700, 0.0) is None


def test_the_table_carries_a_header_and_a_row_per_window() -> None:
    rates = measure(
        (
            _reading(0, 0, 0, 0),
            _reading(0, 0, 1_000, 27),
        )
    )
    lines = format_rates(rates)
    assert lines[0].split() == [
        "window",
        "extractors",
        "readings",
        "span_ms",
        "gained",
        "credits/s",
    ]
    assert lines[1].split() == ["0", "0", "2", "1000", "27", "27.00"]


def test_an_empty_table_is_still_a_table() -> None:
    assert len(format_rates(())) == 1
