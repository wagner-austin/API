"""Carry rates read from a FRED download and fixed at a contract's start."""

from __future__ import annotations

import math

import pytest

from covenant_ml.options.rates import RateObservation, carry_rate, parse_rate_series
from covenant_ml.options.types import ERR_RATE

#: FRED's layout: a header, one row per business day, empty on a holiday.
DOWNLOAD = "observation_date,DGS1\n2019-12-31,1.59\n2020-01-01,\n2020-01-02,-0.05\n"


def test_a_download_reads_every_observed_day_in_order() -> None:
    assert parse_rate_series(DOWNLOAD, "DGS1") == (
        RateObservation("2019-12-31", 1.59),
        RateObservation("2020-01-02", -0.05),
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "observation_date,DGS10\n2019-12-31,1.59\n",
        "observation_date,DGS1\n2019-12-31,1.59,extra\n",
        "observation_date,DGS1\n20191231,1.59\n",
        "observation_date,DGS1\n2019-12-31,1.59\n2019-12-30,1.58\n",
        "observation_date,DGS1\n2019-12-31,1.59\n2019-12-31,1.58\n",
        "observation_date,DGS1\n2019-12-31,.\n",
        "observation_date,DGS1\n2019-12-31,1e2\n",
        "observation_date,DGS1\n2019-12-31,\n",
    ],
)
def test_malformed_downloads_are_refused(text: str) -> None:
    with pytest.raises(ValueError, match=ERR_RATE):
        parse_rate_series(text, "DGS1")


def test_a_start_uses_the_latest_rate_observed_by_then() -> None:
    series = parse_rate_series(DOWNLOAD, "DGS1")
    assert carry_rate(series, "2019-12-31") == pytest.approx(math.log(1.0159))
    assert carry_rate(series, "2020-01-01") == pytest.approx(math.log(1.0159))
    assert carry_rate(series, "2020-01-03") == pytest.approx(math.log(1.0 - 0.0005))


def test_a_start_before_every_observation_is_refused() -> None:
    with pytest.raises(ValueError, match=ERR_RATE):
        carry_rate(parse_rate_series(DOWNLOAD, "DGS1"), "2019-12-30")


def test_a_yield_with_no_continuous_rate_is_refused() -> None:
    with pytest.raises(ValueError, match=ERR_RATE):
        carry_rate((RateObservation("2020-01-02", -100.0),), "2020-01-02")
