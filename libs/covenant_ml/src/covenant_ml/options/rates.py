"""Risk-free carry rates from a FRED constant-maturity Treasury series.

The parent study carries an early payoff to maturity at the one-year Treasury
rate known when the contract starts (FRED ``DGS1``). A FRED download is a CSV
of ``observation_date`` and the series in percent, with an empty value on a
day the series has no observation. The rate a contract uses is the latest
observation dated on or before its start, so a later fixing never reaches it.

``DGS1`` is a bond-equivalent yield. It is converted to a continuously
compounded rate as ``log(1 + y / 100)``, treating it as an annual yield; that
is an approximation the study states, not a market convention it claims.
"""

from __future__ import annotations

import csv
import io
import math
from bisect import bisect_right
from typing import NamedTuple

from .types import ERR_RATE


class RateObservation(NamedTuple):
    """One dated observation of a percent yield.

    Attributes:
        day: ISO date.
        percent: The yield in percent.
    """

    day: str
    percent: float


def _percent(text: str, day: str) -> float:
    """Read one percent value.

    Args:
        text: The CSV cell.
        day: Its date, for the refusal.

    Returns:
        The finite value.

    Raises:
        ValueError: CVML-OPT-003 for text that is not a plain decimal number.
    """
    digits = text.removeprefix("-").replace(".", "", 1)
    if not digits.isdecimal():
        raise ValueError(f"[{ERR_RATE}] {day} has no decimal yield: {text!r}")
    return float(text)


def parse_rate_series(text: str, series: str) -> tuple[RateObservation, ...]:
    """Read a FRED CSV download into dated observations.

    Args:
        text: The whole downloaded file.
        series: The series column FRED names in the header, e.g. ``DGS1``.

    Returns:
        Every day with a value, in date order. Days FRED leaves empty are
        days with no observation and are not returned.

    Raises:
        ValueError: CVML-OPT-003 for a different header, a row of the wrong
            width, a malformed date or value, dates out of order, or no
            observation at all.
    """
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or rows[0] != ["observation_date", series]:
        raise ValueError(f"[{ERR_RATE}] expected the header observation_date,{series}")
    observations: list[RateObservation] = []
    previous = ""
    for row in rows[1:]:
        if len(row) != 2 or len(row[0]) != 10 or row[0][4] != "-" or row[0][7] != "-":
            raise ValueError(f"[{ERR_RATE}] malformed row {row!r}")
        if row[0] <= previous:
            raise ValueError(f"[{ERR_RATE}] dates must increase, {row[0]} follows {previous}")
        previous = row[0]
        if row[1] != "":
            observations.append(RateObservation(row[0], _percent(row[1], row[0])))
    if not observations:
        raise ValueError(f"[{ERR_RATE}] the series has no observations")
    return tuple(observations)


def carry_rate(series: tuple[RateObservation, ...], day: str) -> float:
    """The continuously compounded rate known on a day.

    Args:
        series: Observations in date order.
        day: The contract's start close.

    Returns:
        ``log(1 + y / 100)`` for the latest observation dated on or before ``day``.

    Raises:
        ValueError: CVML-OPT-003 when no observation precedes the day, or the
            yield is at or below -100 percent.
    """
    position = bisect_right([observation.day for observation in series], day)
    if position == 0:
        raise ValueError(f"[{ERR_RATE}] no rate is observed on or before {day}")
    percent = series[position - 1].percent
    if percent <= -100.0:
        raise ValueError(f"[{ERR_RATE}] a yield of {percent}% has no continuous rate")
    return math.log1p(percent / 100.0)


__all__ = ["RateObservation", "carry_rate", "parse_rate_series"]
