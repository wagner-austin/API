"""Date-balanced city location/scale fitting from observed binary bracket outcomes."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from platform_core.json_utils import JSONObject, require_float
from typing_extensions import ReadOnly, TypedDict

from .forecast_inputs import bracket_probability
from .forecast_records import (
    CITY_COORDINATES,
    ForecastRow,
    decode_forecast_row,
    encode_forecast_row,
)
from .validation import require_bounded_int, require_nonempty

BIAS_GRID = tuple(number / 2 for number in range(-12, 13))
SIGMA_GRID = tuple(number / 2 for number in range(2, 17))
MINIMUM_DATES = 10


class CityCalibration(TypedDict):
    """Immutable fitted city parameters and equal-date training objective."""

    series: ReadOnly[str]
    bias_f: ReadOnly[float]
    sigma_f: ReadOnly[float]
    dates: ReadOnly[int]
    brier: ReadOnly[float]


def encode_city_calibration(model: CityCalibration) -> JSONObject:
    """Encode all fitted parameters and training evidence counts.

    Args:
        model: Validated immutable city fit.

    Returns:
        JSON representation of the entire model.
    """
    return {
        "series": model["series"],
        "bias_f": model["bias_f"],
        "sigma_f": model["sigma_f"],
        "dates": model["dates"],
        "brier": model["brier"],
    }


def decode_city_calibration(document: JSONObject) -> CityCalibration:
    """Require a supported city, declared grid point and sufficient training dates.

    Args:
        document: Parsed JSON model record.

    Returns:
        Validated immutable city model.

    Raises:
        ValueError: CALIBRATION-002 for invalid fitted parameters or objective.
        TypeError: A required field has the wrong type.
    """
    series = require_nonempty(document, "series")
    bias = require_float(document, "bias_f")
    sigma = require_float(document, "sigma_f")
    dates = require_bounded_int(document, "dates", MINIMUM_DATES, 100000)
    brier = require_float(document, "brier")
    if (
        series not in CITY_COORDINATES
        or bias not in BIAS_GRID
        or sigma not in SIGMA_GRID
        or not 0 <= brier <= 1
    ):
        raise ValueError("[CALIBRATION-002] city, grid point or training Brier is invalid")
    return CityCalibration(series=series, bias_f=bias, sigma_f=sigma, dates=dates, brier=brier)


def fit_city_calibration(
    rows: Sequence[ForecastRow],
    bounds: dict[str, tuple[float | None, float | None]],
    series: str,
) -> CityCalibration:
    """Minimize equal-date Brier over a fixed grid using earlier labels only.

    Args:
        rows: Chronologically admissible training observations.
        bounds: Verified native temperature boundaries keyed by ticker.
        series: City whose forecast error distribution is being fitted.

    Returns:
        Deterministic grid optimum, ties choosing smaller bias then smaller sigma.

    Raises:
        ValueError: CALIBRATION-003 when fewer than ten city dates are represented.
        KeyError: A training ticker has no verified bracket.
    """
    dates: dict[str, list[ForecastRow]] = defaultdict(list)
    for row in rows:
        if row["series"] == series:
            dates[row["event_day"]].append(row)
    if len(dates) < MINIMUM_DATES:
        raise ValueError(f"[CALIBRATION-003] {series} requires at least ten training dates")
    candidates: list[tuple[float, float, float]] = []
    for bias in BIAS_GRID:
        for sigma in SIGMA_GRID:
            daily = [
                math.fsum(
                    (
                        bracket_probability(row["forecast_f"] + bias, bounds[row["ticker"]], sigma)
                        - row["yes"]
                    )
                    ** 2
                    for row in dates[day]
                )
                / len(dates[day])
                for day in sorted(dates)
            ]
            candidates.append((math.fsum(daily) / len(daily), bias, sigma))
    brier, bias, sigma = min(candidates)
    return decode_city_calibration(
        {
            "series": series,
            "bias_f": bias,
            "sigma_f": sigma,
            "dates": len(dates),
            "brier": brier,
        }
    )


def apply_city_calibration(
    rows: Sequence[ForecastRow],
    bounds: dict[str, tuple[float | None, float | None]],
    models: dict[str, CityCalibration],
) -> tuple[ForecastRow, ...]:
    """Replace only weather probabilities, preserving every observed field.

    Args:
        rows: Training or evaluation observations.
        bounds: Verified temperature boundaries.
        models: Earlier-period fitted model for every represented city.

    Returns:
        Newly validated immutable rows in their original order.

    Raises:
        KeyError: A city model or native bracket is missing.
        ValueError: A reconstructed observation is invalid.
    """
    output: list[ForecastRow] = []
    for row in rows:
        model = models[row["series"]]
        encoded = encode_forecast_row(row)
        encoded["weather_p"] = bracket_probability(
            row["forecast_f"] + model["bias_f"], bounds[row["ticker"]], model["sigma_f"]
        )
        output.append(decode_forecast_row(encoded))
    return tuple(output)
