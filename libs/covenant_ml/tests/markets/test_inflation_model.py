"""Probability arithmetic uses prior releases and exact decimal rounding boundaries."""

from __future__ import annotations

from decimal import Decimal

import pytest

from covenant_ml.markets.inflation_model import (
    earlier_errors,
    inflation_probability,
    rounded_inflation,
)
from covenant_ml.markets.inflation_records import InflationMonth

from ._inflation_experiment_fixtures import histories


def test_rounding_and_empirical_counts() -> None:
    assert rounded_inflation(4.6, 0.05) == Decimal("4.7")
    assert rounded_inflation(-2.65) == Decimal("-2.7")
    mixed: tuple[float, ...] = tuple(0.049 if index < 18 else 0.05 for index in range(36))
    assert inflation_probability(2.6, mixed, 2.6) == 0.5
    assert inflation_probability(2.6, (0.05,) * 36, 2.6) == 36.5 / 37
    assert inflation_probability(2.6, (0.049,) * 36, 2.6) == 0.5 / 37


@pytest.mark.parametrize("value,error", [(float("inf"), 0.0), (0.0, float("nan"))])
def test_rounding_finiteness(value: float, error: float) -> None:
    with pytest.raises(ValueError, match="INFLATION-009"):
        rounded_inflation(value, error)


@pytest.mark.parametrize("kind", ["short", "long", "forecast", "strike", "error", "precision"])
def test_invalid_model_inputs(kind: str) -> None:
    errors = (0.1,) * (35 if kind == "short" else 61 if kind == "long" else 36)
    if kind == "error":
        errors = (float("nan"),) * 36
    forecast = float("inf") if kind == "forecast" else 2.6
    strike = float("nan") if kind == "strike" else 2.65 if kind == "precision" else 2.6
    with pytest.raises(ValueError, match="INFLATION-010"):
        inflation_probability(forecast, errors, strike)


def test_prior_release_filter_and_window() -> None:
    history = histories()
    errors = earlier_errors(history, "2024-01", "2024-02-13")
    assert errors == (2.7 - 2.6,) * 60
    missing: InflationMonth = {"month": "2017-01", "forecasts": (), "actual": None}
    stale: InflationMonth = {
        "month": "2017-02",
        "forecasts": (),
        "actual": {"day": "2017-03-13", "value": 2.7},
    }
    assert earlier_errors((missing, stale, history[0]), "2018-02", "2018-02-13") == ()
    assert earlier_errors((history[0],), "2018-02", "2018-02-14") == (2.7 - 2.6,)
    changed: InflationMonth = {
        "month": "2024-01",
        "forecasts": history[-3]["forecasts"],
        "actual": {"day": "2024-02-13", "value": 999.0},
    }
    assert earlier_errors((*history[:-3], changed), "2024-01", "2024-02-13") == errors
