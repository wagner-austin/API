"""Actual grid fitting, immutable codecs and held-out-label independence."""

from __future__ import annotations

import math

import pytest
from platform_core.json_utils import JSONObject, JSONValue, require_dict, require_float

from covenant_ml.markets.calibration_evaluation import evaluate_city_calibration
from covenant_ml.markets.calibration_model import (
    apply_city_calibration,
    decode_city_calibration,
    encode_city_calibration,
    fit_city_calibration,
)
from covenant_ml.markets.forecast_inputs import bracket_probability
from covenant_ml.markets.forecast_model import chronological_split
from covenant_ml.markets.forecast_records import (
    CITY_COORDINATES,
    ComparisonPlan,
    ForecastRow,
    decode_comparison_plan,
    decode_forecast_row,
    encode_comparison_plan,
    encode_forecast_row,
)

from ._forecast_fixtures import comparison_plan, comparison_rows


def calibration_panel() -> tuple[
    ComparisonPlan, tuple[ForecastRow, ...], dict[str, tuple[float | None, float | None]]
]:
    plan = encode_comparison_plan(comparison_plan())
    plan["train_end"] = "2026-04-14"
    rows: list[ForecastRow] = []
    bounds: dict[str, tuple[float | None, float | None]] = {}
    for series in CITY_COORDINATES:
        for source in comparison_rows():
            row = encode_forecast_row(source)
            ticker = source["ticker"].replace("KXHIGHTBOS", series)
            row["ticker"], row["series"], row["forecast_f"] = ticker, series, 59.0
            bounds[ticker] = (59.5, 61.5) if source["yes"] else (61.5, None)
            row["weather_p"] = bracket_probability(59.0, bounds[ticker], 3.0)
            rows.append(decode_forecast_row(row))
    return decode_comparison_plan(plan), tuple(rows), bounds


@pytest.mark.parametrize("mean,sigma", [(math.nan, 3.0), (0.0, math.inf), (0.0, 0.0), (0.0, -1.0)])
def test_normal_refuses_invalid_location_or_scale(mean: float, sigma: float) -> None:
    with pytest.raises(ValueError, match="CALIBRATION-001"):
        bracket_probability(mean, (None, 1.0), sigma)


def test_normal_scale_changes_mass_and_partition_remains_one() -> None:
    assert bracket_probability(0.0, (-1.0, 1.0), 1.0) == pytest.approx(0.6826894921370859)
    assert bracket_probability(0.0, (-1.0, 1.0), 3.0) == pytest.approx(0.26111731963647267)
    assert (
        sum(
            bracket_probability(0.0, bound, 2.0)
            for bound in ((None, -1.0), (-1.0, 1.0), (1.0, None))
        )
        == 1.0
    )


@pytest.mark.parametrize(
    "field,value", [("series", "OTHER"), ("bias_f", 8.0), ("sigma_f", 0.0), ("brier", math.nan)]
)
def test_model_codec_refuses_invalid_fields(field: str, value: JSONValue) -> None:
    document: JSONObject = {
        "series": "KXHIGHTBOS",
        "bias_f": 0.0,
        "sigma_f": 3.0,
        "dates": 10,
        "brier": 0.2,
    }
    document[field] = value
    with pytest.raises(ValueError, match="CALIBRATION-002"):
        decode_city_calibration(document)


def test_fit_is_date_balanced_and_codecs_roundtrip() -> None:
    plan, rows, bounds = calibration_panel()
    train, _, _ = chronological_split(rows, plan)
    model = fit_city_calibration(train, bounds, "KXHIGHTBOS")
    assert model["dates"] == 11
    assert model["sigma_f"] == 1.0
    assert model["bias_f"] == 1.5
    assert model["brier"] < 0.064
    assert decode_city_calibration(encode_city_calibration(model)) == model
    heterogeneous: list[ForecastRow] = []
    for row in train:
        encoded = encode_forecast_row(row)
        if row["event_day"] == "2026-04-03":
            encoded["yes"] = 1 - row["yes"]
        heterogeneous.append(decode_forecast_row(encoded))
    first_day = tuple(row for row in heterogeneous if row["event_day"] == "2026-04-03")
    varied = fit_city_calibration(heterogeneous, bounds, "KXHIGHTBOS")
    duplicated = fit_city_calibration((*heterogeneous, *first_day), bounds, "KXHIGHTBOS")
    assert varied["brier"] > model["brier"]
    assert duplicated == varied
    with pytest.raises(ValueError, match="CALIBRATION-003"):
        fit_city_calibration(first_day, bounds, "KXHIGHTBOS")


def test_held_out_labels_cannot_change_parameters_or_predictions() -> None:
    plan, rows, bounds = calibration_panel()
    report = evaluate_city_calibration(rows, plan, bounds)
    altered: list[ForecastRow] = []
    for row in rows:
        encoded = encode_forecast_row(row)
        if row["event_day"] > plan["train_end"]:
            encoded["yes"] = 1 - row["yes"]
        altered.append(decode_forecast_row(encoded))
    flipped = evaluate_city_calibration(altered, plan, bounds)
    assert report["city_models"] == flipped["city_models"]
    before = require_dict(report, "city_calibrated")
    after = require_dict(flipped, "city_calibrated")
    assert before["models"] == after["models"]
    before_arms, after_arms = require_dict(before, "arms"), require_dict(after, "arms")
    for arm in before_arms:
        assert (
            require_dict(before_arms, arm)["predictions"]
            == require_dict(after_arms, arm)["predictions"]
        )
    raw = require_dict(before_arms, "raw_forecast")
    assert require_float(raw, "brier") < 0.064
    assert require_float(require_dict(after_arms, "raw_forecast"), "brier") > 0.5
    comparison = require_dict(require_dict(report, "calibrated_minus_fixed"), "raw_forecast")
    assert require_float(comparison, "ci_high") < 0
    models = {series: fit_city_calibration(rows, bounds, series) for series in CITY_COORDINATES}
    calibrated = apply_city_calibration(rows, bounds, models)
    for original, changed in zip(rows, calibrated, strict=True):
        expected = encode_forecast_row(original)
        expected["weather_p"] = changed["weather_p"]
        assert encode_forecast_row(changed) == expected
