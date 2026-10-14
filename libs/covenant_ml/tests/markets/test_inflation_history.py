"""Native chart decoding rejects drift before a forecast can reach a model."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
)

from covenant_ml.markets.inflation_history import (
    decode_inflation_chart,
    decode_inflation_history,
)
from covenant_ml.markets.inflation_records import InflationPoint, forecast_before

from ._inflation_fixtures import inflation_chart


@pytest.mark.parametrize(
    "field,value",
    [("caption", "wrong"), ("yaxisname", "Monthly"), ("subcaption", "2023/12")],
)
def test_chart_identity(field: str, value: str) -> None:
    document = inflation_chart()
    require_dict(document, "chart")[field] = value
    with pytest.raises(ValueError, match="INFLATION-007"):
        decode_inflation_chart(document)


@pytest.mark.parametrize("kind", ["groups", "marker", "label", "reverse", "empty", "vline"])
def test_chart_dates(kind: str) -> None:
    document = inflation_chart()
    groups = require_list(document, "categories")
    labels = require_list(narrow_json_to_dict(groups[0]), "category")
    if kind == "groups":
        groups.append(groups[0])
    elif kind == "marker":
        narrow_json_to_dict(labels[-1])["label"] = "unsupported"
    elif kind == "label":
        narrow_json_to_dict(labels[0])["label"] = "12/1"
    elif kind == "reverse":
        labels[1] = labels[0]
    elif kind == "empty":
        labels.clear()
    else:
        narrow_json_to_dict(labels[-1])["vline"] = "false"
    with pytest.raises(ValueError, match="INFLATION-005"):
        decode_inflation_chart(document)


@pytest.mark.parametrize("kind", ["count", "tooltip", "shape", "number", "bool"])
def test_native_values(kind: str) -> None:
    document = inflation_chart()
    series = narrow_json_to_dict(require_list(document, "dataset")[0])
    points = require_list(series, "data")
    point = narrow_json_to_dict(points[0])
    if kind == "count":
        points.pop()
    elif kind == "tooltip":
        point["tooltext"] = "CPI Inflation{br}12/28{br}2.6{br}"
    elif kind == "shape":
        point["tooltext"] = "invalid"
    elif kind == "number":
        point["value"] = "NaN"
        point["tooltext"] = "CPI Inflation{br}12/29{br}NaN{br}"
    else:
        point["value"] = True
    with pytest.raises(
        TypeError if kind == "bool" else ValueError,
        match="value" if kind == "bool" else "INFLATION-006",
    ):
        decode_inflation_chart(document)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "multiple_actual"])
def test_required_series(kind: str) -> None:
    document = inflation_chart()
    series = require_list(document, "dataset")
    if kind == "duplicate":
        series.append(series[0])
    elif kind == "missing":
        series.pop()
    else:
        actuals = require_list(narrow_json_to_dict(series[1]), "data")
        actuals[1] = {"value": "2.8", "tooltext": "Actual CPI Inflation{br}01/02{br}2.8{br}"}
    with pytest.raises(ValueError, match="INFLATION-007"):
        decode_inflation_chart(document)


def test_empty_actual_and_unselected_series() -> None:
    document = inflation_chart()
    series = require_list(document, "dataset")
    series.append({"seriesname": "PCE Inflation", "data": []})
    require_list(narrow_json_to_dict(series[1]), "data")[2] = {
        "value": "",
        "tooltext": "Actual CPI Inflation{br}01/03{br}{br}",
    }
    month = decode_inflation_chart(document)
    assert month["actual"] is None
    expected: tuple[InflationPoint, ...] = (
        {"day": "2023-12-29", "value": 2.6},
        {"day": "2024-01-02", "value": 2.65},
    )
    assert month["forecasts"] == expected


@pytest.mark.parametrize("value", [[], [inflation_chart(), inflation_chart()]])
def test_feed_order(value: JSONValue) -> None:
    with pytest.raises(ValueError, match="INFLATION-008"):
        decode_inflation_history(value)


def test_feed_type_and_increasing_months() -> None:
    with pytest.raises(TypeError, match="array"):
        decode_inflation_history({})
    earlier: JSONObject = inflation_chart()
    require_dict(earlier, "chart")["subcaption"] = "2022-12"
    decoded = decode_inflation_history([earlier, inflation_chart()])
    assert tuple(month["month"] for month in decoded) == ("2022-12", "2023-12")
    expected: InflationPoint = {"day": "2023-01-03", "value": 2.7}
    assert decoded[0]["actual"] == expected


def test_captured_official_chart() -> None:
    source = Path(__file__).parent / "fixtures" / "inflation_year_2014_01.json"
    month = decode_inflation_chart(narrow_json_to_dict(load_json_str(source.read_text("utf-8"))))
    assert month["month"] == "2014-01"
    assert len(month["forecasts"]) == 33
    expected_actual: InflationPoint = {"day": "2014-02-20", "value": 1.5789473684}
    expected_forecast: InflationPoint = {"day": "2014-02-19", "value": 1.4688104404}
    assert month["actual"] == expected_actual
    assert forecast_before(month, "2014-02-20", 1) == expected_forecast
