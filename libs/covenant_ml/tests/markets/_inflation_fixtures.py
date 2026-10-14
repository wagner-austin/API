"""Small native chart documents exercised through production decoders."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONValue


def inflation_chart() -> JSONObject:
    """Return a December chart crossing into a January release."""
    dates = ("12/29", "01/02", "01/03")
    series: list[JSONValue] = []
    for name, values in (
        ("CPI Inflation", ("2.6", "2.65", "")),
        ("Actual CPI Inflation", ("", "", "2.7")),
    ):
        series.append(
            {
                "seriesname": name,
                "data": [
                    {"value": value, "tooltext": name + "{br}" + day + "{br}" + value + "{br}"}
                    for day, value in zip(dates, values, strict=True)
                ],
            }
        )
    return {
        "chart": {
            "caption": "Inflation Nowcasting",
            "subcaption": "2023-12",
            "yaxisname": "Year-over-year percent change",
        },
        "categories": [
            {
                "category": [
                    {"label": "12/29"},
                    {"label": "01/02"},
                    {"label": "01/03"},
                    {"label": "CPI Dec", "vline": "true"},
                ]
            }
        ],
        "dataset": series,
    }
