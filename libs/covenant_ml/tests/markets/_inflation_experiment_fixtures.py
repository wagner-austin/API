"""Native monthly histories and contracts for real experiment execution."""

from __future__ import annotations

from calendar import month_name
from datetime import date, timedelta

from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.inflation_history import decode_inflation_history
from covenant_ml.markets.inflation_records import InflationMonth


def feed() -> list[JSONValue]:
    """Build seventy-five ordered native charts with known earlier errors."""
    charts: list[JSONValue] = []
    for offset in range(75):
        year, month = 2018 + offset // 12, offset % 12 + 1
        release = date(year + int(month == 12), month % 12 + 1, 13)
        before = release - timedelta(days=1)
        labels = (before.strftime("%m/%d"), release.strftime("%m/%d"))
        dataset: list[JSONValue] = []
        for name, values in (("CPI Inflation", ("2.6", "")), ("Actual CPI Inflation", ("", "2.7"))):
            dataset.append(
                {
                    "seriesname": name,
                    "data": [
                        {
                            "value": value,
                            "tooltext": name + "{br}" + label + "{br}" + value + "{br}",
                        }
                        for label, value in zip(labels, values, strict=True)
                    ],
                }
            )
        charts.append(
            {
                "chart": {
                    "caption": "Inflation Nowcasting",
                    "subcaption": f"{year}-{month}",
                    "yaxisname": "Year-over-year percent change",
                },
                "categories": [{"category": [{"label": label} for label in labels]}],
                "dataset": dataset,
            }
        )
    return charts


def histories() -> tuple[InflationMonth, ...]:
    """Run the native feed through the production decoder."""
    return decode_inflation_history(feed())


def market(month: int = 1) -> JSONObject:
    """Return a settled 2024 native inflation threshold with exact rules."""
    event = "KXCPIYOY-24" + month_name[month][:3].upper()
    day = f"2024-{month + 1:02d}-13"
    return {
        "ticker": event + "-T2.6",
        "event_ticker": event,
        "floor_strike": 2.6,
        "strike_type": "greater",
        "status": "finalized",
        "result": "yes",
        "settlement_value_dollars": "1.0000",
        "open_time": "2023-12-01T00:00:00Z",
        "close_time": day + "T13:29:00Z",
        "settlement_ts": day + "T14:00:00Z",
        "rules_primary": f"If the Consumer Price Index (CPI) increases by more than 2.6% in the "
        f"twelve months ending {month_name[month]} 2024 (as represented by the one-decimal place "
        "value reported by the Bureau of Labor Statistics), then the market resolves to Yes.",
    }
