"""Native forecast and market documents exercised through actual codecs and HTTP."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.forecast_records import (
    ComparisonPlan,
    ForecastRow,
    decode_comparison_plan,
    decode_forecast_row,
)
from covenant_ml.markets.kalshi import decode_market

from ._server import ApiServer


def comparison_plan() -> ComparisonPlan:
    return decode_comparison_plan(
        {
            "name": "forecast-test",
            "start": "2026-04-03",
            "train_end": "2026-04-08",
            "end": "2026-04-18",
            "protocol_digest": "a" * 64,
            "excluded_days": [],
            "forecast_hour": 0,
        }
    )


def forecast_document(day: date, temperature: float = 16.0) -> JSONObject:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC) - timedelta(days=1)
    times: list[JSONValue] = [
        (start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(72)
    ]
    values: list[JSONValue] = [temperature + (i % 24) / 20 for i in range(72)]
    return {
        "utc_offset_seconds": 0,
        "hourly_units": {"temperature_2m": "°C"},
        "hourly": {"time": times, "temperature_2m": values},
    }


def native_market(day: date, series: str = "KXHIGHTBOS", yes: int = 1) -> JSONObject:
    event = f"{series}-{day.strftime('%y%b%d').upper()}"
    bound = 60 if yes else 62
    close = datetime(day.year, day.month, day.day, 4, tzinfo=UTC) + timedelta(days=1)
    return {
        "ticker": event + f"-B{bound}.5",
        "event_ticker": event,
        "result": "yes" if yes else "no",
        "settlement_value_dollars": "1.0000" if yes else "0.0000",
        "close_time": close.isoformat(),
        "settlement_ts": (close + timedelta(hours=8)).isoformat(),
        "strike_type": "between",
        "floor_strike": bound,
        "cap_strike": bound + 1,
        "rules_primary": "Test temperature bracket settles against the declared published value.",
    }


def forecast_row(day: date = date(2026, 4, 3), yes: int = 1) -> ForecastRow:
    spec = decode_market(native_market(day, yes=yes), "KXHIGHTBOS", "America/New_York")
    initialized = datetime(day.year, day.month, day.day, tzinfo=UTC) - timedelta(days=1)
    return decode_forecast_row(
        {
            "ticker": spec["ticker"],
            "series": spec["series"],
            "event_day": spec["day"],
            "decision_ms": spec["decision_ms"],
            "settled_ms": spec["settled_ms"],
            "quote_ms": spec["decision_ms"],
            "forecast_init_ms": int(initialized.timestamp() * 1000),
            "yes": yes,
            "bid": 4000,
            "ask": 6000,
            "forecast_f": 61.0,
            "weather_p": 0.8 if yes else 0.2,
            "settlement_rule": "Declared source",
        }
    )


def comparison_rows() -> tuple[ForecastRow, ...]:
    return tuple(forecast_row(date(2026, 4, day), yes) for day in range(3, 19) for yes in (0, 1))


def serve_comparison(server: ApiServer) -> None:
    server.documents["/historical/cutoff"] = [{"market_settled_ts": "2027-01-01T00:00:00Z"}]
    server.documents["/markets"] = []
    server.documents["/historical/markets"] = []
    server.documents["/v1/forecast"] = []
    for series in ("KXHIGHTBOS", "KXHIGHPHIL", "KXHIGHTDC"):
        server.documents["/series/" + series] = [
            {
                "series": {
                    "ticker": series,
                    "fee_type": "quadratic",
                    "fee_multiplier": 1,
                }
            }
        ]
        native: list[JSONValue] = []
        for number in range(3, 19):
            day = date(2026, 4, number)
            server.documents["/v1/forecast"].append(forecast_document(day))
            for yes in (0, 1):
                document = native_market(day, series, yes)
                native.append(document)
                spec = decode_market(document, series, "America/New_York")
                path = "/historical/markets/" + spec["ticker"] + "/candlesticks"
                server.documents[path] = [
                    {
                        "ticker": spec["ticker"],
                        "candlesticks": [
                            {
                                "end_period_ts": spec["decision_ms"] // 1000,
                                "yes_bid": {"close": "0.40"},
                                "yes_ask": {"close": "0.60"},
                            }
                        ],
                    }
                ]
        server.documents["/markets"].append({"markets": native, "cursor": ""})
        server.documents["/historical/markets"].append({"markets": native, "cursor": ""})
