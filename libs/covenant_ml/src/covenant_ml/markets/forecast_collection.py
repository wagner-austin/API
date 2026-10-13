"""Join frozen market quotes to explicit issue-time forecasts without target fitting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import date, timedelta

from platform_core.json_utils import JSONObject, require_str

from .collection import MarketCollector
from .forecast_inputs import bracket_probability, forecast_maximum, strike_bounds
from .forecast_records import (
    CITY_COORDINATES,
    ComparisonPlan,
    ForecastRow,
    decode_forecast_row,
)
from .kalshi import MarketSpec, decode_market
from .types import Quote
from .validation import parse_timestamp


def _quote_time(quote: Quote) -> int:
    """Return the timestamp key for strictly prior quote selection.

    Args:
        quote: Timestamped quote.

    Returns:
        Epoch milliseconds.
    """
    return quote["timestamp_ms"]


def select_quote(quotes: tuple[Quote, ...], decision: int) -> tuple[Quote | None, str]:
    """Select a fresh noncrossed two-sided quote, with an explicit exclusion reason.

    Args:
        quotes: Native normalized quote history.
        decision: Frozen decision timestamp in milliseconds.

    Returns:
        Accepted quote and status, or null and its exclusion code.
    """
    prior = [quote for quote in quotes if quote["timestamp_ms"] <= decision]
    if not prior:
        return None, "no_quote"
    quote = max(prior, key=_quote_time)
    if decision - quote["timestamp_ms"] > 3600000:
        return None, "stale_quote"
    bid, ask = quote["bid"], quote["ask"]
    if bid is None or ask is None:
        return None, "missing_quote_side"
    if bid > ask:
        return None, "crossed_quote"
    return quote, "included"


def forecast_inventory(
    plan: ComparisonPlan,
    market: MarketCollector,
    series: str,
) -> tuple[dict[str, JSONObject], dict[str, MarketSpec]]:
    """Reconcile normalized identities and the economic metadata this study needs.

    Args:
        plan: Frozen selected date window.
        market: Shared settled inventory reader.
        series: Requested city series.

    Returns:
        Native documents and normalized specs keyed by the same unique tickers.

    Raises:
        ValueError: FORECAST-005 for conflicting evidence; FORECAST-011 for an empty city.
    """
    native: dict[str, JSONObject] = {}
    specs: dict[str, MarketSpec] = {}
    for document in market.market_documents(series):
        spec = decode_market(document, series, "America/New_York")
        if not plan["start"] <= spec["day"] <= plan["end"]:
            continue
        ticker = spec["ticker"]
        if ticker in native and (
            specs[ticker] != spec
            or strike_bounds(native[ticker]) != strike_bounds(document)
            or require_str(native[ticker], "rules_primary")
            != require_str(document, "rules_primary")
        ):
            raise ValueError("[FORECAST-005] conflicting contract/strike evidence")
        native[ticker], specs[ticker] = document, spec
    if not native:
        raise ValueError(f"[FORECAST-011] no selected contracts for required city {series}")
    return native, specs


def collect_forecast_rows(
    plan: ComparisonPlan,
    market: MarketCollector,
    forecast_read: Callable[[str, dict[str, str | int]], JSONObject],
    progress: Callable[[str], None],
) -> tuple[tuple[ForecastRow, ...], dict[str, int]]:
    """Read each native market once and obtain one forecast per city/event date.

    Args:
        plan: Frozen chronological partitions.
        market: Market collector with explicit archive acquisition mode.
        forecast_read: Explicit capture, resume or replay operation.
        progress: Count-only status sink, never preliminary performance.

    Returns:
        Validated rows and all exclusion counts over unique selected contracts.

    Raises:
        ValueError: FORECAST-005 for conflicting cross-tier market metadata.
        HTTPStatusError: Failed source request; no empty replacement is used.
    """
    cutoff = market.read("/historical/cutoff", {})
    counts: Counter[str] = Counter()
    rows: list[ForecastRow] = []
    for series, (latitude, longitude) in CITY_COORDINATES.items():
        market.verify_fee(series)
        native, specs = forecast_inventory(plan, market, series)
        maxima: dict[str, float] = {}
        for ticker in sorted(native):
            document, spec = native[ticker], specs[ticker]
            counts["selected_contracts"] += 1
            if spec["day"] in plan["excluded_days"]:
                counts["protocol_excluded_day"] += 1
                continue
            if spec["settlement"] not in (0, 10000):
                counts["nonbinary_settlement"] += 1
                continue
            bounds = strike_bounds(document)
            if bounds is None:
                counts["unsupported_strike"] += 1
                continue
            quote, reason = select_quote(market.quotes(spec, cutoff), spec["decision_ms"])
            if quote is None:
                counts[reason] += 1
                continue
            day = date.fromisoformat(spec["day"])
            initialized = (day - timedelta(days=1)).isoformat() + f"T{plan['forecast_hour']:02d}:00"
            if spec["day"] not in maxima:
                forecast = forecast_read(
                    "/v1/forecast",
                    {
                        "latitude": latitude,
                        "longitude": longitude,
                        "hourly": "temperature_2m",
                        "models": "gfs_global",
                        "run": initialized,
                        "forecast_days": 3,
                    },
                )
                maxima[spec["day"]] = forecast_maximum(forecast, day)
            mean = maxima[spec["day"]]
            rows.append(
                decode_forecast_row(
                    {
                        "ticker": ticker,
                        "series": series,
                        "event_day": spec["day"],
                        "decision_ms": spec["decision_ms"],
                        "settled_ms": spec["settled_ms"],
                        "quote_ms": quote["timestamp_ms"],
                        "forecast_init_ms": parse_timestamp(initialized + "Z"),
                        "yes": spec["settlement"] // 10000,
                        "bid": quote["bid"],
                        "ask": quote["ask"],
                        "forecast_f": mean,
                        "weather_p": bracket_probability(mean, bounds, 3.0),
                        "settlement_rule": require_str(document, "rules_primary"),
                    }
                )
            )
            counts["included"] += 1
        progress(f"{series}: {len(specs)} contracts; {len(maxima)} forecast days verified")
    return tuple(rows), dict(counts)


__all__ = ["collect_forecast_rows", "forecast_inventory", "select_quote"]
