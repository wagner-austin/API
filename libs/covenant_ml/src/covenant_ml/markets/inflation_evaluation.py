"""Release-weighted inflation forecast scoring and one-contract hypothetical trades."""

from __future__ import annotations

from math import fsum

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import JSONObject, JSONValue, require_int, require_str

from ..evaluation.codec_records import encode_interval
from ..evaluation.paired import paired_mean_difference
from ..evaluation.resampling import build_resampling
from ..evaluation.types import ResamplingUnit
from .codecs import encode_quote
from .kalshi import MarketSpec
from .sports_strategy import quote_decision
from .types import DOLLAR, Quote
from .validation import parse_timestamp


def _candidate_key(row: tuple[str, float, float, float, JSONObject]) -> tuple[float, str]:
    """Rank a candidate without reading its settlement or payoff.

    Args:
        row: Ticker, disagreement, scored losses and hypothetical decision.

    Returns:
        Descending disagreement and ascending ticker key.
    """
    return -row[1], row[0]


def evaluate_inflation(rows: tuple[tuple[MarketSpec, Quote, float], ...]) -> JSONObject:
    """Score paired probabilities and trade at most one contract per release.

    Args:
        rows: Eligible exact-cutoff quotes and earlier-fitted probabilities.

    Returns:
        Release-weighted Brier scores, decisions, costs, interval and frozen gates.

    Raises:
        ValueError: INFLATION-014 for duplicated contracts or timing drift;
            INFLATION-015 for fewer than two eligible release dates.
            The shared quote decision rejects invalid forecasts, quotes or payouts.
        TypeError: Required encoded quote prices are not integers.
    """
    by_day: dict[str, list[tuple[str, float, float, float, JSONObject]]] = {}
    seen: set[str] = set()
    for spec, quote, probability in rows:
        ticker, day = spec["ticker"], spec["day"]
        if (
            ticker in seen
            or quote["timestamp_ms"] != spec["decision_ms"]
            or spec["decision_ms"] != parse_timestamp(day + "T00:00:00Z")
        ):
            raise ValueError("[INFLATION-014] duplicate contract or midnight quote mismatch")
        seen.add(ticker)
        decision = quote_decision(quote, probability, spec["settlement"])
        encoded = encode_quote(quote)
        midpoint = (require_int(encoded, "bid") + require_int(encoded, "ask")) / (2 * DOLLAR)
        outcome = spec["settlement"] / DOLLAR
        by_day.setdefault(day, []).append(
            (
                ticker,
                abs(probability - midpoint),
                (probability - outcome) ** 2,
                (midpoint - outcome) ** 2,
                decision,
            )
        )
    if len(by_day) < 2:
        raise ValueError("[INFLATION-015] comparison requires two eligible release dates")
    decisions: list[JSONValue] = []
    model_losses: list[float] = []
    market_losses: list[float] = []
    daily: dict[str, int] = {}
    total, stress, trades = 0, 0, 0
    for day in sorted(by_day):
        candidates = by_day[day]
        model_losses.append(fsum(row[2] for row in candidates) / len(candidates))
        market_losses.append(fsum(row[3] for row in candidates) / len(candidates))
        ticker, disagreement, _, _, decision = min(candidates, key=_candidate_key)
        profit = require_int(decision, "profit_units")
        daily[day] = profit
        total += profit
        stress += require_int(decision, "double_fee_profit_units")
        trades += require_str(decision, "side") != "pass"
        decisions.append(
            {
                "day": day,
                "ticker": ticker,
                "eligible_contracts": len(candidates),
                "absolute_disagreement": disagreement,
                "decision": decision,
            }
        )
    daily_values: list[float] = [daily[day] / DOLLAR for day in sorted(daily)]
    values: NDArray[np.float64] = np.asarray(daily_values, dtype=np.float64)
    groups: NDArray[np.int64] = np.arange(len(values), dtype=np.int64)
    interval = paired_mean_difference(
        values, build_resampling(ResamplingUnit.ROW, len(values), groups, 1), 0.95
    )
    model_brier = fsum(model_losses) / len(model_losses)
    market_brier = fsum(market_losses) / len(market_losses)
    gates = {
        "release_dates": len(by_day) >= 24,
        "trades": trades >= 20,
        "improved_brier": model_brier < market_brier,
        "positive_lower_bound": interval["ci_low"] > 0,
        "positive_double_fee_profit": stress > 0,
    }
    return {
        "schema_version": 1,
        "exploratory": True,
        "eligible_contracts": len(rows),
        "release_dates": len(by_day),
        "model_brier": model_brier,
        "market_brier": market_brier,
        "trades": trades,
        "hypothetical_profit_units": total,
        "double_fee_profit_units": stress,
        "daily_profit_units": dict(daily),
        "release_profit_interval_dollars": encode_interval(interval),
        "gates": dict(gates),
        "advance_to_prospective_demo": all(gates.values()),
        "decisions": decisions,
        "orders_submitted": 0,
        "interpretation": "Exploratory retrospective quote comparison; historical feed receipt, "
        "advance release calendars, depth and fills are unverified. Fees are assumed. "
        "The interval assumes independent release dates and may understate serial dependence.",
    }
