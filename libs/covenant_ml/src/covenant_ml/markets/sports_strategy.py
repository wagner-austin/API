"""Frozen chronological probability calibration and explicitly hypothetical payoffs."""

from __future__ import annotations

from collections.abc import Sequence
from math import fsum, pow

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import JSONObject, JSONValue

from ..evaluation.codec_records import encode_interval
from ..evaluation.paired import paired_mean_difference
from ..evaluation.resampling import build_resampling
from ..evaluation.types import ResamplingUnit
from .kalshi import MarketSpec
from .replay import quadratic_fee
from .sports_history import TEST_END, TEST_START, TRAIN_START
from .types import DOLLAR, Quote
from .validation import parse_timestamp

EXPONENTS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def adjusted_probability(probability: float, exponent: float) -> float:
    """Apply the declared probability temperature, without fitting to later games.

    Args:
        probability: Strictly interior market probability.
        exponent: One of the protocol's six fixed candidate values.

    Returns:
        Calibrated probability as a float.

    Raises:
        ValueError: STRATEGY-004 for invalid probability or exponent.
    """
    if not 0 < probability < 1 or exponent not in EXPONENTS:
        raise ValueError("[STRATEGY-004] invalid probability or frozen exponent")
    yes, no = pow(probability, exponent), pow(1 - probability, exponent)
    return yes / (yes + no)


def _prices(quote: Quote) -> tuple[int, int]:
    """Require eligible bid and ask integers at the evaluation boundary.

    Args:
        quote: Timestamped quote record.

    Returns:
        Positive uncrossed bid and ask in ten-thousandths of a dollar.

    Raises:
        ValueError: STRATEGY-005 for missing or ineligible prices.
    """
    bid, ask = quote["bid"], quote["ask"]
    if bid is None or ask is None or not 0 < bid <= ask < DOLLAR or ask - bid > 500:
        raise ValueError("[STRATEGY-005] evaluation requires eligible bid and ask")
    return bid, ask


def _brier(rows: Sequence[tuple[MarketSpec, Quote]], exponent: float) -> float:
    """Score one exponent on the supplied partition only.

    Args:
        rows: Nonempty selected observations.
        exponent: Frozen-grid probability exponent.

    Returns:
        Mean squared probability error.
    """
    return fsum(
        (
            adjusted_probability(sum(_prices(quote)) / (2 * DOLLAR), exponent)
            - spec["settlement"] / DOLLAR
        )
        ** 2
        for spec, quote in rows
    ) / len(rows)


def quote_decision(quote: Quote, probability: float, payout: int) -> JSONObject:
    """Apply the fixed hurdle with exact fees and adverse-price allowance.

    Args:
        quote: Eligible bid and ask record.
        probability: Forecast selected without the outcome.
        payout: Actual binary settlement, used only after action selection.

    Returns:
        Selected side, costs and hypothetical primary and doubled-fee profit.

    Raises:
        ValueError: STRATEGY-006 for invalid forecast or settlement.
    """
    bid, ask = _prices(quote)
    if not 0 <= probability <= 1 or payout not in (0, DOLLAR):
        raise ValueError("[STRATEGY-006] decision needs a probability and binary payout")
    candidates: list[tuple[float, str, int, int, int]] = []
    for side, price, forecast, settlement in (
        ("yes", ask + 100, probability, payout),
        ("no", DOLLAR - bid + 100, 1 - probability, DOLLAR - payout),
    ):
        if price > DOLLAR:
            continue
        fee = quadratic_fee(price, 1, 700)
        candidates.append((forecast - (price + fee) / DOLLAR, side, price, fee, settlement))
    margin, side, price, fee, settlement = max(candidates)
    if margin < 0.03:
        return {
            "side": "pass",
            "cost_units": 0,
            "fee_units": 0,
            "profit_units": 0,
            "double_fee_profit_units": 0,
        }
    return {
        "side": side,
        "cost_units": price + fee,
        "fee_units": fee,
        "profit_units": settlement - price - fee,
        "double_fee_profit_units": settlement - price - 2 * fee,
    }


def _observation_key(row: tuple[MarketSpec, Quote]) -> tuple[str, str]:
    """Order observations independently of their forecasts and outcomes.

    Args:
        row: Specification and matching quote.

    Returns:
        Event day and ticker ordering key.
    """
    return row[0]["day"], row[0]["ticker"]


def evaluate_sports(rows: Sequence[tuple[MarketSpec, Quote]]) -> JSONObject:
    """Fit earlier games once and report later-game forecast and cost screens.

    Args:
        rows: Verified one-per-event observations from the frozen date range.

    Returns:
        Full candidate training scores, evaluation decisions, daily interval and gate.

    Raises:
        ValueError: STRATEGY-007 for duplicate games, invalid time ordering or partition;
            STRATEGY-008 for empty partitions or fewer than two evaluation dates.
    """
    from platform_core.json_utils import require_int, require_str

    seen: set[str] = set()
    train: list[tuple[MarketSpec, Quote]] = []
    test: list[tuple[MarketSpec, Quote]] = []
    for spec, quote in rows:
        event = spec["ticker"].rsplit("-", 1)[0]
        if (
            event in seen
            or not TRAIN_START <= spec["day"] <= TEST_END
            or quote["timestamp_ms"] != spec["decision_ms"]
            or spec["decision_ms"] != parse_timestamp(spec["day"] + "T12:00:00Z")
            or not spec["decision_ms"] < spec["close_ms"] <= spec["settled_ms"]
            or spec["settlement"] not in (0, DOLLAR)
        ):
            raise ValueError("[STRATEGY-007] duplicate game or invalid observation chronology")
        seen.add(event)
        _prices(quote)
        if spec["day"] < TEST_START:
            if spec["settled_ms"] >= parse_timestamp(TEST_START + "T00:00:00Z"):
                raise ValueError("[STRATEGY-007] training outcome unavailable before evaluation")
            train.append((spec, quote))
        else:
            test.append((spec, quote))
    if not train or len({spec["day"] for spec, _ in test}) < 2:
        raise ValueError("[STRATEGY-008] require training games and two evaluation dates")
    scores = {str(exponent): _brier(train, exponent) for exponent in EXPONENTS}
    _, _, exponent = min((scores[str(value)], abs(value - 1), value) for value in EXPONENTS)
    decisions: list[JSONValue] = []
    daily: dict[str, int] = {}
    total, stress, trades = 0, 0, 0
    for spec, quote in sorted(test, key=_observation_key):
        probability = adjusted_probability(sum(_prices(quote)) / (2 * DOLLAR), exponent)
        decision = quote_decision(quote, probability, spec["settlement"])
        profit = require_int(decision, "profit_units")
        daily[spec["day"]] = daily.get(spec["day"], 0) + profit
        total += profit
        stress += require_int(decision, "double_fee_profit_units")
        trades += require_str(decision, "side") != "pass"
        decisions.append(
            {
                "ticker": spec["ticker"],
                "day": spec["day"],
                "probability": probability,
                "settlement": spec["settlement"],
                "bid": quote["bid"],
                "ask": quote["ask"],
                "decision": decision,
            }
        )
    daily_values: list[float] = [daily[day] / DOLLAR for day in sorted(daily)]
    values: NDArray[np.float64] = np.asarray(daily_values, dtype=np.float64)
    groups: NDArray[np.int64] = np.arange(len(values), dtype=np.int64)
    interval = paired_mean_difference(
        values, build_resampling(ResamplingUnit.ROW, len(values), groups, 1), 0.95
    )
    baseline, adjusted = _brier(test, 1.0), _brier(test, exponent)
    gates = {
        "training_games": len(train) >= 100,
        "evaluation_games": len(test) >= 100,
        "evaluation_trades": trades >= 30,
        "improved_brier": adjusted < baseline,
        "positive_daily_lower_bound": interval["ci_low"] > 0,
    }
    return {
        "schema_version": 1,
        "training_games": len(train),
        "evaluation_games": len(test),
        "training_dates": len({spec["day"] for spec, _ in train}),
        "evaluation_dates": len(daily),
        "training_brier": dict(scores),
        "exponent": exponent,
        "market_brier": baseline,
        "adjusted_brier": adjusted,
        "trades": trades,
        "hypothetical_profit_units": total,
        "double_fee_profit_units": stress,
        "daily_profit_units": dict(daily),
        "daily_profit_interval_dollars": encode_interval(interval),
        "gates": dict(gates),
        "advance_to_prospective_demo": all(gates.values()),
        "decisions": decisions,
        "orders_submitted": 0,
        "interpretation": "Retrospective quote-based screen; depth and actual fills unavailable. "
        "Fee coefficient is an assumption, not reconstructed historical charges. "
        "Daily interval assumes independent dates; serial dependence may understate uncertainty.",
    }
