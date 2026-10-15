"""Paired calendar-day scores and explicitly hypothetical quote-based execution costs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import JSONObject, JSONValue

from ..evaluation.codec_records import encode_interval
from ..evaluation.paired import paired_mean_difference
from ..evaluation.policy import binary_contract_payoffs, choose_actions, realized_utility
from ..evaluation.resampling import build_resampling
from ..evaluation.types import ResamplingUnit
from .forecast_model import float_values
from .interval_records import IntervalRow
from .replay import quadratic_fee


def interval_daily_estimate(values: tuple[float, ...], rows: tuple[IntervalRow, ...]) -> JSONObject:
    """Average within shared close dates and estimate seven-lag serial uncertainty.

    Args:
        values: Per-opportunity differences or payoffs, aligned to rows.
        rows: Complete evaluation opportunities for all arms.

    Returns:
        Daily values and shared Newey-West interval, with equal calendar-day weighting.

    Raises:
        ValueError: IFORE-006 for fewer than nine dates or a missing calendar day.
    """
    daily: dict[int, list[float]] = {}
    for value, row in zip(values, rows, strict=True):
        daily.setdefault(row["close_ms"] // 86400000, []).append(value)
    days = sorted(daily)
    if len(days) < 9 or days != list(range(days[0], days[-1] + 1)):
        raise ValueError("[IFORE-006] require at least nine consecutive evaluation dates")
    means = [sum(daily[day]) / len(daily[day]) for day in days]
    array: NDArray[np.float64] = np.asarray(means, dtype=np.float64)
    indices: NDArray[np.int64] = np.arange(len(means), dtype=np.int64)
    units = build_resampling(ResamplingUnit.BLOCK, len(means), indices, 8)
    encoded: list[JSONValue] = [
        {"utc_day_number": day, "opportunities": len(daily[day]), "mean": mean}
        for day, mean in zip(days, means, strict=True)
    ]
    return {
        "days": encoded,
        "interval": encode_interval(paired_mean_difference(array, units, 0.95)),
    }


def interval_quoted_policy(
    rows: tuple[IntervalRow, ...],
    probabilities: tuple[float, ...],
    friction: int,
) -> JSONObject:
    """Apply identical one-contract taker decisions under the declared cost scenario.

    Args:
        rows: Decoder-validated observations with strictly interior quote sides.
        probabilities: Frozen arm predictions aligned to observations.
        friction: Extra currency units per purchased contract, one unit is $0.0001.

    Returns:
        Every action, fee and integer payoff plus calendar-day uncertainty.

    Raises:
        ValueError: IFORE-007 for misaligned probabilities or undeclared friction.
    """
    if (
        len(rows) != len(probabilities)
        or friction not in (0, 100, 200)
        or any(not 0 <= value <= 1 for value in probabilities)
    ):
        raise ValueError("[IFORE-007] require aligned probabilities and a declared cost scenario")
    yes_prices = [row["ask"] for row in rows]
    no_prices = [10000 - row["bid"] for row in rows]
    yes_fees = [quadratic_fee(price, 1, 700) for price in yes_prices]
    no_fees = [quadratic_fee(price, 1, 700) for price in no_prices]
    yes: NDArray[np.float64] = np.asarray(yes_prices, dtype=np.float64) / 10000
    no: NDArray[np.float64] = np.asarray(no_prices, dtype=np.float64) / 10000
    yes_cost: NDArray[np.float64] = (np.asarray(yes_fees, dtype=np.float64) + friction) / 10000
    no_cost: NDArray[np.float64] = (np.asarray(no_fees, dtype=np.float64) + friction) / 10000
    actual = binary_contract_payoffs(yes, no, yes_cost, no_cost)
    decision = binary_contract_payoffs(yes, no, yes_cost + 0.02, no_cost + 0.02)
    proba: NDArray[np.float64] = np.asarray(probabilities, dtype=np.float64)
    label_values: list[int] = [row["yes"] for row in rows]
    labels: NDArray[np.int64] = np.asarray(label_values, dtype=np.int64)
    actions = choose_actions(proba, decision)
    values = float_values(realized_utility(actions, labels, actual))
    integer_payouts: list[int] = [round(value * 10000) for value in values]
    payouts: list[JSONValue] = list(integer_payouts)
    chosen: list[JSONValue] = []
    fees: list[JSONValue] = []
    for index in range(len(rows)):
        scalar: np.int64 = actions[index]
        action = int(scalar)
        chosen.append(actual.action_names[action])
        fees.append((0, yes_fees[index], no_fees[index])[action])
    return {
        "friction": friction,
        "actions": chosen,
        "fees": fees,
        "pnl_by_row": payouts,
        "pnl": sum(integer_payouts),
        "trades": sum(value != "pass" for value in chosen),
        "daily": interval_daily_estimate(values, rows),
        "interpretation": (
            "Hypothetical taker quotes; historical fees, depth and latency unverified."
        ),
    }
