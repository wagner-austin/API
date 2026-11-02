"""Quoted taker policy with exact currency accounting and shared action selection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import JSONObject, JSONValue
from typing_extensions import ReadOnly, TypedDict

from ..evaluation.codec_records import encode_interval
from ..evaluation.paired import paired_mean_difference
from ..evaluation.policy import CONTRACT_ACTIONS, choose_actions, payoff_table
from ..evaluation.resampling import build_resampling
from ..evaluation.types import ResamplingUnit
from .replay import quadratic_fee
from .types import CENT

#: Quadratic taker fee coefficient in basis points (0.07), rounded up to a cent per order.
QUOTED_FEE_BPS = 700
#: Expected value, in ten-thousandths of a dollar, an action must clear after its fee.
QUOTED_HURDLE = 200
#: Additional friction per trade in the sensitivity result: one cent.
QUOTED_STRESS = CENT


class QuotedContract(TypedDict):
    """The quote and outcome of one binary contract, all the taker policy reads.

    A retrospective ``ForecastRow`` and a prospective campaign contract both carry
    these fields, so both are scored by the same policy.
    """

    event_day: ReadOnly[str]
    bid: ReadOnly[int]
    ask: ReadOnly[int]
    yes: ReadOnly[int]


def cluster_interval(values: Sequence[float], rows: Sequence[QuotedContract]) -> JSONObject:
    """Estimate a paired mean with event dates as the independent clusters.

    Args:
        values: Per-row economic values or paired loss differences.
        rows: Corresponding observations in exactly the same order.

    Returns:
        Existing encoded interval record.

    Raises:
        ValueError: Invalid lengths or fewer than two event dates.
    """
    array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    group_values: list[int] = [int(row["event_day"].replace("-", "")) for row in rows]
    groups: NDArray[np.int64] = np.asarray(group_values, dtype=np.int64)
    units = build_resampling(ResamplingUnit.CLUSTER, len(rows), groups, 1)
    return encode_interval(paired_mean_difference(array, units, 0.95))


def quoted_policy(rows: Sequence[QuotedContract], probabilities: Sequence[float]) -> JSONObject:
    """Select pass/YES/NO with a two-cent expected-value hurdle and exact fees.

    Args:
        rows: Held-out observations, including actual contract outcomes.
        probabilities: Frozen-model probabilities before seeing those outcomes.

    Returns:
        Every action/payoff, total P&L, fill count, and additional-friction sensitivity.

    Raises:
        ValueError: FORECAST-009 for invalid or misaligned probabilities.
    """
    if len(rows) != len(probabilities) or any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("[FORECAST-009] policy probabilities must align and lie in [0, 1]")
    decision: NDArray[np.float64] = np.zeros((len(rows), 3, 2), dtype=np.float64)
    fees: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        yes_price, no_price = row["ask"], 10000 - row["bid"]
        yes_fee = quadratic_fee(yes_price, 1, QUOTED_FEE_BPS)
        no_fee = quadratic_fee(no_price, 1, QUOTED_FEE_BPS)
        fees.append((yes_fee, no_fee))
        for action, price, fee in ((1, yes_price, yes_fee), (2, no_price, no_fee)):
            if not 0 < price < 10000:
                decision[index, action, :] = -1.0
            else:
                cost = (price + fee + QUOTED_HURDLE) / 10000
                decision[index, action, 0] = float(action == 2) - cost
                decision[index, action, 1] = float(action == 1) - cost
    proba: NDArray[np.float64] = np.asarray(probabilities, dtype=np.float64)
    actions = choose_actions(proba, payoff_table(CONTRACT_ACTIONS, decision))
    pnl: list[int] = []
    encoded_actions: list[JSONValue] = []
    encoded_fees: list[JSONValue] = []
    trades = 0
    for index, row in enumerate(rows):
        action_value: np.int64 = actions[index]
        action = int(action_value)
        fee = 0
        amount = 0
        if action == 1:
            fee = fees[index][0]
            amount = row["yes"] * 10000 - row["ask"] - fee
        elif action == 2:
            fee = fees[index][1]
            amount = (1 - row["yes"]) * 10000 - (10000 - row["bid"]) - fee
        trades += int(action != 0)
        pnl.append(amount)
        encoded_actions.append(CONTRACT_ACTIONS[action])
        encoded_fees.append(fee)
    values: list[float] = [value / 10000 for value in pnl]
    encoded_pnl: list[JSONValue] = list(pnl)
    return {
        "trades": trades,
        "pnl": sum(pnl),
        "extra_cent_pnl": sum(pnl) - trades * QUOTED_STRESS,
        "per_opportunity": cluster_interval(values, rows),
        "actions": encoded_actions,
        "pnl_by_row": encoded_pnl,
        "fees_by_row": encoded_fees,
        "interpretation": "Hypothetical quoted taker fills; depth and latency unverified.",
    }


__all__ = [
    "QUOTED_FEE_BPS",
    "QUOTED_HURDLE",
    "QUOTED_STRESS",
    "QuotedContract",
    "cluster_interval",
    "quoted_policy",
]
