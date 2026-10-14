"""Validated equal-weight paper portfolio outcome arithmetic."""

from __future__ import annotations

from math import isfinite

from platform_core.json_utils import JSONObject, require_float
from typing_extensions import ReadOnly, TypedDict


class EquityWindowResult(TypedDict):
    """Research portfolio returns with an explicit assumed round-trip cost."""

    selected_return: ReadOnly[float]
    benchmark_return: ReadOnly[float]
    round_trip_cost: ReadOnly[float]
    net_selected_return: ReadOnly[float]
    net_excess_return: ReadOnly[float]


def encode_equity_window_result(record: EquityWindowResult) -> JSONObject:
    """Encode a validated portfolio outcome without changing its cost convention.

    Args:
        record: Immutable paper portfolio arithmetic.

    Returns:
        JSON record preserving gross, net and excess returns.
    """
    return {
        "selected_return": record["selected_return"],
        "benchmark_return": record["benchmark_return"],
        "round_trip_cost": record["round_trip_cost"],
        "net_selected_return": record["net_selected_return"],
        "net_excess_return": record["net_excess_return"],
    }


def decode_equity_window_result(document: JSONObject) -> EquityWindowResult:
    """Require finite, internally consistent research returns and costs.

    Args:
        document: Untrusted normalized outcome record.

    Returns:
        Immutable validated portfolio arithmetic.

    Raises:
        ValueError: EQUITY-013 for invalid costs, nonfinite returns or inconsistent
            arithmetic; gross long-only portfolio returns cannot be below -1.
        TypeError: Missing or incorrectly typed numeric fields.
    """
    selected = require_float(document, "selected_return")
    benchmark = require_float(document, "benchmark_return")
    cost = require_float(document, "round_trip_cost")
    net = require_float(document, "net_selected_return")
    excess = require_float(document, "net_excess_return")
    if (
        not all(isfinite(value) for value in (selected, benchmark, cost, net, excess))
        or selected < -1
        or benchmark < -1
        or not 0 <= cost <= 1
        or net != selected - cost
        or excess != net - benchmark
    ):
        raise ValueError("[EQUITY-013] require finite returns, valid cost and exact net arithmetic")
    return EquityWindowResult(
        selected_return=selected,
        benchmark_return=benchmark,
        round_trip_cost=cost,
        net_selected_return=net,
        net_excess_return=excess,
    )
