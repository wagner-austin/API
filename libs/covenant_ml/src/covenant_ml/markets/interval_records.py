"""Immutable observations and fitted parameters for the crypto development comparison."""

from __future__ import annotations

import math

from platform_core.json_utils import JSONObject, require_float
from typing_extensions import ReadOnly, TypedDict

from .interval_quotes import SERIES
from .validation import require_bounded_int, require_nonempty


class IntervalRow(TypedDict):
    """A prior quote and its separately timestamped binary settlement label."""

    ticker: ReadOnly[str]
    series: ReadOnly[str]
    close_ms: ReadOnly[int]
    decision_ms: ReadOnly[int]
    settled_ms: ReadOnly[int]
    bid: ReadOnly[int]
    ask: ReadOnly[int]
    yes: ReadOnly[int]


def encode_interval_row(row: IntervalRow) -> JSONObject:
    """Encode a complete comparison row.

    Args:
        row: Validated immutable observation.

    Returns:
        All native identities, timing, prices and label.
    """
    return {
        "ticker": row["ticker"],
        "series": row["series"],
        "close_ms": row["close_ms"],
        "decision_ms": row["decision_ms"],
        "settled_ms": row["settled_ms"],
        "bid": row["bid"],
        "ask": row["ask"],
        "yes": row["yes"],
    }


def decode_interval_row(document: JSONObject) -> IntervalRow:
    """Require a binary label, strict interior quote and ordered observation times.

    Args:
        document: Parsed observation fields.

    Returns:
        Immutable typed observation.

    Raises:
        ValueError: IFORE-002 for inconsistent identity, prices or chronology.
        TypeError: Required fields have invalid native types.
    """
    row = IntervalRow(
        ticker=require_nonempty(document, "ticker"),
        series=require_nonempty(document, "series"),
        close_ms=require_bounded_int(document, "close_ms", 0, 2**63 - 1),
        decision_ms=require_bounded_int(document, "decision_ms", 0, 2**63 - 1),
        settled_ms=require_bounded_int(document, "settled_ms", 0, 2**63 - 1),
        bid=require_bounded_int(document, "bid", 1, 9999),
        ask=require_bounded_int(document, "ask", 1, 9999),
        yes=require_bounded_int(document, "yes", 0, 1),
    )
    if (
        row["series"] not in SERIES
        or not row["ticker"].startswith(row["series"] + "-")
        or row["bid"] >= row["ask"]
        or not row["decision_ms"] < row["close_ms"] <= row["settled_ms"]
    ):
        raise ValueError("[IFORE-002] inconsistent comparison identity, quote or timing")
    return row


class IntervalFit(TypedDict):
    """Frozen coefficients and earlier-only smoothed per-asset base rates."""

    intercept: ReadOnly[float]
    market_weight: ReadOnly[float]
    ether_weight: ReadOnly[float]
    bitcoin_base: ReadOnly[float]
    ether_base: ReadOnly[float]
    iterations: ReadOnly[int]


def encode_interval_fit(model: IntervalFit) -> JSONObject:
    """Encode every fitted parameter without mutable estimator state.

    Args:
        model: Validated fitted parameter record.

    Returns:
        Portable model document.
    """
    return {
        "intercept": model["intercept"],
        "market_weight": model["market_weight"],
        "ether_weight": model["ether_weight"],
        "bitcoin_base": model["bitcoin_base"],
        "ether_base": model["ether_base"],
        "iterations": model["iterations"],
    }


def decode_interval_fit(document: JSONObject) -> IntervalFit:
    """Require finite weights, bounded base rates and a converged iteration count.

    Args:
        document: Parsed fitted model.

    Returns:
        Immutable parameter record.

    Raises:
        ValueError: IFORE-004 for nonfinite weights or invalid base rates.
        TypeError: Required fields have invalid types.
    """
    model = IntervalFit(
        intercept=require_float(document, "intercept"),
        market_weight=require_float(document, "market_weight"),
        ether_weight=require_float(document, "ether_weight"),
        bitcoin_base=require_float(document, "bitcoin_base"),
        ether_base=require_float(document, "ether_base"),
        iterations=require_bounded_int(document, "iterations", 0, 1999),
    )
    weights = (model["intercept"], model["market_weight"], model["ether_weight"])
    if not all(math.isfinite(v) for v in weights) or not (
        0 <= model["bitcoin_base"] <= 1 and 0 <= model["ether_base"] <= 1
    ):
        raise ValueError("[IFORE-004] fitted weights and base rates must be finite and valid")
    return model
