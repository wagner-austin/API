"""Fixed causal quote diagnostics, independent of order selection and outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from itertools import pairwise
from math import fsum, sqrt

from platform_core.json_utils import JSONObject, JSONValue

from .regime_records import (
    HOUR_MS,
    QUOTE_COUNT,
    QuoteRegime,
    decode_quote_regime,
    encode_quote_regime,
)
from .statistics import encode_summary, summarize_replay
from .types import CENT, MarketHistory, ReplayResult

REGIME_LABELS = (
    "low_falling",
    "low_flat",
    "low_rising",
    "high_falling",
    "high_flat",
    "high_rising",
    "unclassified",
)


def pre_entry_regime(history: MarketHistory) -> QuoteRegime:
    """Measure only six exact hourly closes ending at the decision cutoff.

    Args:
        history: Validated unique-timestamp market evidence.

    Returns:
        Population standard deviation of five midpoint changes and total change,
        in ten-thousandths of a dollar, or explicit nulls for incomplete coverage.
    """
    grid = {history["decision_ms"] - offset * HOUR_MS for offset in range(QUOTE_COUNT)}
    midpoints = sorted(
        (quote["timestamp_ms"], (quote["bid"] + quote["ask"]) / 2)
        for quote in history["quotes"]
        if quote["timestamp_ms"] in grid and quote["bid"] is not None and quote["ask"] is not None
    )
    variation: float | None = None
    change: float | None = None
    if len(midpoints) == QUOTE_COUNT:
        changes = [right[1] - left[1] for left, right in pairwise(midpoints)]
        mean = fsum(changes) / len(changes)
        variation = sqrt(fsum((value - mean) ** 2 for value in changes) / len(changes))
        change = midpoints[-1][1] - midpoints[0][1]
    return decode_quote_regime(
        {"usable_quotes": len(midpoints), "hourly_change_std": variation, "net_change": change}
    )


def regime_label(record: QuoteRegime) -> str:
    """Classify with the fixed one-cent variability threshold and signed movement.

    Args:
        record: Validated pre-entry measurements.

    Returns:
        One of six descriptive groups, or unclassified for incomplete evidence.
    """
    variation, change = record["hourly_change_std"], record["net_change"]
    if variation is None or change is None:
        return "unclassified"
    band = "low" if variation < CENT else "high"
    direction = "falling" if change < 0 else "rising" if change > 0 else "flat"
    return band + "_" + direction


def regime_manifest(features: Mapping[str, QuoteRegime]) -> JSONObject:
    """Record the fixed definition and every contract's coverage and measurements.

    Args:
        features: Pre-entry measurements keyed by unique contract ticker.

    Returns:
        Complete descriptive definition and sorted per-contract feature records.
    """
    contracts: list[JSONValue] = [
        {
            "ticker": ticker,
            "measurements": encode_quote_regime(record),
            "regime": regime_label(record),
        }
        for ticker, record in sorted(features.items())
    ]
    return {
        "definition": "hourly-probability-movement-v1",
        "required_closes": QUOTE_COUNT,
        "spacing_ms": HOUR_MS,
        "last_close": "decision_ms",
        "variation": "population_std_of_midpoint_changes",
        "high_variation_at_or_above": CENT,
        "units": "ten_thousandths_of_a_dollar",
        "direction": "sign_of_final_minus_initial_midpoint",
        "use": "descriptive_only_no_trade_selection_or_gate_changes",
        "contracts": contracts,
    }


def summarize_regimes(
    results: Sequence[ReplayResult],
    features: Mapping[str, QuoteRegime],
    start: date,
    end: date,
    friction: int,
) -> JSONObject:
    """Partition all contracts and reuse the existing dependence-aware summaries.

    Args:
        results: Complete unique-contract replay, including every exclusion.
        features: Matching pre-entry measurements for every contract.
        start: First declared calendar date.
        end: Last declared calendar date.
        friction: Extra execution cost per filled contract in price units.

    Returns:
        All seven groups, including empty groups and unavailable evidence.

    Raises:
        ValueError: REGIME-003 for mismatched or duplicate identities;
            shared statistical validation failures propagate.
    """
    if len(results) != len(features) or {row["ticker"] for row in results} != set(features):
        raise ValueError("[REGIME-003] regimes require exactly one matching record per contract")
    groups: JSONObject = {}
    for label in REGIME_LABELS:
        rows = [row for row in results if regime_label(features[row["ticker"]]) == label]
        groups[label] = encode_summary(summarize_replay(rows, start, end, friction))
    return groups
