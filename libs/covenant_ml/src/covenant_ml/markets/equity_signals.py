"""Immutable financial-strength observations and deterministic ordinal ranking."""

from __future__ import annotations

from math import isfinite

from platform_core.json_utils import JSONObject, require_float
from typing_extensions import ReadOnly, TypedDict

from .validation import require_bounded_int, require_nonempty


class StrengthSignal(TypedDict):
    """An issuer's validated ratios, not a forecast probability or recommendation."""

    symbol: ReadOnly[str]
    cik: ReadOnly[int]
    current_ratio: ReadOnly[float]
    annual_roa: ReadOnly[float]


def encode_strength_signal(record: StrengthSignal) -> JSONObject:
    """Encode immutable issuer ratios for a frozen research snapshot.

    Args:
        record: Validated financial signal.

    Returns:
        JSON observation preserving ticker and issuer identity.
    """
    return {
        "symbol": record["symbol"],
        "cik": record["cik"],
        "current_ratio": record["current_ratio"],
        "annual_roa": record["annual_roa"],
    }


def decode_strength_signal(document: JSONObject) -> StrengthSignal:
    """Require finite ratios and a nonnegative liquidity measure.

    Args:
        document: Untrusted normalized signal observation.

    Returns:
        Immutable issuer ratios with validated identifiers.

    Raises:
        ValueError: EQUITY-011 for invalid ratios; shared identity errors propagate.
        TypeError: Missing or incorrectly typed fields.
    """
    current_ratio, annual_roa = (
        require_float(document, "current_ratio"),
        require_float(document, "annual_roa"),
    )
    if not isfinite(current_ratio) or current_ratio < 0 or not isfinite(annual_roa):
        raise ValueError("[EQUITY-011] require finite ratios and nonnegative current ratio")
    return StrengthSignal(
        symbol=require_nonempty(document, "symbol"),
        cik=require_bounded_int(document, "cik", 1, 9999999999),
        current_ratio=current_ratio,
        annual_roa=annual_roa,
    )


def rank_financial_strength(
    signals: tuple[StrengthSignal, ...],
) -> tuple[tuple[StrengthSignal, int, int], ...]:
    """Rank a fixed cohort by equal-weight strictly-lower feature counts.

    Args:
        signals: Validated immutable observations for unique issuers and tickers.

    Returns:
        Observations and their liquidity and profitability counts, ordered by
        descending summed count and ascending ticker for ties.

    Raises:
        ValueError: EQUITY-012 for empty cohorts, duplicate tickers or issuers.
    """
    if (
        not signals
        or len({row["symbol"] for row in signals}) != len(signals)
        or len({row["cik"] for row in signals}) != len(signals)
    ):
        raise ValueError("[EQUITY-012] cohort must be nonempty with unique tickers and issuers")
    scored = tuple(
        (
            row,
            sum(other["current_ratio"] < row["current_ratio"] for other in signals),
            sum(other["annual_roa"] < row["annual_roa"] for other in signals),
        )
        for row in signals
    )

    def order(row: tuple[StrengthSignal, int, int]) -> tuple[int, str]:
        return -(row[1] + row[2]), row[0]["symbol"]

    return tuple(sorted(scored, key=order))
