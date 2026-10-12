"""Immutable records for historical market replay.

Prices and currency amounts use ten-thousandths of a dollar. Quantities use
hundredths of a contract. Integer arithmetic preserves exchange precision;
only the statistical boundary converts realized payoffs to floating point.
Every record has a codec in ``codecs`` or ``replay_codecs``.
"""

from __future__ import annotations

from enum import StrEnum

from typing_extensions import ReadOnly, TypedDict

DOLLAR = 10_000
CENT = 100
CONTRACT = 100
SCHEMA_VERSION = 1


class Quote(TypedDict):
    """A timestamped binary contract quote, with explicitly missing sides."""

    timestamp_ms: ReadOnly[int]
    bid: ReadOnly[int | None]
    ask: ReadOnly[int | None]


class Trade(TypedDict):
    """A public trade; block prints cannot establish an order-book fill."""

    trade_id: ReadOnly[str]
    timestamp_ms: ReadOnly[int]
    price: ReadOnly[int]
    quantity: ReadOnly[int]
    taker_yes: ReadOnly[bool]
    block: ReadOnly[bool]


class MarketHistory(TypedDict):
    """Evidence for exactly one contract and one predeclared decision time."""

    ticker: ReadOnly[str]
    series: ReadOnly[str]
    event_day: ReadOnly[str]
    decision_ms: ReadOnly[int]
    close_ms: ReadOnly[int]
    settlement: ReadOnly[int]
    quotes: ReadOnly[tuple[Quote, ...]]
    trades: ReadOnly[tuple[Trade, ...]]


class MakerRule(TypedDict):
    """Frozen selection, price, size, freshness and fill-proxy assumptions."""

    min_ask: ReadOnly[int]
    max_ask: ReadOnly[int]
    improve: ReadOnly[int]
    through: ReadOnly[int]
    contracts: ReadOnly[int]
    max_quote_age_ms: ReadOnly[int]
    latency_ms: ReadOnly[int]
    maker_rate_bps: ReadOnly[int]


class ReplayStatus(StrEnum):
    """Why a simulated order did or did not produce a payoff."""

    NO_QUOTE = "no_quote"
    STALE = "stale_quote"
    NO_SIDES = "missing_quote_side"
    OUTSIDE_BAND = "outside_price_band"
    CROSSES = "would_cross_bid"
    UNFILLED = "insufficient_matching_volume"
    FILLED = "filled_proxy"


class ReplayResult(TypedDict):
    """One order's result, never a claim that an actual order was executed."""

    ticker: ReadOnly[str]
    series: ReadOnly[str]
    event_day: ReadOnly[str]
    settlement: ReadOnly[int]
    status: ReadOnly[ReplayStatus]
    offer: ReadOnly[int | None]
    matching_quantity: ReadOnly[int]
    contracts: ReadOnly[int]
    fee: ReadOnly[int]
    pnl: ReadOnly[int]
    capital: ReadOnly[int]


__all__ = [
    "CENT",
    "CONTRACT",
    "DOLLAR",
    "SCHEMA_VERSION",
    "MakerRule",
    "MarketHistory",
    "Quote",
    "ReplayResult",
    "ReplayStatus",
    "Trade",
]
