"""One resting sell-YES order per contract, with auditable fill proxies.

Only subsequent non-block YES-taker volume beyond the required price can
fill an order. A trade-through proxy does not establish queue priority or
counterfactual fills. The result deliberately says ``filled_proxy``.
"""

from __future__ import annotations

from collections.abc import Sequence

from .types import (
    CENT,
    CONTRACT,
    DOLLAR,
    MakerRule,
    MarketHistory,
    Quote,
    ReplayResult,
    ReplayStatus,
)


def _quote_timestamp(quote: Quote) -> int:
    """Return the chronological key used to select the last available quote.

    Args:
        quote: Timestamped quote record.

    Returns:
        Epoch milliseconds.
    """
    return quote["timestamp_ms"]


def quadratic_fee(price: int, contracts: int, rate_bps: int) -> int:
    """Round an order's quadratic fee upward to a whole cent, using integers.

    Args:
        price: Price in ten-thousandths of a dollar.
        contracts: Positive integer order size.
        rate_bps: Quadratic coefficient in basis points, e.g. 700 for 0.07.

    Returns:
        Fee in ten-thousandths of a dollar.

    Raises:
        ValueError: MARKET-010 for invalid fee inputs.
    """
    if not 0 <= price <= DOLLAR or contracts < 1 or not 0 <= rate_bps <= DOLLAR:
        raise ValueError("[MARKET-010] invalid price, contracts or fee coefficient")
    numerator = rate_bps * contracts * price * (DOLLAR - price)
    denominator = DOLLAR * DOLLAR * CENT
    return ((numerator + denominator - 1) // denominator) * CENT


def replay_maker(history: MarketHistory, rule: MakerRule) -> ReplayResult:
    """Replay one validated market history without looking ahead in quote selection.

    Args:
        history: Complete normalized evidence for a single contract.
        rule: Validated frozen execution assumptions.

    Returns:
        A typed result including explicit reasons for all exclusions.
    """
    prior = [q for q in history["quotes"] if q["timestamp_ms"] <= history["decision_ms"]]
    status = ReplayStatus.NO_QUOTE
    offer: int | None = None
    matching = 0
    if prior:
        quote = max(prior, key=_quote_timestamp)
        bid, ask = quote["bid"], quote["ask"]
        if history["decision_ms"] - quote["timestamp_ms"] > rule["max_quote_age_ms"]:
            status = ReplayStatus.STALE
        elif bid is None or ask is None:
            status = ReplayStatus.NO_SIDES
        elif not rule["min_ask"] <= ask <= rule["max_ask"]:
            status = ReplayStatus.OUTSIDE_BAND
        else:
            offer = ask - rule["improve"]
            if offer <= bid:
                status = ReplayStatus.CROSSES
            else:
                for trade in history["trades"]:
                    if (
                        trade["taker_yes"]
                        and not trade["block"]
                        and history["decision_ms"] + rule["latency_ms"]
                        < trade["timestamp_ms"]
                        < history["close_ms"]
                        and trade["price"] >= offer + rule["through"]
                    ):
                        matching += trade["quantity"]
                status = (
                    ReplayStatus.FILLED
                    if matching >= rule["contracts"] * CONTRACT
                    else ReplayStatus.UNFILLED
                )
    fee = 0
    pnl = 0
    capital = 0
    contracts = 0
    if status is ReplayStatus.FILLED:
        assert offer is not None
        contracts = rule["contracts"]
        fee = quadratic_fee(offer, contracts, rule["maker_rate_bps"])
        pnl = (offer - history["settlement"]) * contracts - fee
        capital = (DOLLAR - offer) * contracts
    return ReplayResult(
        ticker=history["ticker"],
        series=history["series"],
        event_day=history["event_day"],
        settlement=history["settlement"],
        status=status,
        offer=offer,
        matching_quantity=matching,
        contracts=contracts,
        fee=fee,
        pnl=pnl,
        capital=capital,
    )


def replay_markets(histories: Sequence[MarketHistory], rule: MakerRule) -> tuple[ReplayResult, ...]:
    """Replay a unique-contract panel, refusing accidental duplicated exposure.

    Args:
        histories: Histories with one decision per contract.
        rule: Frozen order rule.

    Returns:
        Results in input order.

    Raises:
        ValueError: MARKET-011 for an empty or duplicate-contract panel.
    """
    tickers = [h["ticker"] for h in histories]
    if not histories or len(set(tickers)) != len(tickers):
        raise ValueError("[MARKET-011] replay requires a nonempty unique-contract panel")
    return tuple(replay_maker(history, rule) for history in histories)


__all__ = ["quadratic_fee", "replay_maker", "replay_markets"]
