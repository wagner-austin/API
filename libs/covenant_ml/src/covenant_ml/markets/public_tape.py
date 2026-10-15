"""Strategy-independent public executions with explicit metadata coverage."""

from __future__ import annotations

from collections.abc import Generator
from urllib.parse import urlencode

from platform_core.json_utils import JSONObject, JSONValue, require_str

from .collection import MarketCollector
from .kalshi import decode_public_trade
from .validation import parse_fixed, parse_timestamp, require_nonempty


def _ticker_batches(tickers: tuple[str, ...]) -> Generator[tuple[str, ...], None, None]:
    """Pack identities within a fixed encoded-query budget without truncation.

    Args:
        tickers: Sorted unique native identifiers.

    Yields:
        Nonempty batches with at most 6000 encoded query characters.

    Raises:
        ValueError: TAPE-007 if a single identifier exceeds the request budget.
    """
    batch: list[str] = []
    size = len(urlencode({"limit": 1000, "tickers": ""}))
    base_size = size
    for ticker in tickers:
        encoded_size = len(urlencode({"": ticker})) - 1
        if base_size + encoded_size > 6000:
            raise ValueError("[TAPE-007] contract identifier exceeds query size budget")
        if batch and size + encoded_size + 3 > 6000:
            yield tuple(batch)
            batch = []
            size = base_size
        if batch:
            size += 3
        batch.append(ticker)
        size += encoded_size
    if batch:
        yield tuple(batch)


def _market_documents(
    collector: MarketCollector, tickers: tuple[str, ...]
) -> tuple[JSONObject, ...]:
    """Join every traded contract against both metadata tiers without guessing.

    Args:
        collector: Explicit capture, resume or replay transport.
        tickers: Sorted unique contract identities.

    Returns:
        Native metadata in ticker order, including present-day rules and quotes.

    Raises:
        ValueError: TAPE-003 for duplicates, conflicts or unexpected identities;
            TAPE-004 for missing contracts; TAPE-005 for invalid result fields.
        TypeError: Required metadata has the wrong native type.
    """
    records: dict[str, JSONObject] = {}
    for batch in _ticker_batches(tickers):
        for path in ("/markets", "/historical/markets"):
            seen: set[str] = set()
            for row in collector.pages(path, {"tickers": ",".join(batch)}, "markets"):
                ticker = require_nonempty(row, "ticker")
                if ticker not in batch or ticker in seen:
                    raise ValueError("[TAPE-003] unexpected or duplicate metadata identity")
                seen.add(ticker)
                require_nonempty(row, "event_ticker")
                require_nonempty(row, "status")
                result = require_str(row, "result")
                if result not in ("", "yes", "no", "scalar"):
                    raise ValueError("[TAPE-005] unsupported native market result")
                if result:
                    payout = parse_fixed(require_str(row, "settlement_value_dollars"), 4)
                    if (
                        not 0 <= payout <= 10000
                        or (result == "yes" and payout != 10000)
                        or (result == "no" and payout != 0)
                    ):
                        raise ValueError("[TAPE-005] result disagrees with cash settlement")
                if ticker in records and records[ticker] != row:
                    raise ValueError("[TAPE-003] conflicting metadata across tiers")
                records[ticker] = row
    missing = sorted(set(tickers) - records.keys())
    if missing:
        raise ValueError(f"[TAPE-004] missing metadata for traded contracts: {missing}")
    return tuple(records[ticker] for ticker in tickers)


def collect_public_tape(collector: MarketCollector, start_ts: int, end_ts: int) -> JSONObject:
    """Collect all exposed executions in a fixed half-open UTC seconds interval.

    This is an execution census, not a universe of untraded opportunities.
    Outcome and quote fields in joined metadata are known at capture time only.
    No trade is removed because of price, size, result, category or block status.

    Args:
        collector: Shared collector bound to one explicit archive mode.
        start_ts: Inclusive Unix second, nonnegative.
        end_ts: Exclusive Unix second, at most one day after start_ts.

    Returns:
        Validated native rows, metadata and exact coverage counts. Native
        timestamps retain their original precision; no queue order is inferred.

    Raises:
        ValueError: TAPE-001 for an invalid window; TAPE-002 for out-of-window
            trades or invalid book side; TAPE-003 through TAPE-005 for metadata
            quality failures. Shared pagination and schema errors propagate.
        HTTPStatusError: A capture request fails; no partial report is returned.
    """
    if not 0 <= start_ts < end_ts <= start_ts + 86400:
        raise ValueError("[TAPE-001] require a positive window of at most 86400 seconds")
    cutoff = collector.read("/historical/cutoff", {})
    boundary = parse_timestamp(require_str(cutoff, "trades_created_ts"))
    paths: list[str] = []
    if start_ts * 1000 < boundary:
        paths.append("/historical/trades")
    if end_ts * 1000 > boundary:
        paths.append("/markets/trades")
    # Preserve the final fractional second and tolerate an exclusive minimum.
    query_start = max(0, start_ts - 1)
    rows = collector.trade_documents(paths, {"min_ts": query_start, "max_ts": end_ts})
    tickers: set[str] = set()
    trades: list[JSONValue] = []
    quantities = 0
    blocks = 0
    boundary_rows = 0
    for row in rows:
        ticker = require_nonempty(row, "ticker")
        trade = decode_public_trade(row, ticker)
        if not query_start * 1000 <= trade["timestamp_ms"] <= end_ts * 1000 or require_str(
            row, "taker_book_side"
        ) not in ("bid", "ask"):
            raise ValueError("[TAPE-002] trade outside requested window or invalid book side")
        no_price = parse_fixed(require_str(row, "no_price_dollars"), 4)
        if trade["price"] + no_price != 10000:
            raise ValueError("[TAPE-002] YES and NO trade prices are not complementary")
        if not start_ts * 1000 <= trade["timestamp_ms"] < end_ts * 1000:
            boundary_rows += 1
            continue
        quantities += trade["quantity"]
        blocks += int(trade["block"])
        tickers.add(ticker)
        trades.append(row)
    metadata = _market_documents(collector, tuple(sorted(tickers)))
    events = {require_str(row, "event_ticker") for row in metadata}
    declared = sum(require_str(row, "result") != "" for row in metadata)
    markets: list[JSONValue] = list(metadata)
    return {
        "schema_version": 1,
        "start_ts_inclusive": start_ts,
        "end_ts_exclusive": end_ts,
        "trade_tiers": list(paths),
        "trade_count": len(trades),
        "contract_count": len(tickers),
        "event_count": len(events),
        "contracts_with_declared_result_at_capture": declared,
        "contracts_without_declared_result_at_capture": len(metadata) - declared,
        "quantity_centi_contracts": quantities,
        "block_trade_count": blocks,
        "boundary_rows_excluded": boundary_rows,
        "trades": trades,
        "markets": markets,
        "orders_submitted": 0,
        "interpretation": (
            "Public executions only; event IDs are not independent observations. "
            "Metadata, results and quotes are capture-time snapshots, not past features. "
            "No order-book history, fees, trader identities, queue position or profitability "
            "is established. Untraded contracts are outside this cohort."
        ),
    }


__all__ = ["collect_public_tape"]
