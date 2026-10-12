"""Complete pagination and independently routed quote/trade evidence."""

from __future__ import annotations

from collections.abc import Callable, Generator
from datetime import date

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_list,
    require_str,
)

from .codecs import decode_history, encode_quote, encode_trade
from .kalshi import MarketSpec, decode_candle, decode_market, decode_public_trade
from .replay import replay_maker
from .types import MakerRule, MarketHistory, ReplayStatus, Trade
from .validation import parse_timestamp


class MarketCollector:
    """Collect through an injected archive operation: capture or offline replay."""

    def __init__(self, read: Callable[[str, dict[str, str | int]], JSONObject]) -> None:
        """Bind one explicit acquisition mode.

        Args:
            read: Archive capture, checkpoint continuation, or offline replay.
        """
        self.read = read

    def pages(
        self, path: str, params: dict[str, str | int], key: str
    ) -> Generator[JSONObject, None, None]:
        """Yield every response row, refusing cursor cycles.

        Args:
            path: Public API endpoint.
            params: Fixed query parameters.
            key: Required response array name.

        Yields:
            Native JSON objects in server order.

        Raises:
            ValueError: MARKET-021 for repeated pagination cursors.
            TypeError: Missing or malformed rows or cursor.
        """
        cursor = ""
        seen: set[str] = set()
        while True:
            query = dict(params, limit=1000)
            if cursor:
                query["cursor"] = cursor
            document = self.read(path, query)
            for row in require_list(document, key):
                yield narrow_json_to_dict(row)
            cursor = require_str(document, "cursor")
            if not cursor:
                return
            if cursor in seen:
                raise ValueError("[MARKET-021] repeated pagination cursor")
            seen.add(cursor)

    def inventory(
        self, series: str, timezone: str, start: date, end: date
    ) -> tuple[MarketSpec, ...]:
        """Collect both inventory tiers and enforce a unique settled contract panel.

        Args:
            series: Daily weather series.
            timezone: Placement timezone.
            start: First event date.
            end: Last event date.

        Returns:
            All in-window settled contracts.

        Raises:
            ValueError: MARKET-022 for conflicting or absent selected contracts.
        """
        selected: dict[str, MarketSpec] = {}
        for path in ("/markets", "/historical/markets"):
            params: dict[str, str | int] = {"series_ticker": series}
            if path == "/markets":
                params["status"] = "settled"
            for document in self.pages(path, params, "markets"):
                market = decode_market(document, series, timezone)
                if start.isoformat() <= market["day"] <= end.isoformat():
                    if market["ticker"] in selected and selected[market["ticker"]] != market:
                        raise ValueError(
                            "[MARKET-022] conflicting inventory; reconcile the capture"
                        )
                    selected[market["ticker"]] = market
        if not selected:
            raise ValueError("[MARKET-022] no contracts in the requested date window")
        return tuple(selected[ticker] for ticker in sorted(selected))

    def verify_fee(self, series: str) -> None:
        """Require the zero-maker-fee series schedule assumed by this experiment.

        Args:
            series: Series whose current fee metadata is checked and archived.

        Raises:
            ValueError: MARKET-023 if the required schedule or identity differs.
        """
        metadata = require_dict(self.read("/series/" + series, {}), "series")
        if (
            require_str(metadata, "ticker") != series
            or require_str(metadata, "fee_type") != "quadratic"
            or require_float(metadata, "fee_multiplier") != 1.0
        ):
            raise ValueError("[MARKET-023] experiment requires quadratic fees with multiplier one")

    def history(self, market: MarketSpec, cutoff: JSONObject, rule: MakerRule) -> MarketHistory:
        """Collect prior quotes and subsequent trades for orders the frozen rule permits.

        Market/candle routing uses settlement time. Trade routing uses trade time
        and its own cutoff, including both tiers for a crossing time window.

        Args:
            market: Validated settled contract.
            cutoff: Archived native cutoff response.
            rule: Frozen rule; excluded orders do not require trade requests.

        Returns:
            Evidence sufficient for this rule, with duplicates rejected by its codec.

        Raises:
            ValueError: MARKET-024 for a candle response naming another contract.
            TypeError: Required native schema changed or is missing.
        """
        market_cutoff = parse_timestamp(require_str(cutoff, "market_settled_ts"))
        trade_cutoff = parse_timestamp(require_str(cutoff, "trades_created_ts"))
        ticker, series = market["ticker"], market["series"]
        prefix = "/historical" if market["settled_ms"] < market_cutoff else "/series/" + series
        decision, close = market["decision_ms"], market["close_ms"]
        candles = self.read(
            prefix + "/markets/" + ticker + "/candlesticks",
            {
                "start_ts": decision // 1000 - 21600,
                "end_ts": decision // 1000,
                "period_interval": 60,
            },
        )
        if require_str(candles, "ticker") != ticker:
            raise ValueError("[MARKET-024] candle response contract mismatch")
        quotes: list[JSONValue] = [
            encode_quote(
                decode_candle(narrow_json_to_dict(row), market["settled_ms"] < market_cutoff)
            )
            for row in require_list(candles, "candlesticks")
        ]
        document: JSONObject = {
            "ticker": ticker,
            "series": series,
            "event_day": market["day"],
            "decision_ms": decision,
            "close_ms": close,
            "settlement": market["settlement"],
            "quotes": quotes,
            "trades": [],
        }
        quoted = decode_history(document)
        if replay_maker(quoted, rule)["status"] is not ReplayStatus.UNFILLED:
            return quoted
        paths: list[str] = []
        if decision < trade_cutoff:
            paths.append("/historical/trades")
        if close >= trade_cutoff:
            paths.append("/markets/trades")
        trades: list[JSONValue] = [
            encode_trade(trade)
            for trade in self._trades(
                paths,
                {"ticker": ticker, "min_ts": decision // 1000, "max_ts": close // 1000},
                ticker,
            )
        ]
        document["trades"] = trades
        return decode_history(document)

    def _trades(
        self, paths: list[str], params: dict[str, str | int], ticker: str
    ) -> tuple[Trade, ...]:
        """Reconcile identical cross-tier overlap; reject conflicts and page duplicates.

        Args:
            paths: Trade tiers intersecting the requested time window.
            params: Fixed query parameters.
            ticker: Expected contract identity.

        Returns:
            Unique normalized trades in first-observed order.

        Raises:
            ValueError: MARKET-029 for a page duplicate, MARKET-030 for a conflict.
        """
        records: dict[str, Trade] = {}
        for path in paths:
            tier_ids: set[str] = set()
            for row in self.pages(path, params, "trades"):
                trade = decode_public_trade(row, ticker)
                identity = trade["trade_id"]
                if identity in tier_ids:
                    raise ValueError("[MARKET-029] duplicate trade within one endpoint's pages")
                tier_ids.add(identity)
                if identity in records and records[identity] != trade:
                    raise ValueError("[MARKET-030] conflicting trade evidence across archive tiers")
                records[identity] = trade
        return tuple(records.values())


__all__ = ["MarketCollector"]
