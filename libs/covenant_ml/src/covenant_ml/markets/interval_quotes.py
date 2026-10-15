"""Attach strictly prior quotes to an outcome-independent crypto interval panel."""

from __future__ import annotations

from collections import Counter

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)

from .codecs import encode_quote
from .collection import MarketCollector
from .event_cohort import require_cohort_window
from .kalshi import decode_candle
from .types import Quote
from .validation import parse_timestamp

SERIES = ("KXBTC15M", "KXETH15M")


def require_interval_plan(plan: JSONObject, cohort: JSONObject) -> None:
    """Require a contained window and an explicit lagged decision rule.

    Args:
        plan: Window, decision offset, minimum lag and maximum quote age in seconds.
        cohort: Verified complete event inventory.

    Raises:
        ValueError: IQUOTE-001 for an unsupported or uncontained plan.
        TypeError: Required fields are malformed.
    """
    start, end = require_int(plan, "start_ts"), require_int(plan, "end_ts")
    require_cohort_window(SERIES, start, end)
    offset = require_int(plan, "decision_offset_seconds")
    lag, age = (
        require_int(plan, "min_quote_age_seconds"),
        require_int(plan, "max_quote_age_seconds"),
    )
    if not (
        require_int(cohort, "schema_version") == 1
        and require_str(cohort, "selection_clock") == "capture_time_close_time"
        and require_int(cohort, "start_ts_inclusive") <= start
        and end <= require_int(cohort, "end_ts_exclusive")
        and 0 < lag <= age <= offset < 900
    ):
        raise ValueError(
            "[IQUOTE-001] require a contained cohort window and 0 < lag <= age <= offset < 900"
        )


def _contracts(cohort: JSONObject, plan: JSONObject) -> tuple[JSONObject, ...]:
    """Validate interval identities without consulting outcomes or traded volume.

    Args:
        cohort: Complete event inventory.
        plan: Validated quote plan.

    Returns:
        All selected native contracts with integer timing fields.

    Raises:
        ValueError: IQUOTE-002 for missing families, duplicate identities or changed intervals.
    """
    found: set[str] = set()
    tickers: set[str] = set()
    slots: set[tuple[str, int]] = set()
    selected: list[JSONObject] = []
    start, end = require_int(plan, "start_ts"), require_int(plan, "end_ts")
    for value in require_list(cohort, "families"):
        family = narrow_json_to_dict(value)
        series = require_str(family, "series")
        if series not in SERIES:
            continue
        if series in found:
            raise ValueError("[IQUOTE-002] duplicate crypto family")
        found.add(series)
        for raw in require_list(family, "markets"):
            market = narrow_json_to_dict(raw)
            close = parse_timestamp(require_str(market, "close_time"))
            if not start * 1000 <= close < end * 1000:
                continue
            opened = parse_timestamp(require_str(market, "open_time"))
            ticker, event = require_str(market, "ticker"), require_str(market, "event_ticker")
            if (
                opened % 900000 != 0
                or close - opened != 900000
                or not event.startswith(series + "-")
                or not ticker.startswith(event + "-")
                or ticker in tickers
                or (series, opened) in slots
            ):
                raise ValueError(
                    "[IQUOTE-002] require unique grid-aligned fifteen-minute contracts"
                )
            tickers.add(ticker)
            slots.add((series, opened))
            selected.append(
                dict(
                    market,
                    series=series,
                    open_ms=opened,
                    decision_ms=opened + require_int(plan, "decision_offset_seconds") * 1000,
                )
            )
    if found != set(SERIES):
        raise ValueError("[IQUOTE-002] both crypto families must be present, including empty ones")
    selected.sort(key=lambda row: (require_int(row, "open_ms"), require_str(row, "ticker")))
    return tuple(selected)


def _attachment(
    market: JSONObject, document: JSONObject, historical: bool, plan: JSONObject
) -> JSONObject:
    """Keep the latest prior candle even when its sides are unusable.

    Args:
        market: Contract with fixed decision time.
        document: Native candle array for this contract.
        historical: Select the documented native price schema.
        plan: Explicit lag and maximum age.

    Returns:
        Temporal eligibility, missingness and the selected quote, without outcomes.

    Raises:
        ValueError: IQUOTE-003 for duplicate or non-minute candle timestamps.
        TypeError: Required candle fields are malformed.
    """
    decision, opened = require_int(market, "decision_ms"), require_int(market, "open_ms")
    lag = require_int(plan, "min_quote_age_seconds") * 1000
    seen: set[int] = set()
    prior: list[Quote] = []
    for value in require_list(document, "candlesticks"):
        quote = decode_candle(narrow_json_to_dict(value), historical)
        stamp = quote["timestamp_ms"]
        if stamp in seen or stamp % 60000 != 0:
            raise ValueError("[IQUOTE-003] duplicate or non-minute candle timestamp")
        seen.add(stamp)
        if opened < stamp <= decision - lag:
            prior.append(quote)
    row: JSONObject = {
        "ticker": require_str(market, "ticker"),
        "event_ticker": require_str(market, "event_ticker"),
        "series": require_str(market, "series"),
        "open_ms": opened,
        "decision_ms": decision,
        "tier": "historical" if historical else "current",
        "eligible_candle_count": len(prior),
        "quote": None,
        "quote_age_ms": None,
        "status": "no_prior_candle",
    }
    if not prior:
        return row
    prior.sort(key=lambda quote: quote["timestamp_ms"])
    latest = prior[-1]
    age = decision - latest["timestamp_ms"]
    bid, ask = latest["bid"], latest["ask"]
    if age > require_int(plan, "max_quote_age_seconds") * 1000:
        status = "stale"
    elif bid is None or ask is None:
        status = "missing_side"
    elif bid > ask:
        status = "crossed"
    elif bid == ask:
        status = "locked"
    else:
        status = "two_sided"
    row.update(quote=encode_quote(latest), quote_age_ms=age, status=status)
    return row


def _current_batch(
    collector: MarketCollector, markets: list[JSONObject], plan: JSONObject
) -> list[JSONObject]:
    """Bound worst-case candles below the endpoint cap and reconcile identities.

    Args:
        collector: Archive-backed transport.
        markets: One three-hour bucket of current-tier intervals.
        plan: Validated fixed rule.

    Returns:
        One attachment for every requested contract, including omitted responses.

    Raises:
        ValueError: IQUOTE-004 for batch capacity, unexpected or duplicate identities.
        HTTPStatusError: Native acquisition fails explicitly.
    """
    start = min(require_int(row, "open_ms") for row in markets) // 1000
    end = max(require_int(row, "decision_ms") for row in markets) // 1000
    if len(markets) > 100 or len(markets) * ((end - start) // 60 + 1) >= 10000:
        raise ValueError("[IQUOTE-004] batch exceeds conservative candle capacity")
    expected = {require_str(row, "ticker") for row in markets}
    response = collector.read(
        "/markets/candlesticks",
        {
            "market_tickers": ",".join(sorted(expected)),
            "start_ts": start,
            "end_ts": end,
            "period_interval": 1,
        },
    )
    documents: dict[str, JSONObject] = {}
    for value in require_list(response, "markets"):
        document = narrow_json_to_dict(value)
        ticker = require_str(document, "market_ticker")
        if ticker not in expected or ticker in documents:
            raise ValueError("[IQUOTE-004] unexpected or duplicate batch contract")
        documents[ticker] = document
    rows: list[JSONObject] = []
    for market in markets:
        ticker = require_str(market, "ticker")
        if ticker in documents:
            rows.append(_attachment(market, documents[ticker], False, plan))
        else:
            missing = _attachment(market, {"candlesticks": []}, False, plan)
            missing["status"] = "omitted_by_endpoint"
            rows.append(missing)
    return rows


def collect_interval_quotes(
    collector: MarketCollector, cohort: JSONObject, plan: JSONObject
) -> JSONObject:
    """Attach minute quotes through independently routed native history tiers.

    Args:
        collector: Existing archive service in capture, resume or replay mode.
        cohort: Source-verified complete event inventory.
        plan: Explicit closed window and quote timing rule.

    Returns:
        Complete attachment panel and availability counts, never a fill claim.

    Raises:
        ValueError: IQUOTE-001 through IQUOTE-005 for invalid plans or evidence.
        TypeError: Required native fields are malformed.
        HTTPStatusError: Cutoff or either quote tier is unavailable.
    """
    require_interval_plan(plan, cohort)
    contracts = _contracts(cohort, plan)
    cutoff = collector.read("/historical/cutoff", {})
    boundary = parse_timestamp(require_str(cutoff, "market_settled_ts"))
    current: dict[int, list[JSONObject]] = {}
    rows: list[JSONObject] = []
    for market in contracts:
        if parse_timestamp(require_str(market, "settlement_ts")) < boundary:
            ticker = require_str(market, "ticker")
            document = collector.read(
                "/historical/markets/" + ticker + "/candlesticks",
                {
                    "start_ts": require_int(market, "open_ms") // 1000,
                    "end_ts": require_int(market, "decision_ms") // 1000,
                    "period_interval": 1,
                },
            )
            if require_str(document, "ticker") != ticker:
                raise ValueError("[IQUOTE-005] historical candle contract mismatch")
            rows.append(_attachment(market, document, True, plan))
        else:
            bucket = require_int(market, "open_ms") // 10800000
            current.setdefault(bucket, []).append(market)
    for markets in current.values():
        rows.extend(_current_batch(collector, markets, plan))
    rows.sort(key=lambda row: (require_int(row, "open_ms"), require_str(row, "ticker")))
    counts = Counter(require_str(row, "status") for row in rows)
    count_document: JSONObject = dict(sorted(counts.items()))
    encoded: list[JSONValue] = list(rows)
    return {
        "schema_version": 1,
        "plan": plan,
        "cutoff": cutoff,
        "contract_count": len(rows),
        "status_counts": count_document,
        "contracts": encoded,
        "orders_submitted": 0,
        "interpretation": (
            "Retrospective minute closes, not contemporaneous receipt or executable depth. "
            "Open times are capture-time metadata; decision offsets are fixed without outcomes. "
            "No missing side is filled from an older candle. "
            "Outcomes and final volumes are not features. Shared intervals and serial dependence "
            "remain; quote availability does not prove an edge."
        ),
    }


__all__ = ["collect_interval_quotes", "require_interval_plan"]
