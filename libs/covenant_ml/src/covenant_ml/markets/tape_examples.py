"""Recount a public tape and select examples without inspecting their returns."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)

from .public_tape_records import decode_tape_trade, require_market_snapshot
from .validation import require_nonempty


def summarize_tape(report: JSONObject) -> JSONObject:
    """Validate counts and retain one deterministic native example per namespace.

    Args:
        report: Source-verified public tape report, including native metadata.

    Returns:
        Exact counts and the lexicographically smallest trade ID per ticker
        namespace. Namespaces are not independently verified asset categories.
        Outcomes and metadata remain explicitly capture-time observations.

    Raises:
        ValueError: TAPE-008 for duplicate identities, missing or extra metadata,
            inconsistent counts, or a trade outside the selected window.
        TypeError: Required native fields or combination legs are malformed.
    """
    start = require_int(report, "start_ts_inclusive")
    end = require_int(report, "end_ts_exclusive")
    if not 0 <= start < end <= start + 86400:
        raise ValueError("[TAPE-008] invalid public tape window")
    markets: dict[str, JSONObject] = {}
    events: set[str] = set()
    legs: set[str] = set()
    combinations = 0
    for value in require_list(report, "markets"):
        market = narrow_json_to_dict(value)
        ticker = require_market_snapshot(market)
        if ticker in markets:
            raise ValueError("[TAPE-008] duplicate metadata identity")
        markets[ticker] = market
        events.add(require_nonempty(market, "event_ticker"))
        selected = market.get("mve_selected_legs")
        if selected is not None:
            leg_rows = require_list(market, "mve_selected_legs")
            combinations += int(bool(leg_rows))
            for leg in leg_rows:
                legs.add(require_nonempty(narrow_json_to_dict(leg), "event_ticker"))
    ids: set[str] = set()
    tickers: set[str] = set()
    counts: dict[str, int] = {}
    contracts: dict[str, set[str]] = {}
    examples: dict[str, JSONObject] = {}
    quantity = 0
    blocks = 0
    for value in require_list(report, "trades"):
        row = narrow_json_to_dict(value)
        trade = decode_tape_trade(row, start, end)
        identity = require_nonempty(row, "trade_id")
        ticker = require_nonempty(row, "ticker")
        if identity in ids or ticker not in markets or trade["timestamp_ms"] >= end * 1000:
            raise ValueError("[TAPE-008] duplicate trade, missing metadata or excluded boundary")
        ids.add(identity)
        tickers.add(ticker)
        quantity += trade["quantity"]
        blocks += int(trade["block"])
        namespace = ticker.split("-", 1)[0]
        counts[namespace] = counts.get(namespace, 0) + 1
        contracts.setdefault(namespace, set()).add(ticker)
        if namespace not in examples or identity < require_str(examples[namespace], "trade_id"):
            examples[namespace] = row
    measured = {
        "trade_count": len(ids),
        "contract_count": len(tickers),
        "event_count": len(events),
        "quantity_centi_contracts": quantity,
        "block_trade_count": blocks,
    }
    if tickers != markets.keys() or any(
        require_int(report, name) != count for name, count in measured.items()
    ):
        raise ValueError(
            "[TAPE-008] metadata coverage or reported counts disagree with native rows"
        )
    rows: list[JSONValue] = [
        {
            "namespace": namespace,
            "trade_count": counts[namespace],
            "contract_count": len(contracts[namespace]),
            "example_trade": examples[namespace],
            "example_market_at_capture": markets[require_str(examples[namespace], "ticker")],
        }
        for namespace in sorted(counts)
    ]
    return {
        "start_ts_inclusive": start,
        "end_ts_exclusive": end,
        **measured,
        "contracts_with_combination_legs": combinations,
        "distinct_declared_leg_events": len(legs),
        "namespaces": rows,
        "example_selection": "Smallest trade ID per ticker namespace; no return selection.",
        "interpretation": (
            "Metadata and outcomes are capture-time observations, not historical features. "
            "Event and leg counts do not establish independent observations. "
            "Namespaces are identifiers, not verified asset categories. No profit is estimated."
        ),
    }


__all__ = ["summarize_tape"]
