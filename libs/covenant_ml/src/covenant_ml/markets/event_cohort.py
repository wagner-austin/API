"""Outcome inventory independent of executions, prices and eventual winning side."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONValue, require_str

from .collection import MarketCollector
from .public_tape_records import require_market_snapshot
from .validation import parse_fixed, parse_timestamp, require_nonempty


def _reconcile_snapshot(first: JSONObject, second: JSONObject) -> JSONObject:
    """Require exact agreement except for explicitly recorded mutable open interest.

    Args:
        first: Current-tier native snapshot.
        second: Historical-tier native snapshot of the same contract.

    Returns:
        Both open-interest observations, without treating either as historical input.

    Raises:
        ValueError: COHORT-003 for a difference in any other native field.
        TypeError: A differing open-interest observation is missing or malformed.
    """
    left = {key: value for key, value in first.items() if key != "open_interest_fp"}
    right = {key: value for key, value in second.items() if key != "open_interest_fp"}
    if left != right:
        raise ValueError("[COHORT-003] conflicting cross-tier market snapshot")
    values = [require_str(row, "open_interest_fp") for row in (first, second)]
    for value in values:
        parse_fixed(value, 2)
    return {
        "ticker": require_str(first, "ticker"),
        "current_open_interest_fp": values[0],
        "historical_open_interest_fp": values[1],
    }


def require_cohort_window(series: tuple[str, ...], start: int, end: int) -> None:
    """Validate a bounded, explicitly named acquisition before requesting evidence.

    Args:
        series: Unique exchange series identifiers.
        start: Inclusive close-time boundary in Unix seconds.
        end: Exclusive close-time boundary in Unix seconds.

    Raises:
        ValueError: COHORT-001 for invalid families or a window exceeding 366 days.
    """
    if (
        not series
        or len(set(series)) != len(series)
        or any(not name.isalnum() or not name.isascii() for name in series)
        or not 0 <= start < end <= start + 366 * 86400
    ):
        raise ValueError("[COHORT-001] require unique alphanumeric series and a bounded window")


def _family_inventory(collector: MarketCollector, series: str, start: int, end: int) -> JSONObject:
    """Reconcile both native inventories, preserving all selected source documents.

    Args:
        collector: Explicit capture or replay reader with shared pagination.
        series: One validated exchange series identifier.
        start: Inclusive close-time boundary in Unix seconds.
        end: Exclusive close-time boundary in Unix seconds.

    Returns:
        Family inventory with native records and separate exclusion counts by tier.

    Raises:
        ValueError: COHORT-002 for mismatched or repeated identities; COHORT-003
            for conflicting overlap; COHORT-004 for invalid open/close chronology.
        TypeError: Required source fields are missing or malformed.
        HTTPStatusError: Acquisition failed; no partial family report is returned.
    """
    selected: dict[str, JSONObject] = {}
    tiers: list[JSONValue] = []
    overlap = 0
    differences: list[JSONValue] = []
    for path in ("/markets", "/historical/markets"):
        params: dict[str, str | int] = {"series_ticker": series}
        if path == "/markets":
            params.update(min_close_ts=start - 1, max_close_ts=end)
        seen: set[str] = set()
        outside = 0
        for document in collector.pages(path, params, "markets"):
            ticker = require_nonempty(document, "ticker")
            event = require_nonempty(document, "event_ticker")
            if (
                not ticker.startswith(series + "-")
                or not event.startswith(series + "-")
                or not ticker.startswith(event + "-")
                or ticker in seen
            ):
                raise ValueError("[COHORT-002] unexpected family, event or duplicate market")
            seen.add(ticker)
            close_ms = parse_timestamp(require_str(document, "close_time"))
            if not start * 1000 <= close_ms < end * 1000:
                outside += 1
                continue
            require_market_snapshot(document)
            opened_ms = parse_timestamp(require_str(document, "open_time"))
            if not 0 <= opened_ms < close_ms:
                raise ValueError("[COHORT-004] market must open before its recorded close")
            parse_fixed(require_str(document, "volume_fp"), 2)
            if ticker in selected:
                if selected[ticker] != document:
                    differences.append(_reconcile_snapshot(selected[ticker], document))
                else:
                    overlap += 1
            else:
                selected[ticker] = document
        tiers.append({"path": path, "rows_received": len(seen), "outside_window": outside})
    markets = [selected[ticker] for ticker in sorted(selected)]
    events = sorted({require_str(row, "event_ticker") for row in markets})
    result_counts: dict[str, JSONValue] = {
        result: sum(require_str(row, "result") == result for row in markets)
        for result in ("", "yes", "no", "scalar")
    }
    zero_volume = sum(parse_fixed(require_str(row, "volume_fp"), 2) == 0 for row in markets)
    return {
        "series": series,
        "tiers": tiers,
        "identical_cross_tier_overlap": overlap,
        "open_interest_differences": differences,
        "contract_count": len(markets),
        "event_count": len(events),
        "events": list(events),
        "zero_volume_contract_count": zero_volume,
        "declared_result_counts": result_counts,
        "markets": list(markets),
    }


def collect_event_cohort(
    collector: MarketCollector, series: tuple[str, ...], start: int, end: int
) -> JSONObject:
    """Collect complete endpoint inventories without selecting on volume or result.

    Args:
        collector: Existing archive-backed capture or offline replay service.
        series: Unique family identifiers, fixed before acquisition.
        start: Inclusive close-time boundary in Unix seconds.
        end: Exclusive close-time boundary in Unix seconds.

    Returns:
        Native outcome inventory; metadata is explicitly capture-time evidence.

    Raises:
        ValueError: COHORT-001 through COHORT-004 for invalid plans or evidence.
        TypeError: Native response schema is malformed.
        HTTPStatusError: Either tier fails; acquisition errors propagate.
    """
    require_cohort_window(series, start, end)
    families: list[JSONValue] = [
        _family_inventory(collector, name, start, end) for name in sorted(series)
    ]
    return {
        "schema_version": 1,
        "start_ts_inclusive": start,
        "end_ts_exclusive": end,
        "selection_clock": "capture_time_close_time",
        "families": families,
        "orders_submitted": 0,
        "interpretation": (
            "Endpoint-complete inventory, not a historical availability census. "
            "Close times, volumes, rules and results are capture-time snapshots. "
            "Early-close timestamps must not set retrospective prediction times. "
            "Event IDs and adjacent intervals are not independent observations. "
            "Missing quotes, features, fees, depth and receipt latency are not imputed. "
            "No predictive skill, executable fills or profitability is established."
        ),
    }


__all__ = ["collect_event_cohort", "require_cohort_window"]
