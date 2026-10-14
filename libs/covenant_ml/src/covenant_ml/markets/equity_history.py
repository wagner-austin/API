"""Strict daily chart decoding and future close-to-close outcome measurement."""

from __future__ import annotations

from itertools import pairwise
from math import isfinite

from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_dict,
    require_dict,
    require_list,
    require_str,
)

from .equity_records import EquityPrice, decode_equity_price
from .validation import parse_timestamp, require_bounded_int, require_nonempty


def decode_equity_chart(
    document: JSONObject, symbol: str, currency: str, *, received_at: str
) -> tuple[EquityPrice, ...]:
    """Decode one archived daily equity response with no price substitution.

    Args:
        document: Native Yahoo chart object, including explicit null error.
        symbol: Exact expected ticker; historical issuer mapping is caller-owned.
        currency: Exact expected quote currency; no conversion is performed.
        received_at: Aware archive receipt time. Bars less than 24 hours old
            are refused to avoid interpreting an unfinished bar as a close.

    Returns:
        Nonempty, strictly ordered daily observations from aligned price arrays.
        Raw corporate actions remain in the source archive; these prices
        already incorporate adjustments and must not be adjusted again.

    Raises:
        ValueError: EQUITY-002 for provider errors or ambiguous result arrays;
            EQUITY-003 for identity, interval or instrument mismatch;
            EQUITY-004 for empty, unequal, unordered or unfinished history.
        TypeError: Missing or malformed native fields, including null prices.
    """
    chart = require_dict(document, "chart")
    if "error" not in chart or chart["error"] is not None:
        raise ValueError("[EQUITY-002] chart must explicitly report no provider error")
    results = require_list(chart, "result")
    if len(results) != 1:
        raise ValueError("[EQUITY-002] chart must contain exactly one result")
    result = narrow_json_to_dict(results[0])
    meta = require_dict(result, "meta")
    expected = (
        require_nonempty({"symbol": symbol}, "symbol"),
        require_nonempty({"currency": currency}, "currency"),
        "EQUITY",
        "1d",
    )
    observed = tuple(
        require_str(meta, key)
        for key in ("symbol", "currency", "instrumentType", "dataGranularity")
    )
    if observed != expected:
        raise ValueError("[EQUITY-003] chart identity, instrument or daily interval differs")
    indicators = require_dict(result, "indicators")
    quotes, adjusted = require_list(indicators, "quote"), require_list(indicators, "adjclose")
    if len(quotes) != 1 or len(adjusted) != 1:
        raise ValueError("[EQUITY-002] require exactly one quote and adjusted-price series")
    timestamps = require_list(result, "timestamp")
    closes = require_list(narrow_json_to_dict(quotes[0]), "close")
    adjusted_closes = require_list(narrow_json_to_dict(adjusted[0]), "adjclose")
    if not timestamps or len({len(timestamps), len(closes), len(adjusted_closes)}) != 1:
        raise ValueError("[EQUITY-004] daily price arrays must be nonempty and aligned")
    received_ms = parse_timestamp(received_at)
    records: list[EquityPrice] = []
    previous = -1
    for index, timestamp in enumerate(timestamps):
        record = decode_equity_price(
            {
                "symbol": symbol,
                "currency": currency,
                "session_start_s": timestamp,
                "split_adjusted_close": closes[index],
                "distribution_adjusted_close": adjusted_closes[index],
            }
        )
        start = record["session_start_s"]
        if start <= previous or (start + 86400) * 1000 > received_ms:
            raise ValueError("[EQUITY-004] history must be ordered, unique and completed")
        records.append(record)
        previous = start
    return tuple(records)


def adjusted_forward_return(
    prices: tuple[EquityPrice, ...], *, available_at: str, holding_bars: int
) -> tuple[EquityPrice, EquityPrice, float] | None:
    """Measure an adjusted close return starting after information availability.

    Args:
        prices: Validated observations for one symbol and currency, in order.
        available_at: Aware decision or disclosure availability timestamp.
        holding_bars: Positive number of subsequent observed daily bars to hold.
            This does not certify that the exchange calendar has no missing days.

    Returns:
        Entry, exit and adjusted price ratio minus one, or None when the
        archived window cannot supply either endpoint. Entry uses the first
        bar whose session starts strictly after availability, at that bar's
        close. The return is a provider-adjusted research outcome, not an
        executable fill or fee-inclusive trading profit.

    Raises:
        ValueError: EQUITY-005 for mixed identities or unordered history;
            EQUITY-006 for an unrepresentable return; shared validators reject
            invalid holding intervals or availability timestamps.
    """
    horizon = require_bounded_int({"holding_bars": holding_bars}, "holding_bars", 1, 1000000)
    available_ms = parse_timestamp(available_at)
    identities = {(row["symbol"], row["currency"]) for row in prices}
    starts = tuple(row["session_start_s"] for row in prices)
    if len(identities) > 1 or any(left >= right for left, right in pairwise(starts)):
        raise ValueError("[EQUITY-005] returns require one identity and strictly ordered bars")
    for index, entry in enumerate(prices):
        if entry["session_start_s"] * 1000 <= available_ms:
            continue
        if index + horizon >= len(prices):
            return None
        exit_price = prices[index + horizon]
        ratio = exit_price["distribution_adjusted_close"] / entry["distribution_adjusted_close"]
        if not isfinite(ratio) or ratio <= 0:
            raise ValueError("[EQUITY-006] adjusted return ratio is not finite and positive")
        return entry, exit_price, ratio - 1
    return None
