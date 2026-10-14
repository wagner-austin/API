"""Exact-cutoff quote validation shared across declared market experiments."""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject, narrow_json_to_dict, require_list, require_str

from .kalshi import MarketSpec, decode_candle
from .types import DOLLAR, Quote


def decision_quote(
    spec: MarketSpec,
    read: Callable[[str, dict[str, str | int]], JSONObject],
    *,
    historical: bool,
) -> tuple[Quote | None, str]:
    """Request one exact decision candle without manufacturing an executable fill.

    Args:
        spec: Validated selected contract.
        read: Explicit archive capture or replay method.
        historical: Required native partition selected from captured inventory membership.

    Returns:
        Eligible quote and status, or an explicit exclusion reason.

    Raises:
        ValueError: STRATEGY-003 for wrong ticker, duplicate or off-cutoff candles.
        HTTPStatusError: Acquisition failed.
    """
    cutoff = spec["decision_ms"] // 1000
    prefix = "/historical/markets/" if historical else "/series/" + spec["series"] + "/markets/"
    document = read(
        prefix + spec["ticker"] + "/candlesticks",
        {"start_ts": cutoff, "end_ts": cutoff, "period_interval": 1},
    )
    candles = require_list(document, "candlesticks")
    if require_str(document, "ticker") != spec["ticker"] or len(candles) > 1:
        raise ValueError("[STRATEGY-003] candle identity or count mismatch")
    if not candles:
        return None, "missing_candle"
    quote = decode_candle(narrow_json_to_dict(candles[0]), historical=historical)
    if quote["timestamp_ms"] != spec["decision_ms"]:
        raise ValueError("[STRATEGY-003] candle does not end at the declared cutoff")
    bid, ask = quote["bid"], quote["ask"]
    if bid is not None and ask is not None and bid > ask:
        raise ValueError("[STRATEGY-003] crossed historical quote")
    if bid is None or ask is None or bid == 0 or ask == DOLLAR:
        return None, "unavailable_quote"
    if ask - bid > 500:
        return None, "wide_spread"
    return quote, "eligible"
