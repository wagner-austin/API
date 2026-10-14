"""Explicit split-adjusted and distribution-adjusted daily price evidence."""

from __future__ import annotations

from math import isfinite

from platform_core.json_utils import JSONObject, require_float
from typing_extensions import ReadOnly, TypedDict

from .validation import require_bounded_int, require_nonempty


class EquityPrice(TypedDict):
    """One historical daily bar; its timestamp is session start, not close."""

    symbol: ReadOnly[str]
    currency: ReadOnly[str]
    session_start_s: ReadOnly[int]
    split_adjusted_close: ReadOnly[float]
    distribution_adjusted_close: ReadOnly[float]


def encode_equity_price(record: EquityPrice) -> JSONObject:
    """Encode a validated daily observation with explicit adjustment semantics.

    Args:
        record: Immutable equity observation.

    Returns:
        JSON record preserving both provider price conventions.
    """
    return {
        "symbol": record["symbol"],
        "currency": record["currency"],
        "session_start_s": record["session_start_s"],
        "split_adjusted_close": record["split_adjusted_close"],
        "distribution_adjusted_close": record["distribution_adjusted_close"],
    }


def decode_equity_price(document: JSONObject) -> EquityPrice:
    """Require finite positive prices without replacing null observations.

    Args:
        document: Untrusted normalized daily observation.

    Returns:
        Immutable observation with epoch-second session start.

    Raises:
        ValueError: EQUITY-001 for nonpositive or nonfinite prices; shared
            validators reject invalid identity and timestamp values.
        TypeError: Missing or incorrectly typed required fields.
    """
    close = require_float(document, "split_adjusted_close")
    adjusted = require_float(document, "distribution_adjusted_close")
    if not all(isfinite(value) and value > 0 for value in (close, adjusted)):
        raise ValueError("[EQUITY-001] both adjusted prices must be finite and positive")
    return EquityPrice(
        symbol=require_nonempty(document, "symbol"),
        currency=require_nonempty(document, "currency"),
        session_start_s=require_bounded_int(document, "session_start_s", 0, 253402214399),
        split_adjusted_close=close,
        distribution_adjusted_close=adjusted,
    )
