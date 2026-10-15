"""Pure native-field validation for public tape collection and offline audits."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_str

from .kalshi import decode_public_trade
from .types import Trade
from .validation import parse_fixed, require_nonempty


def decode_tape_trade(document: JSONObject, query_start: int, query_end: int) -> Trade:
    """Validate native execution fields against the requested query envelope.

    Args:
        document: Native public execution object.
        query_start: Inclusive lower Unix-second envelope boundary.
        query_end: Inclusive upper Unix-second envelope boundary.

    Returns:
        Existing immutable Trade with validated prices, volume and timestamp.

    Raises:
        ValueError: TAPE-002 for an out-of-envelope timestamp, invalid book side
            or noncomplementary YES and NO prices; native codec failures propagate.
        TypeError: Required fields have incorrect native types.
    """
    trade = decode_public_trade(document, require_nonempty(document, "ticker"))
    if not query_start * 1000 <= trade["timestamp_ms"] <= query_end * 1000 or require_str(
        document, "taker_book_side"
    ) not in ("bid", "ask"):
        raise ValueError("[TAPE-002] trade outside requested window or invalid book side")
    no_price = parse_fixed(require_str(document, "no_price_dollars"), 4)
    if trade["price"] + no_price != 10000:
        raise ValueError("[TAPE-002] YES and NO trade prices are not complementary")
    return trade


def require_market_snapshot(document: JSONObject) -> str:
    """Require metadata identity, status and internally consistent declared payouts.

    Args:
        document: Native market metadata received after the execution window.

    Returns:
        Nonempty contract identity after all required fields validate.

    Raises:
        ValueError: TAPE-005 for unsupported or inconsistent result fields.
        TypeError: A required field has an incorrect native type.
    """
    ticker = require_nonempty(document, "ticker")
    require_nonempty(document, "event_ticker")
    require_nonempty(document, "status")
    result = require_str(document, "result")
    if result not in ("", "yes", "no", "scalar"):
        raise ValueError("[TAPE-005] unsupported native market result")
    if result:
        payout = parse_fixed(require_str(document, "settlement_value_dollars"), 4)
        if (
            not 0 <= payout <= 10000
            or (result == "yes" and payout != 10000)
            or (result == "no" and payout != 0)
        ):
            raise ValueError("[TAPE-005] result disagrees with cash settlement")
    return ticker


__all__ = ["decode_tape_trade", "require_market_snapshot"]
