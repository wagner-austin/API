"""Typed serialization of simulated executions, with accounting invariants."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_int, require_str

from .codecs import MAX_INTEGER
from .types import CONTRACT, DOLLAR, ReplayResult, ReplayStatus
from .validation import require_bounded_int, require_day, require_nonempty, require_optional_price


def encode_result(record: ReplayResult) -> JSONObject:
    """Encode an execution result without changing its monetary precision.

    Args:
        record: Validated result.

    Returns:
        JSON object.
    """
    return {
        "ticker": record["ticker"],
        "series": record["series"],
        "event_day": record["event_day"],
        "settlement": record["settlement"],
        "status": record["status"].value,
        "offer": record["offer"],
        "matching_quantity": record["matching_quantity"],
        "contracts": record["contracts"],
        "fee": record["fee"],
        "pnl": record["pnl"],
        "capital": record["capital"],
    }


def decode_result(document: JSONObject) -> ReplayResult:
    """Validate a result, including the relationship between fills and money.

    Args:
        document: JSON object.

    Returns:
        Immutable execution result.

    Raises:
        ValueError: MARKET-012 for impossible accounting or an invalid enum.
        TypeError: Missing or incorrectly typed fields.
    """
    status = ReplayStatus(require_str(document, "status"))
    offer = require_optional_price(document, "offer")
    quantity = require_bounded_int(document, "matching_quantity", 0, MAX_INTEGER)
    contracts = require_bounded_int(document, "contracts", 0, MAX_INTEGER)
    fee = require_bounded_int(document, "fee", 0, MAX_INTEGER)
    pnl = require_int(document, "pnl")
    capital = require_bounded_int(document, "capital", 0, MAX_INTEGER)
    settlement = require_bounded_int(document, "settlement", 0, DOLLAR)
    if status is ReplayStatus.FILLED:
        if offer is None or contracts == 0 or quantity < contracts * CONTRACT:
            raise ValueError(
                "[MARKET-012] filled result requires an offer, size and matching volume"
            )
        if capital != (DOLLAR - offer) * contracts:
            raise ValueError("[MARKET-012] collateral does not match offer and size")
        if pnl != (offer - settlement) * contracts - fee:
            raise ValueError("[MARKET-012] payoff does not match recorded settlement")
    elif contracts != 0 or fee != 0 or pnl != 0 or capital != 0:
        raise ValueError("[MARKET-012] unfilled results cannot carry positions, fees or payoff")
    return ReplayResult(
        ticker=require_nonempty(document, "ticker"),
        series=require_nonempty(document, "series"),
        event_day=require_day(document, "event_day"),
        settlement=settlement,
        status=status,
        offer=offer,
        matching_quantity=quantity,
        contracts=contracts,
        fee=fee,
        pnl=pnl,
        capital=capital,
    )


__all__ = ["decode_result", "encode_result"]
