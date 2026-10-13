"""Strict finalized-settlement states and immutable saved probability records."""

from __future__ import annotations

from enum import StrEnum

from platform_core.json_utils import JSONObject, require_bool, require_float, require_list
from typing_extensions import ReadOnly, TypedDict

from .forecast_records import CITY_COORDINATES
from .kalshi import decode_market
from .types import DOLLAR
from .validation import require_bounded_int, require_day, require_nonempty, require_optional_price

ARMS = ("raw_market", "raw_forecast", "market_calibrated", "market_plus_forecast")
PENDING_STATUSES = (
    "initialized",
    "active",
    "inactive",
    "closed",
    "determined",
    "disputed",
    "amended",
)


class SettlementState(StrEnum):
    """Explicit scoring disposition, distinct from the exchange's lifecycle state."""

    PENDING = "pending"
    BINARY = "binary"
    SCALAR = "scalar"


class SavedPrediction(TypedDict):
    """Frozen probabilities and identities required for settlement reconciliation."""

    ticker: ReadOnly[str]
    series: ReadOnly[str]
    event_day: ReadOnly[str]
    recorded_ms: ReadOnly[int]
    rules_primary: ReadOnly[str]
    probabilities: ReadOnly[tuple[float, ...]]


class Settlement(TypedDict):
    """Native payout and receipt chronology; pending payouts are always explicit null."""

    ticker: ReadOnly[str]
    state: ReadOnly[SettlementState]
    native_status: ReadOnly[str]
    received_ms: ReadOnly[int]
    settled_ms: ReadOnly[int | None]
    payout: ReadOnly[int | None]
    rules_changed: ReadOnly[bool]


def encode_saved_prediction(row: SavedPrediction) -> JSONObject:
    """Encode frozen prediction identity and all registered arm probabilities.

    Args:
        row: Validated saved prediction.

    Returns:
        JSON record.
    """
    return {
        "ticker": row["ticker"],
        "series": row["series"],
        "event_day": row["event_day"],
        "recorded_ms": row["recorded_ms"],
        "rules_primary": row["rules_primary"],
        "probabilities": list(row["probabilities"]),
    }


def decode_saved_prediction(document: JSONObject) -> SavedPrediction:
    """Require supported identities and four finite probabilities in the unit interval.

    Args:
        document: Decoded frozen prediction record.

    Returns:
        Immutable validated prediction.

    Raises:
        ValueError: SETTLEMENT-001 for identity or probability contract violations.
        TypeError: Required fields have incorrect types.
    """
    series, ticker = require_nonempty(document, "series"), require_nonempty(document, "ticker")
    probabilities = tuple(
        require_float({"p": value}, "p") for value in require_list(document, "probabilities")
    )
    if (
        series not in CITY_COORDINATES
        or not ticker.startswith(series + "-")
        or len(probabilities) != len(ARMS)
        or any(not 0 <= p <= 1 for p in probabilities)
    ):
        raise ValueError("[SETTLEMENT-001] invalid prediction identity or probabilities")
    return SavedPrediction(
        ticker=ticker,
        series=series,
        event_day=require_day(document, "event_day"),
        recorded_ms=require_bounded_int(document, "recorded_ms", 0, 2**63 - 1),
        rules_primary=require_nonempty(document, "rules_primary"),
        probabilities=probabilities,
    )


def encode_settlement(row: Settlement) -> JSONObject:
    """Encode explicit pending or final payout evidence.

    Args:
        row: Validated settlement disposition.

    Returns:
        JSON record with required nullable fields.
    """
    return {
        "ticker": row["ticker"],
        "state": row["state"].value,
        "native_status": row["native_status"],
        "received_ms": row["received_ms"],
        "settled_ms": row["settled_ms"],
        "payout": row["payout"],
        "rules_changed": row["rules_changed"],
    }


def decode_settlement(document: JSONObject) -> Settlement:
    """Enforce lifecycle, payout and receipt consistency for all three states.

    Args:
        document: Decoded settlement record.

    Returns:
        Immutable settlement evidence.

    Raises:
        ValueError: SETTLEMENT-002 for inconsistent lifecycle or chronology.
        TypeError: Required fields are malformed.
        KeyError: The required nullable settlement timestamp is absent.
    """
    state = SettlementState(require_nonempty(document, "state"))
    status = require_nonempty(document, "native_status")
    received = require_bounded_int(document, "received_ms", 0, 2**63 - 1)
    payout = require_optional_price(document, "payout")
    settled = (
        None
        if document["settled_ms"] is None
        else require_bounded_int(document, "settled_ms", 0, 2**63 - 1)
    )
    if state is SettlementState.PENDING:
        valid = status in PENDING_STATUSES and payout is None and settled is None
    else:
        valid = (
            status == "finalized"
            and payout is not None
            and settled is not None
            and settled <= received
            and (state is SettlementState.SCALAR or payout in (0, DOLLAR))
        )
    if not valid:
        raise ValueError("[SETTLEMENT-002] inconsistent lifecycle, payout or receipt chronology")
    return Settlement(
        ticker=require_nonempty(document, "ticker"),
        state=state,
        native_status=status,
        received_ms=received,
        settled_ms=settled,
        payout=payout,
        rules_changed=require_bool(document, "rules_changed"),
    )


def decode_native_settlement(
    document: JSONObject, row: SavedPrediction, received: int
) -> Settlement:
    """Use only finalized native payouts, preserving provisional states without labels.

    Args:
        document: Native market object.
        row: Frozen pre-outcome prediction.
        received: Actual response receipt in epoch milliseconds.

    Returns:
        Validated settlement or explicit pending state.

    Raises:
        ValueError: SETTLEMENT-003 for identity, unknown status or prediction lookahead.
        TypeError: Native response fields have incorrect types.
    """
    ticker, status = require_nonempty(document, "ticker"), require_nonempty(document, "status")
    if ticker != row["ticker"] or received < row["recorded_ms"]:
        raise ValueError("[SETTLEMENT-003] response identity or receipt predates prediction")
    payout: int | None = None
    settled: int | None = None
    state = SettlementState.PENDING
    if status == "finalized":
        spec = decode_market(document, row["series"], "America/New_York")
        if spec["day"] != row["event_day"] or spec["settled_ms"] <= row["recorded_ms"]:
            raise ValueError(
                "[SETTLEMENT-003] wrong event date or prediction recorded after settlement"
            )
        payout, settled = spec["settlement"], spec["settled_ms"]
        state = SettlementState.SCALAR if document["result"] == "scalar" else SettlementState.BINARY
    elif status not in PENDING_STATUSES:
        raise ValueError("[SETTLEMENT-003] unknown exchange lifecycle status")
    return decode_settlement(
        {
            "ticker": ticker,
            "state": state.value,
            "native_status": status,
            "received_ms": received,
            "settled_ms": settled,
            "payout": payout,
            "rules_changed": require_nonempty(document, "rules_primary") != row["rules_primary"],
        }
    )


__all__ = [
    "ARMS",
    "PENDING_STATUSES",
    "SavedPrediction",
    "Settlement",
    "SettlementState",
    "decode_native_settlement",
    "decode_saved_prediction",
    "decode_settlement",
    "encode_saved_prediction",
    "encode_settlement",
]
