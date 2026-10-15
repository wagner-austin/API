"""Strict immutable records for observed quotes and local submission timing."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

from .campaign_records import require_digest
from .execution_records import OrderIntent, decode_intent, encode_intent
from .paper_records import decode_native_book
from .validation import parse_timestamp, require_bounded_int, require_nonempty


class SubmissionCapture(TypedDict):
    """Native response contents and local receipt times preceding a reserved send."""

    intent: ReadOnly[OrderIntent]
    market_json: ReadOnly[str]
    book_json: ReadOnly[str]
    preflight_completed_ms: ReadOnly[int]
    market_received_ms: ReadOnly[int]
    book_received_ms: ReadOnly[int]
    decision_ms: ReadOnly[int]
    started_ms: ReadOnly[int]


class SubmissionTiming(TypedDict):
    """Local workflow interval bound to the saved native acknowledgement digest."""

    client_order_id: ReadOnly[str]
    started_ms: ReadOnly[int]
    returned_ms: ReadOnly[int]
    capture_sha256: ReadOnly[str]
    acknowledgement_sha256: ReadOnly[str]


def encode_submission_capture(record: SubmissionCapture) -> JSONObject:
    """Encode the exact parsed native responses and local timestamps.

    Args:
        record: Validated immutable submission capture.

    Returns:
        JSON object with native response objects, including supplied contract rules.
    """
    return {
        "intent": encode_intent(record["intent"]),
        "market": narrow_json_to_dict(load_json_str(record["market_json"])),
        "book": narrow_json_to_dict(load_json_str(record["book_json"])),
        "preflight_completed_ms": record["preflight_completed_ms"],
        "market_received_ms": record["market_received_ms"],
        "book_received_ms": record["book_received_ms"],
        "decision_ms": record["decision_ms"],
        "started_ms": record["started_ms"],
    }


def decode_submission_capture(document: JSONObject) -> SubmissionCapture:
    """Validate executable liquidity, identity and chronological receipt evidence.

    Args:
        document: Parsed capture JSON object.

    Returns:
        Immutable record retaining full native responses as serialized JSON.

    Raises:
        ValueError: DEMO-TRIAL-006 for unavailable liquidity; DEMO-EVIDENCE-001
            for invalid chronology or an expired submission stage.
        TypeError: A required field is missing or malformed.
    """
    intent = decode_intent(require_dict(document, "intent"))
    market, native_book = require_dict(document, "market"), require_dict(document, "book")
    book = decode_native_book(native_book)
    times = tuple(
        require_bounded_int(document, key, 0, 2**63 - 1)
        for key in (
            "preflight_completed_ms",
            "market_received_ms",
            "book_received_ms",
            "decision_ms",
            "started_ms",
        )
    )
    preflight, market_received, book_received, decision, started = times
    if (
        times != tuple(sorted(times))
        or started - preflight > 5000
        or not intent["created_ms"] <= started < intent["expires_ms"]
    ):
        raise ValueError("[DEMO-EVIDENCE-001] require ordered fresh receipts and unexpired intent")
    if (
        require_str(market, "ticker") != intent["ticker"]
        or require_str(market, "status") != "active"
        or parse_timestamp(require_str(market, "close_time")) <= started
        or book is None
        or book["ask"] > intent["limit_price"]
        or book["ask_size"] < intent["contracts"] * 100
    ):
        raise ValueError("[DEMO-TRIAL-006] require active market and fresh executable depth")
    return SubmissionCapture(
        intent=intent,
        market_json=dump_json_str(market),
        book_json=dump_json_str(native_book),
        preflight_completed_ms=preflight,
        market_received_ms=market_received,
        book_received_ms=book_received,
        decision_ms=decision,
        started_ms=started,
    )


def encode_submission_timing(record: SubmissionTiming) -> JSONObject:
    """Encode local timing without implying a matching-engine timestamp.

    Args:
        record: Validated immutable timing record.

    Returns:
        JSON object bound to the acknowledgement file's exact byte digest.
    """
    return {
        "client_order_id": record["client_order_id"],
        "started_ms": record["started_ms"],
        "returned_ms": record["returned_ms"],
        "capture_sha256": record["capture_sha256"],
        "acknowledgement_sha256": record["acknowledgement_sha256"],
    }


def decode_submission_timing(document: JSONObject) -> SubmissionTiming:
    """Require ordered local timestamps and an exact acknowledgement identity.

    Args:
        document: Parsed timing JSON object.

    Returns:
        Validated timing record. The interval includes persistence and parsing.

    Raises:
        ValueError: CAMPAIGN-001 for a malformed digest; MARKET-001 for
            reversed or invalid timestamps.
        TypeError: A required field is missing or malformed.
    """
    started = require_bounded_int(document, "started_ms", 0, 2**63 - 1)
    returned = require_bounded_int(document, "returned_ms", started, 2**63 - 1)
    return SubmissionTiming(
        client_order_id=require_nonempty(document, "client_order_id"),
        started_ms=started,
        returned_ms=returned,
        capture_sha256=require_digest(document, "capture_sha256"),
        acknowledgement_sha256=require_digest(document, "acknowledgement_sha256"),
    )
