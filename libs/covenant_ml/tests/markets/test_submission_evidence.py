"""Quote and timing codecs reject chronology errors independently of order results."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, require_dict

from covenant_ml.markets.execution_records import encode_intent
from covenant_ml.markets.submission_evidence import (
    decode_submission_capture,
    decode_submission_timing,
    encode_submission_capture,
    encode_submission_timing,
)

from ._execution_fixtures import intent


def capture_document() -> JSONObject:
    return {
        "intent": encode_intent(intent(created=120000)),
        "market": {
            "ticker": "MARKET-1",
            "status": "active",
            "close_time": "2099-01-01T00:00:00Z",
            "rules_primary": "Exact native rules.",
        },
        "book": {
            "orderbook_fp": {
                "yes_dollars": [["0.9000", "2.00"]],
                "no_dollars": [["0.0200", "1.00"]],
            },
        },
        "preflight_completed_ms": 120000,
        "market_received_ms": 120010,
        "book_received_ms": 120020,
        "decision_ms": 120030,
        "started_ms": 120040,
    }


def test_capture_roundtrip_detaches_nested_native_evidence() -> None:
    document = capture_document()
    record = decode_submission_capture(document)
    assert encode_submission_capture(record) == document
    require_dict(document, "market")["rules_primary"] = "Later revision."
    assert require_dict(encode_submission_capture(record), "market")["rules_primary"] == (
        "Exact native rules."
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("market_received_ms", 119999),
        ("book_received_ms", 120009),
        ("decision_ms", 120019),
        ("started_ms", 120029),
        ("started_ms", 125001),
    ],
)
def test_chronology_and_age_fail(field: str, value: int) -> None:
    document = capture_document()
    document[field] = value
    with pytest.raises(ValueError, match="DEMO-EVIDENCE-001"):
        decode_submission_capture(document)


def test_exact_freshness_boundary_and_intent_expiration() -> None:
    document = capture_document()
    document["started_ms"] = 125000
    assert decode_submission_capture(document)["started_ms"] == 125000
    require_dict(document, "intent")["expires_ms"] = 125000
    with pytest.raises(ValueError, match="DEMO-EVIDENCE-001"):
        decode_submission_capture(document)


def test_quote_above_limit_is_not_executable() -> None:
    document = capture_document()
    require_dict(document, "intent")["limit_price"] = 9700
    with pytest.raises(ValueError, match="DEMO-TRIAL-006"):
        decode_submission_capture(document)


def test_timing_roundtrip_and_clock_reversal() -> None:
    document: JSONObject = {
        "client_order_id": "recorded-intent",
        "started_ms": 120000,
        "returned_ms": 120100,
        "capture_sha256": "b" * 64,
        "acknowledgement_sha256": "a" * 64,
    }
    assert encode_submission_timing(decode_submission_timing(document)) == document
    document["returned_ms"] = 119999
    with pytest.raises(ValueError, match="MARKET-001"):
        decode_submission_timing(document)


@pytest.mark.parametrize("digest", ["a" * 63, "G" * 64])
def test_timing_rejects_invalid_digest(digest: str) -> None:
    with pytest.raises(ValueError, match="CAMPAIGN-001"):
        decode_submission_timing(
            {
                "client_order_id": "recorded-intent",
                "started_ms": 120000,
                "returned_ms": 120100,
                "capture_sha256": "b" * 64,
                "acknowledgement_sha256": digest,
            }
        )
