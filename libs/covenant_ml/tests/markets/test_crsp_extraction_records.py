"""Strict codecs for extraction limits and the published metadata receipt."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject

from covenant_ml.markets.crsp_extraction_records import (
    decode_crsp_extraction_receipt,
    decode_crsp_limits,
    decode_crsp_partition_receipt,
    encode_crsp_extraction_receipt,
    encode_crsp_limits,
)

DIGEST = "ab" * 32


def limits_document() -> JSONObject:
    """Bounds a small synthetic extraction satisfies."""
    return {
        "max_rows": 20000,
        "max_relation_bytes": 4_000_000,
        "max_partition_bytes": 8_000_000,
        "seconds": 300,
    }


def _file(name: str) -> JSONObject:
    return {"name": name, "sha256": DIGEST, "size_bytes": 10}


def partition_document(partition: str) -> JSONObject:
    """A sealed partition's metadata."""
    return {
        "partition": partition,
        "observation_end": "2019-12-31",
        "calendar_end": "2020-02-29",
        "manifest": _file("manifest.json"),
        "bundle": _file("bundle.json"),
        "membership_rows": 8,
        "returns_rows": 7000,
        "calendar_rows": 1520,
        "delists_rows": 1,
        "plaintext_bytes": 900_000,
    }


def receipt_document() -> JSONObject:
    """A complete receipt as the writer encodes it."""
    return {
        "schema": "pg_temp",
        "setup_sha256": DIGEST,
        "protocol": _file("protocol.json"),
        "driver_sha256": DIGEST,
        "started_at": "2026-10-31T09:00:00+00:00",
        "completed_at": "2026-10-31T09:00:05+00:00",
        "snapshot": "750:750:",
        "transaction_started_at": "2026-10-31 09:00:00.5+00",
        "limits": limits_document(),
        "train_validation": partition_document("train-validation"),
        "final": partition_document("final"),
        "plaintext_removed": True,
    }


def test_receipt_round_trips_exactly() -> None:
    document = receipt_document()
    receipt = decode_crsp_extraction_receipt(document)
    assert encode_crsp_extraction_receipt(receipt) == document
    assert receipt["final"]["partition"] == "final"


def test_vendor_receipt_carries_no_setup() -> None:
    document = receipt_document()
    document["schema"] = "crsp_a_stock"
    document["setup_sha256"] = None
    assert decode_crsp_extraction_receipt(document)["setup_sha256"] is None


def test_limits_round_trip_and_refuse_a_partition_budget_below_one_relation() -> None:
    assert encode_crsp_limits(decode_crsp_limits(limits_document())) == limits_document()
    document = limits_document()
    document["max_partition_bytes"] = 3_999_999
    with pytest.raises(ValueError, match=r"\[CRSP-010\] the partition budget"):
        decode_crsp_limits(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_rows", 0), ("max_relation_bytes", 0), ("seconds", 86401), ("max_partition_bytes", 0)],
)
def test_limits_outside_their_intervals_are_refused(field: str, value: int) -> None:
    document = limits_document()
    document[field] = value
    with pytest.raises(ValueError, match=r"\[MARKET-001\]"):
        decode_crsp_limits(document)


def test_unknown_or_missing_fields_are_refused() -> None:
    document = limits_document()
    document["extra"] = 1
    with pytest.raises(ValueError, match=r"\[CRSP-010\] limits requires exactly"):
        decode_crsp_limits(document)
    receipt = receipt_document()
    del receipt["snapshot"]
    with pytest.raises(ValueError, match=r"\[CRSP-010\] an extraction receipt requires"):
        decode_crsp_extraction_receipt(receipt)
    partition = partition_document("final")
    partition["rows"] = 1
    with pytest.raises(ValueError, match=r"\[CRSP-010\] a partition receipt requires"):
        decode_crsp_partition_receipt(partition, "final")


def test_partition_must_sit_in_its_own_position() -> None:
    receipt = receipt_document()
    receipt["final"] = partition_document("train-validation")
    with pytest.raises(ValueError, match=r"\[CRSP-010\] expected the final partition"):
        decode_crsp_extraction_receipt(receipt)


def test_negative_counts_and_noncanonical_days_are_refused() -> None:
    partition = partition_document("final")
    partition["returns_rows"] = -1
    with pytest.raises(ValueError, match=r"\[MARKET-001\]"):
        decode_crsp_partition_receipt(partition, "final")
    partition = partition_document("final")
    partition["observation_end"] = "2019-12-1"
    with pytest.raises(ValueError, match="Invalid isoformat"):
        decode_crsp_partition_receipt(partition, "final")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("started_at", "2026-10-31T09:00:00", "started_at must carry UTC"),
        ("completed_at", "2026-10-31T08:59:59+00:00", "completion must not precede"),
        ("snapshot", " ", "the database snapshot must be recorded"),
        ("plaintext_removed", False, "private plaintext must have been removed"),
    ],
)
def test_receipt_refuses_inconsistent_evidence(field: str, value: str | bool, message: str) -> None:
    receipt = receipt_document()
    receipt[field] = value
    with pytest.raises(ValueError, match=rf"\[CRSP-010\] {message}"):
        decode_crsp_extraction_receipt(receipt)


def test_receipt_refuses_a_malformed_setup_digest() -> None:
    receipt = receipt_document()
    receipt["setup_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match=r"\[CAMPAIGN-001\]"):
        decode_crsp_extraction_receipt(receipt)
