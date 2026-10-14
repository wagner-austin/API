"""Strict evidence codecs reject missing vintages and impossible outcome histories."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.evaluation.warning_records import (
    decode_warning_disclosure,
    decode_warning_followup,
    encode_warning_disclosure,
    encode_warning_followup,
)


def disclosure_document() -> JSONObject:
    """Return a synthetic dated disclosure for real codec tests."""
    return {
        "borrower": "A",
        "period_end_ms": 10,
        "available_ms": 20,
        "source_sha256": "a" * 64,
        "debt_to_ebitda": 4.0,
        "interest_cover": 2.0,
        "current_ratio": 1.0,
    }


def followup_document() -> JSONObject:
    """Return synthetic complete surveillance with a future breach."""
    return {
        "borrower": "A",
        "coverage_start_ms": 0,
        "observed_through_ms": 100,
        "available_ms": 110,
        "first_breach_ms": 70,
        "source_sha256": "b" * 64,
    }


def test_roundtrips_preserve_all_evidence() -> None:
    disclosure = disclosure_document()
    followup = followup_document()
    assert encode_warning_disclosure(decode_warning_disclosure(disclosure)) == disclosure
    assert encode_warning_followup(decode_warning_followup(followup)) == followup
    followup["first_breach_ms"] = None
    assert encode_warning_followup(decode_warning_followup(followup)) == followup
    disclosure["debt_to_ebitda"] = -2.0
    assert decode_warning_disclosure(disclosure)["debt_to_ebitda"] == -2.0


@pytest.mark.parametrize(
    "key,value,code",
    [
        ("period_end_ms", -1, "WARN-001"),
        ("available_ms", 2**53, "WARN-001"),
        ("borrower", " ", "WARN-002"),
        ("borrower", " A", "WARN-002"),
        ("source_sha256", "a", "WARN-002"),
        ("source_sha256", "Z" * 64, "WARN-002"),
        ("debt_to_ebitda", float("nan"), "WARN-003"),
        ("interest_cover", float("inf"), "WARN-003"),
        ("current_ratio", float("-inf"), "WARN-003"),
        ("available_ms", 9, "WARN-004"),
    ],
)
def test_disclosure_rejects_invalid_values(key: str, value: JSONValue, code: str) -> None:
    document = disclosure_document()
    document[key] = value
    with pytest.raises(ValueError, match=code):
        decode_warning_disclosure(document)


@pytest.mark.parametrize("key", list(disclosure_document()))
def test_disclosure_requires_each_field(key: str) -> None:
    document = disclosure_document()
    del document[key]
    with pytest.raises(TypeError, match=key):
        decode_warning_disclosure(document)


@pytest.mark.parametrize("key,value", [("available_ms", True), ("debt_to_ebitda", "4")])
def test_disclosure_rejects_incorrect_types(key: str, value: JSONValue) -> None:
    document = disclosure_document()
    document[key] = value
    with pytest.raises(TypeError):
        decode_warning_disclosure(document)


@pytest.mark.parametrize(
    "key,value",
    [
        ("coverage_start_ms", 101),
        ("available_ms", 99),
        ("first_breach_ms", 101),
        ("coverage_start_ms", 71),
    ],
)
def test_followup_rejects_impossible_chronology(key: str, value: JSONValue) -> None:
    document = followup_document()
    document[key] = value
    with pytest.raises(ValueError, match="WARN-005"):
        decode_warning_followup(document)


def test_missing_breach_is_not_no_breach() -> None:
    document = followup_document()
    del document["first_breach_ms"]
    with pytest.raises(ValueError, match=r"WARN-005.*explicit"):
        decode_warning_followup(document)


@pytest.mark.parametrize(
    "key", ["borrower", "coverage_start_ms", "observed_through_ms", "available_ms", "source_sha256"]
)
def test_followup_requires_each_field(key: str) -> None:
    document = followup_document()
    del document[key]
    with pytest.raises(TypeError, match=key):
        decode_warning_followup(document)


def test_missing_availability_is_not_replaced_with_period_end() -> None:
    document = disclosure_document()
    del document["available_ms"]
    with pytest.raises(TypeError, match="available_ms"):
        decode_warning_disclosure(document)
