"""Strict financial codecs reject malformed identities, periods and values."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.disclosure_records import (
    Disclosure,
    FinancialFact,
    decode_disclosure,
    decode_financial_fact,
    encode_disclosure,
    encode_financial_fact,
)


def filing_document() -> JSONObject:
    """Return valid normalized input for real codec validation."""
    return {
        "cik": 320193,
        "accession": "0001193125-09-214859",
        "form": "10-K",
        "filed": "2009-10-27",
        "accepted_at": "2009-10-28T00:18:29Z",
    }


def fact_document() -> JSONObject:
    """Return the original captured financial observation."""
    return {
        "cik": 320193,
        "accession": "0001193125-09-214859",
        "form": "10-K",
        "filed": "2009-10-27",
        "tag": "us-gaap:Assets",
        "unit": "USD",
        "start": None,
        "end": "2009-09-26",
        "value": 53851000000,
    }


def test_exact_roundtrips() -> None:
    filing: Disclosure = {
        "cik": 320193,
        "accession": "0001193125-09-214859",
        "form": "10-K",
        "filed": "2009-10-27",
        "accepted_at": "2009-10-28T00:18:29Z",
    }
    fact: FinancialFact = {
        "cik": 320193,
        "accession": "0001193125-09-214859",
        "form": "10-K",
        "filed": "2009-10-27",
        "tag": "us-gaap:Assets",
        "unit": "USD",
        "start": None,
        "end": "2009-09-26",
        "value": 53851000000.0,
    }
    assert decode_disclosure(filing_document()) == filing
    assert decode_disclosure(encode_disclosure(filing)) == filing
    assert decode_financial_fact(fact_document()) == fact
    assert decode_financial_fact(encode_financial_fact(fact)) == fact
    duration = fact_document()
    duration.update({"start": "2009-01-01", "value": -1.25})
    assert encode_financial_fact(decode_financial_fact(duration)) == duration


@pytest.mark.parametrize(
    "key,value,code",
    [
        ("accession", "000119312509214859", "DISCLOSURE-001"),
        ("cik", 0, "MARKET-001"),
        ("cik", 10000000000, "MARKET-001"),
        ("form", " ", "MARKET-003"),
        ("accepted_at", "2009-10-28T00:18:29", "MARKET-006"),
    ],
)
def test_filing_validation(key: str, value: JSONValue, code: str) -> None:
    document = filing_document()
    document[key] = value
    with pytest.raises(ValueError, match=code):
        decode_disclosure(document)


@pytest.mark.parametrize(
    "key,value",
    [
        ("start", "2009-09-27"),
        ("end", "2009-10-28"),
        ("value", float("nan")),
        ("value", float("inf")),
        ("value", float("-inf")),
    ],
)
def test_fact_chronology_and_finite_value(key: str, value: JSONValue) -> None:
    document = fact_document()
    document[key] = value
    with pytest.raises(ValueError, match="DISCLOSURE-002"):
        decode_financial_fact(document)


def test_missing_and_wrongly_typed_fields() -> None:
    missing = fact_document()
    del missing["start"]
    with pytest.raises(ValueError, match="DISCLOSURE-002"):
        decode_financial_fact(missing)
    for value in (True, "1.2", None):
        malformed = fact_document()
        malformed["value"] = value
        with pytest.raises(TypeError, match="value"):
            decode_financial_fact(malformed)
    missing_filing = filing_document()
    del missing_filing["accepted_at"]
    with pytest.raises(TypeError, match="accepted_at"):
        decode_disclosure(missing_filing)
