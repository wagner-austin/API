"""Actual matched financial observations exercise the strict feature boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue, narrow_json_to_dict, require_dict, require_list

from covenant_ml.markets.disclosure_records import (
    Disclosure,
    FinancialFact,
    decode_disclosure,
    decode_financial_fact,
    encode_disclosure,
    encode_financial_fact,
)
from covenant_ml.markets.equity_features import financial_strength_inputs
from covenant_ml.markets.validation import read_document


def inputs(
    index: int = 0,
) -> tuple[
    tuple[FinancialFact, FinancialFact, FinancialFact, FinancialFact],
    tuple[Disclosure, Disclosure],
]:
    """Decode the captured Apple or Microsoft observation through real codecs."""
    document = read_document(Path(__file__).parent / "fixtures/equity_features_actual.json")
    row = narrow_json_to_dict(require_list(document, "rows")[index])
    measures = require_dict(row, "measures")

    def fact(tag: str) -> FinancialFact:
        observations = require_list(require_dict(measures, tag), "facts")
        assert len(observations) == 1
        return decode_financial_fact(narrow_json_to_dict(observations[0]))

    return (
        (fact("AssetsCurrent"), fact("LiabilitiesCurrent"), fact("Assets"), fact("NetIncomeLoss")),
        (
            decode_disclosure(require_dict(row, "current_filing")),
            decode_disclosure(require_dict(row, "annual_filing")),
        ),
    )


def test_actual_ratios_and_negative_annual_income() -> None:
    for index, expected in (
        (0, (149818 / 149326, 112010 / 359241)),
        (1, (207710 / 168825, 133749 / 758376)),
    ):
        facts, filings = inputs(index)
        assert (
            financial_strength_inputs(
                *facts,
                *filings,
                received_at="2026-10-14T06:00:00Z",
                decision_at="2026-10-14T13:00:00Z",
            )
            == expected
        )
    facts, filings = inputs()
    changed = encode_financial_fact(facts[3])
    changed["value"] = -1000.0
    assert financial_strength_inputs(
        facts[0],
        facts[1],
        facts[2],
        decode_financial_fact(changed),
        *filings,
        received_at="2026-10-14T06:00:00Z",
        decision_at="2026-10-14T13:00:00Z",
    ) == (149818 / 149326, -1000 / 359241000000)


@pytest.mark.parametrize(
    "index,field,value,code",
    [
        (0, "cik", 1, "EQUITY-007"),
        (1, "accession", "0000000001-26-000001", "EQUITY-007"),
        (2, "form", "10-Q", "EQUITY-007"),
        (3, "filed", "2026-01-01", "EQUITY-007"),
        (0, "tag", "us-gaap:Assets", "EQUITY-007"),
        (0, "unit", "EUR", "EQUITY-007"),
        (0, "start", "2026-01-01", "EQUITY-008"),
        (1, "end", "2026-06-26", "EQUITY-008"),
        (2, "end", "2025-09-26", "EQUITY-008"),
        (3, "start", None, "EQUITY-008"),
        (3, "start", "2025-01-01", "EQUITY-008"),
        (3, "start", "2024-01-01", "EQUITY-008"),
        (0, "value", -1, "EQUITY-010"),
        (1, "value", 0, "EQUITY-010"),
        (2, "value", 0, "EQUITY-010"),
        (1, "value", 1e-310, "EQUITY-010"),
        (2, "value", 1e-310, "EQUITY-010"),
    ],
)
def test_malformed_pairings(index: int, field: str, value: JSONValue, code: str) -> None:
    facts, filings = inputs()
    changed = encode_financial_fact(facts[index])
    changed[field] = value
    updated = list(facts)
    updated[index] = decode_financial_fact(changed)
    with pytest.raises(ValueError, match=code):
        financial_strength_inputs(
            updated[0],
            updated[1],
            updated[2],
            updated[3],
            *filings,
            received_at="2026-10-14T06:00:00Z",
            decision_at="2026-10-14T13:00:00Z",
        )


def test_filing_issuer_and_forms() -> None:
    facts, filings = inputs()
    changed = encode_disclosure(filings[1])
    changed["cik"] = 1
    with pytest.raises(ValueError, match="EQUITY-007"):
        financial_strength_inputs(
            *facts,
            filings[0],
            decode_disclosure(changed),
            received_at="2026-10-14T06:00:00Z",
            decision_at="2026-10-14T13:00:00Z",
        )
    for index in (0, 1):
        changed_filing = encode_disclosure(filings[index])
        changed_filing["form"] = "8-K"
        updated_filings = list(filings)
        updated_filings[index] = decode_disclosure(changed_filing)
        updated_facts = list(facts)
        for position in (0, 1) if index == 0 else (2, 3):
            changed_fact = encode_financial_fact(facts[position])
            changed_fact["form"] = "8-K"
            updated_facts[position] = decode_financial_fact(changed_fact)
        with pytest.raises(ValueError, match="EQUITY-007"):
            financial_strength_inputs(
                updated_facts[0],
                updated_facts[1],
                updated_facts[2],
                updated_facts[3],
                updated_filings[0],
                updated_filings[1],
                received_at="2026-10-14T06:00:00Z",
                decision_at="2026-10-14T13:00:00Z",
            )


@pytest.mark.parametrize(
    "received,decision",
    [
        ("2020-01-01T00:00:00Z", "2026-10-14T13:00:00Z"),
        ("2026-10-14T13:00:00Z", "2026-10-14T13:00:00Z"),
        ("2026-10-14T13:00:01Z", "2026-10-14T13:00:00Z"),
    ],
)
def test_receipts_must_precede_decisions(received: str, decision: str) -> None:
    facts, filings = inputs()
    with pytest.raises(ValueError, match="EQUITY-009"):
        financial_strength_inputs(*facts, *filings, received_at=received, decision_at=decision)
