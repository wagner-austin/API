"""Financial ratios bound to exact filings, periods and observed availability."""

from __future__ import annotations

from datetime import date
from math import isfinite

from .disclosure_records import Disclosure, FinancialFact
from .validation import parse_timestamp


def financial_strength_inputs(
    current_assets: FinancialFact,
    current_liabilities: FinancialFact,
    annual_assets: FinancialFact,
    annual_income: FinancialFact,
    current_filing: Disclosure,
    annual_filing: Disclosure,
    *,
    received_at: str,
    decision_at: str,
) -> tuple[float, float]:
    """Compute current ratio and annual income over ending assets without imputation.

    Args:
        current_assets: Validated instantaneous current assets in USD.
        current_liabilities: Validated instantaneous current liabilities in USD.
        annual_assets: Validated annual ending total assets in USD.
        annual_income: Validated annual-duration net income in USD.
        current_filing: Exact filing metadata for both current observations.
        annual_filing: Exact annual filing metadata for both annual observations.
        received_at: Latest actual receipt of all source evidence used here.
        decision_at: Aware decision timestamp strictly after the receipt.

    Returns:
        Current ratio and annual net income divided by ending total assets.

    Raises:
        ValueError: EQUITY-007 for issuer, accession, metadata, concept or unit
            mismatch; EQUITY-008 for invalid instant/annual period pairing;
            EQUITY-009 for receipt/acceptance/decision chronology violations;
            EQUITY-010 for negative assets, invalid denominators or arithmetic.
    """
    received, decision = parse_timestamp(received_at), parse_timestamp(decision_at)
    facts = (current_assets, current_liabilities, annual_assets, annual_income)
    filings = (current_filing, current_filing, annual_filing, annual_filing)
    concepts = ("AssetsCurrent", "LiabilitiesCurrent", "Assets", "NetIncomeLoss")
    if current_filing["cik"] != annual_filing["cik"]:
        raise ValueError("[EQUITY-007] current and annual filings must identify one issuer")
    for fact, filing, concept in zip(facts, filings, concepts, strict=True):
        if (
            (fact["cik"], fact["accession"], fact["form"], fact["filed"])
            != (filing["cik"], filing["accession"], filing["form"], filing["filed"])
            or fact["tag"] != "us-gaap:" + concept
            or fact["unit"] != "USD"
        ):
            raise ValueError("[EQUITY-007] financial input differs from required filing or concept")
        if parse_timestamp(filing["accepted_at"]) > received or received >= decision:
            raise ValueError("[EQUITY-009] require acceptance <= observed receipt < decision")
    if current_filing["form"] not in ("10-K", "10-K/A", "10-Q", "10-Q/A") or annual_filing[
        "form"
    ] not in ("10-K", "10-K/A"):
        raise ValueError("[EQUITY-007] ratios require current periodic and annual filing forms")
    if (
        any(
            fact["start"] is not None
            for fact in (current_assets, current_liabilities, annual_assets)
        )
        or current_assets["end"] != current_liabilities["end"]
        or annual_assets["end"] != annual_income["end"]
    ):
        raise ValueError("[EQUITY-008] paired balances must be instant and share exact period ends")
    start = annual_income["start"]
    if (
        start is None
        or not 300
        <= (date.fromisoformat(annual_income["end"]) - date.fromisoformat(start)).days
        <= 400
    ):
        raise ValueError("[EQUITY-008] annual income must span 300 through 400 days")
    if (
        current_assets["value"] < 0
        or current_liabilities["value"] <= 0
        or annual_assets["value"] <= 0
    ):
        raise ValueError("[EQUITY-010] assets must be nonnegative and denominators positive")
    current_ratio = current_assets["value"] / current_liabilities["value"]
    annual_roa = annual_income["value"] / annual_assets["value"]
    if not all(isfinite(value) for value in (current_ratio, annual_roa)):
        raise ValueError("[EQUITY-010] financial ratios must remain finite")
    return current_ratio, annual_roa
