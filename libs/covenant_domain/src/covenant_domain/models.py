from __future__ import annotations

from typing import Literal, TypedDict


class DealId(TypedDict, total=True):
    """Immutable deal identifier."""

    value: str  # UUID string


class Deal(TypedDict, total=True):
    """Loan deal record. Immutable by convention."""

    id: DealId
    name: str
    borrower: str
    sector: str
    region: str
    commitment_amount_cents: int  # Store as cents to avoid Decimal
    currency: str
    maturity_date_iso: str  # ISO 8601 date string


class CovenantId(TypedDict, total=True):
    """Immutable covenant identifier."""

    value: str  # UUID string


class Covenant(TypedDict, total=True):
    """Covenant rule definition. Immutable by convention."""

    id: CovenantId
    deal_id: DealId
    name: str
    formula: str  # e.g., "total_debt / ebitda"
    threshold_value_scaled: int  # Scaled integer (multiply by 1_000_000)
    threshold_direction: Literal["<=", ">="]
    frequency: Literal["QUARTERLY", "ANNUAL"]


class Measurement(TypedDict, total=True):
    """Financial metric measurement for a period."""

    deal_id: DealId
    period_start_iso: str  # ISO 8601 date
    period_end_iso: str  # ISO 8601 date
    metric_name: str  # e.g., "total_debt", "ebitda"
    metric_value_scaled: int  # Scaled integer (multiply by 1_000_000)


class CovenantResult(TypedDict, total=True):
    """Computed covenant status for a period."""

    covenant_id: CovenantId
    period_start_iso: str
    period_end_iso: str
    calculated_value_scaled: int
    status: Literal["OK", "NEAR_BREACH", "BREACH"]


__all__ = [
    "Covenant",
    "CovenantId",
    "CovenantResult",
    "Deal",
    "DealId",
    "Measurement",
]
