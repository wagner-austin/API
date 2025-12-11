"""Pure functions to generate domain objects from seed profiles.

This module contains no IO, no side effects. All functions take seed
data and return domain objects (TypedDicts).
"""

from __future__ import annotations

from typing import Literal

from covenant_domain import (
    Covenant,
    CovenantId,
    CovenantResult,
    Deal,
    DealId,
    Measurement,
)

from .profiles import CovenantSeed, DealSeed, MetricsSeed, PeriodSeed


def generate_deal(deal_id: str, seed: DealSeed) -> Deal:
    """Generate a Deal from seed data.

    Args:
        deal_id: UUID string for the deal.
        seed: Seed data containing deal attributes.

    Returns:
        Fully populated Deal TypedDict.
    """
    return Deal(
        id=DealId(value=deal_id),
        name=seed["name"],
        borrower=seed["borrower"],
        sector=seed["sector"],
        region=seed["region"],
        commitment_amount_cents=seed["commitment_cents"],
        currency=seed["currency"],
        maturity_date_iso=seed["maturity_iso"],
    )


def generate_covenant(cov_id: str, deal_id: str, seed: CovenantSeed) -> Covenant:
    """Generate a Covenant from seed data.

    Args:
        cov_id: UUID string for the covenant.
        deal_id: UUID string for the parent deal.
        seed: Seed data containing covenant attributes.

    Returns:
        Fully populated Covenant TypedDict.
    """
    return Covenant(
        id=CovenantId(value=cov_id),
        deal_id=DealId(value=deal_id),
        name=seed["name"],
        formula=seed["formula"],
        threshold_value_scaled=seed["threshold_scaled"],
        threshold_direction=seed["direction"],
        frequency=seed["frequency"],
    )


def generate_measurements(
    deal_id: str,
    period_start: str,
    period_end: str,
    metrics: MetricsSeed,
) -> tuple[Measurement, ...]:
    """Generate Measurements from seed metrics for a period.

    Args:
        deal_id: UUID string for the parent deal.
        period_start: ISO date string for period start.
        period_end: ISO date string for period end.
        metrics: Seed data containing metric values.

    Returns:
        Tuple of 5 Measurement TypedDicts (one per metric).
    """
    deal_id_typed = DealId(value=deal_id)
    return (
        Measurement(
            deal_id=deal_id_typed,
            period_start_iso=period_start,
            period_end_iso=period_end,
            metric_name="total_debt",
            metric_value_scaled=metrics["total_debt"],
        ),
        Measurement(
            deal_id=deal_id_typed,
            period_start_iso=period_start,
            period_end_iso=period_end,
            metric_name="ebitda",
            metric_value_scaled=metrics["ebitda"],
        ),
        Measurement(
            deal_id=deal_id_typed,
            period_start_iso=period_start,
            period_end_iso=period_end,
            metric_name="interest_expense",
            metric_value_scaled=metrics["interest_expense"],
        ),
        Measurement(
            deal_id=deal_id_typed,
            period_start_iso=period_start,
            period_end_iso=period_end,
            metric_name="current_assets",
            metric_value_scaled=metrics["current_assets"],
        ),
        Measurement(
            deal_id=deal_id_typed,
            period_start_iso=period_start,
            period_end_iso=period_end,
            metric_name="current_liabilities",
            metric_value_scaled=metrics["current_liabilities"],
        ),
    )


def calculate_placeholder_value(status: Literal["OK", "NEAR_BREACH", "BREACH"]) -> int:
    """Calculate a placeholder calculated_value based on status.

    This provides representative values for demonstration purposes.
    Real values would come from formula evaluation.

    Args:
        status: The expected covenant status.

    Returns:
        Scaled integer representing calculated covenant value.
    """
    if status == "OK":
        return 2_000_000  # Well below typical thresholds
    if status == "NEAR_BREACH":
        return 3_800_000  # Near threshold
    return 5_000_000  # Above threshold (BREACH)


def generate_covenant_result(
    cov_id: str,
    period: PeriodSeed,
) -> CovenantResult:
    """Generate a CovenantResult from seed period data.

    Args:
        cov_id: UUID string for the covenant.
        period: Seed data containing period and expected status.

    Returns:
        Fully populated CovenantResult TypedDict.
    """
    return CovenantResult(
        covenant_id=CovenantId(value=cov_id),
        period_start_iso=period["start_iso"],
        period_end_iso=period["end_iso"],
        calculated_value_scaled=calculate_placeholder_value(period["expected_status"]),
        status=period["expected_status"],
    )


__all__ = [
    "calculate_placeholder_value",
    "generate_covenant",
    "generate_covenant_result",
    "generate_deal",
    "generate_measurements",
]
