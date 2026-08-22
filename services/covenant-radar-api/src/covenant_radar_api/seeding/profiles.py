"""Seed data profiles for covenant-radar-api.

This module contains TypedDict definitions and seed data for populating
the database with demo data. All data is pure - no IO, no side effects.
"""

from __future__ import annotations

from typing import Literal, TypedDict

# =============================================================================
# TypedDicts for Seed Data
# =============================================================================


class DealSeed(TypedDict, total=True):
    """Seed data for a deal."""

    name: str
    borrower: str
    sector: Literal["Technology", "Finance", "Healthcare"]
    region: Literal["North America", "Europe", "Asia"]
    commitment_cents: int
    currency: str
    maturity_iso: str


class CovenantSeed(TypedDict, total=True):
    """Seed data for a covenant."""

    name: str
    formula: str
    threshold_scaled: int
    direction: Literal["<=", ">="]
    frequency: Literal["QUARTERLY", "ANNUAL"]


class MetricsSeed(TypedDict, total=True):
    """Seed data for financial metrics in a period."""

    total_debt: int
    ebitda: int
    interest_expense: int
    current_assets: int
    current_liabilities: int


class PeriodSeed(TypedDict, total=True):
    """Seed data for a period with metrics and expected status."""

    start_iso: str
    end_iso: str
    metrics: MetricsSeed
    expected_status: Literal["OK", "NEAR_BREACH", "BREACH"]


class DealProfile(TypedDict, total=True):
    """Complete profile for seeding a deal with all related data."""

    deal: DealSeed
    covenants: tuple[CovenantSeed, ...]
    periods: tuple[PeriodSeed, ...]


# =============================================================================
# Shared Seed Constants and Helpers
# =============================================================================


# Scale factor for financial values (6 decimal places)
_SCALE: int = 1_000_000


# Period dates (Q1 2024 back to Q1 2023)
_PERIOD_DATES: tuple[tuple[str, str], ...] = (
    ("2024-01-01", "2024-03-31"),
    ("2023-10-01", "2023-12-31"),
    ("2023-07-01", "2023-09-30"),
    ("2023-04-01", "2023-06-30"),
    ("2023-01-01", "2023-03-31"),
)


def _build_periods(
    metrics: tuple[MetricsSeed, ...],
    statuses: tuple[Literal["OK", "NEAR_BREACH", "BREACH"], ...],
) -> tuple[PeriodSeed, ...]:
    """Build period seeds from metrics and expected statuses."""
    result: list[PeriodSeed] = []
    for i, (dates, metric) in enumerate(zip(_PERIOD_DATES, metrics, strict=True)):
        result.append(
            PeriodSeed(
                start_iso=dates[0],
                end_iso=dates[1],
                metrics=metric,
                expected_status=statuses[i],
            )
        )
    return tuple(result)


__all__ = [
    "CovenantSeed",
    "DealProfile",
    "DealSeed",
    "MetricsSeed",
    "PeriodSeed",
]
