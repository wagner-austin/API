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
# Seed Data Constants
# =============================================================================

# Scale factor for financial values (6 decimal places)
_SCALE: int = 1_000_000

# Metrics for TechCorp - healthy company, no breaches
_TECHCORP_METRICS: tuple[MetricsSeed, ...] = (
    MetricsSeed(
        total_debt=100 * _SCALE,
        ebitda=50 * _SCALE,
        interest_expense=10 * _SCALE,
        current_assets=80 * _SCALE,
        current_liabilities=40 * _SCALE,
    ),
    MetricsSeed(
        total_debt=95 * _SCALE,
        ebitda=48 * _SCALE,
        interest_expense=9 * _SCALE,
        current_assets=75 * _SCALE,
        current_liabilities=38 * _SCALE,
    ),
    MetricsSeed(
        total_debt=90 * _SCALE,
        ebitda=45 * _SCALE,
        interest_expense=9 * _SCALE,
        current_assets=72 * _SCALE,
        current_liabilities=36 * _SCALE,
    ),
    MetricsSeed(
        total_debt=85 * _SCALE,
        ebitda=43 * _SCALE,
        interest_expense=8 * _SCALE,
        current_assets=68 * _SCALE,
        current_liabilities=34 * _SCALE,
    ),
    MetricsSeed(
        total_debt=80 * _SCALE,
        ebitda=40 * _SCALE,
        interest_expense=8 * _SCALE,
        current_assets=64 * _SCALE,
        current_liabilities=32 * _SCALE,
    ),
)

# Metrics for FinanceGroup - struggling company, has breaches
_FINANCEGROUP_METRICS: tuple[MetricsSeed, ...] = (
    MetricsSeed(
        total_debt=200 * _SCALE,
        ebitda=40 * _SCALE,  # Leverage: 5.0x - BREACH
        interest_expense=20 * _SCALE,
        current_assets=50 * _SCALE,
        current_liabilities=45 * _SCALE,
    ),
    MetricsSeed(
        total_debt=180 * _SCALE,
        ebitda=45 * _SCALE,  # Leverage: 4.0x - NEAR_BREACH
        interest_expense=18 * _SCALE,
        current_assets=55 * _SCALE,
        current_liabilities=48 * _SCALE,
    ),
    MetricsSeed(
        total_debt=160 * _SCALE,
        ebitda=50 * _SCALE,  # Leverage: 3.2x - OK
        interest_expense=16 * _SCALE,
        current_assets=60 * _SCALE,
        current_liabilities=50 * _SCALE,
    ),
    MetricsSeed(
        total_debt=150 * _SCALE,
        ebitda=55 * _SCALE,
        interest_expense=15 * _SCALE,
        current_assets=65 * _SCALE,
        current_liabilities=55 * _SCALE,
    ),
    MetricsSeed(
        total_debt=140 * _SCALE,
        ebitda=60 * _SCALE,
        interest_expense=14 * _SCALE,
        current_assets=70 * _SCALE,
        current_liabilities=60 * _SCALE,
    ),
)

# Metrics for HealthCare - mixed results
_HEALTHCARE_METRICS: tuple[MetricsSeed, ...] = (
    MetricsSeed(
        total_debt=60 * _SCALE,
        ebitda=30 * _SCALE,
        interest_expense=6 * _SCALE,
        current_assets=50 * _SCALE,
        current_liabilities=40 * _SCALE,  # Ratio: 1.25x - OK
    ),
    MetricsSeed(
        total_debt=58 * _SCALE,
        ebitda=29 * _SCALE,
        interest_expense=5 * _SCALE,
        current_assets=48 * _SCALE,
        current_liabilities=42 * _SCALE,  # Ratio: 1.14x - BREACH
    ),
    MetricsSeed(
        total_debt=55 * _SCALE,
        ebitda=28 * _SCALE,
        interest_expense=5 * _SCALE,
        current_assets=52 * _SCALE,
        current_liabilities=38 * _SCALE,  # Ratio: 1.37x - OK
    ),
    MetricsSeed(
        total_debt=52 * _SCALE,
        ebitda=27 * _SCALE,
        interest_expense=5 * _SCALE,
        current_assets=54 * _SCALE,
        current_liabilities=36 * _SCALE,
    ),
    MetricsSeed(
        total_debt=50 * _SCALE,
        ebitda=26 * _SCALE,
        interest_expense=5 * _SCALE,
        current_assets=56 * _SCALE,
        current_liabilities=35 * _SCALE,
    ),
)

# Metrics for CloudTech - near breaches
_CLOUDTECH_METRICS: tuple[MetricsSeed, ...] = (
    MetricsSeed(
        total_debt=180 * _SCALE,
        ebitda=42 * _SCALE,  # Leverage: 4.29x - NEAR_BREACH
        interest_expense=18 * _SCALE,
        current_assets=90 * _SCALE,
        current_liabilities=60 * _SCALE,
    ),
    MetricsSeed(
        total_debt=170 * _SCALE,
        ebitda=40 * _SCALE,  # Leverage: 4.25x - NEAR_BREACH
        interest_expense=17 * _SCALE,
        current_assets=85 * _SCALE,
        current_liabilities=58 * _SCALE,
    ),
    MetricsSeed(
        total_debt=160 * _SCALE,
        ebitda=45 * _SCALE,  # Leverage: 3.56x - OK
        interest_expense=16 * _SCALE,
        current_assets=80 * _SCALE,
        current_liabilities=55 * _SCALE,
    ),
    MetricsSeed(
        total_debt=150 * _SCALE,
        ebitda=48 * _SCALE,
        interest_expense=15 * _SCALE,
        current_assets=78 * _SCALE,
        current_liabilities=52 * _SCALE,
    ),
    MetricsSeed(
        total_debt=140 * _SCALE,
        ebitda=50 * _SCALE,
        interest_expense=14 * _SCALE,
        current_assets=75 * _SCALE,
        current_liabilities=50 * _SCALE,
    ),
)

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


# =============================================================================
# Seed Profiles
# =============================================================================


TECHCORP_PROFILE: DealProfile = DealProfile(
    deal=DealSeed(
        name="TechCorp Senior Credit Facility",
        borrower="TechCorp Inc",
        sector="Technology",
        region="North America",
        commitment_cents=500_000_000_00,
        currency="USD",
        maturity_iso="2027-12-31",
    ),
    covenants=(
        CovenantSeed(
            name="Leverage Ratio",
            formula="total_debt / ebitda",
            threshold_scaled=4 * _SCALE,
            direction="<=",
            frequency="QUARTERLY",
        ),
        CovenantSeed(
            name="Interest Coverage",
            formula="ebitda / interest_expense",
            threshold_scaled=2 * _SCALE,
            direction=">=",
            frequency="QUARTERLY",
        ),
    ),
    periods=_build_periods(
        _TECHCORP_METRICS,
        ("OK", "OK", "OK", "OK", "OK"),
    ),
)


FINANCEGROUP_PROFILE: DealProfile = DealProfile(
    deal=DealSeed(
        name="FinanceGroup Term Loan B",
        borrower="FinanceGroup LLC",
        sector="Finance",
        region="Europe",
        commitment_cents=250_000_000_00,
        currency="EUR",
        maturity_iso="2026-06-30",
    ),
    covenants=(
        CovenantSeed(
            name="Leverage Ratio",
            formula="total_debt / ebitda",
            threshold_scaled=3_500_000,
            direction="<=",
            frequency="QUARTERLY",
        ),
    ),
    periods=_build_periods(
        _FINANCEGROUP_METRICS,
        ("BREACH", "NEAR_BREACH", "OK", "OK", "OK"),
    ),
)


HEALTHCARE_PROFILE: DealProfile = DealProfile(
    deal=DealSeed(
        name="HealthCare Revolving Credit",
        borrower="HealthCare Partners",
        sector="Healthcare",
        region="Asia",
        commitment_cents=150_000_000_00,
        currency="USD",
        maturity_iso="2028-03-31",
    ),
    covenants=(
        CovenantSeed(
            name="Current Ratio",
            formula="current_assets / current_liabilities",
            threshold_scaled=1_200_000,
            direction=">=",
            frequency="QUARTERLY",
        ),
    ),
    periods=_build_periods(
        _HEALTHCARE_METRICS,
        ("OK", "BREACH", "OK", "OK", "OK"),
    ),
)


CLOUDTECH_PROFILE: DealProfile = DealProfile(
    deal=DealSeed(
        name="CloudTech Asset Based Loan",
        borrower="CloudTech Systems",
        sector="Technology",
        region="Europe",
        commitment_cents=300_000_000_00,
        currency="USD",
        maturity_iso="2026-12-31",
    ),
    covenants=(
        CovenantSeed(
            name="Leverage Ratio",
            formula="total_debt / ebitda",
            threshold_scaled=4_500_000,
            direction="<=",
            frequency="QUARTERLY",
        ),
    ),
    periods=_build_periods(
        _CLOUDTECH_METRICS,
        ("NEAR_BREACH", "NEAR_BREACH", "OK", "OK", "OK"),
    ),
)


# =============================================================================
# Additional Safe Profiles (no breaches - low debt ratios)
# =============================================================================

# SafeTech - very healthy, debt/EBITDA around 1.5x
_SAFETECH_METRICS: tuple[MetricsSeed, ...] = tuple(
    MetricsSeed(
        total_debt=75 * _SCALE,
        ebitda=50 * _SCALE,  # Leverage: 1.5x
        interest_expense=7 * _SCALE,
        current_assets=90 * _SCALE,
        current_liabilities=40 * _SCALE,
    )
    for _ in range(5)
)

SAFETECH_PROFILE: DealProfile = DealProfile(
    deal=DealSeed(
        name="SafeTech Industries",
        borrower="SafeTech Corp",
        sector="Technology",
        region="North America",
        commitment_cents=400_000_000_00,
        currency="USD",
        maturity_iso="2028-06-30",
    ),
    covenants=(
        CovenantSeed(
            name="Leverage Ratio",
            formula="total_debt / ebitda",
            threshold_scaled=4 * _SCALE,
            direction="<=",
            frequency="QUARTERLY",
        ),
    ),
    periods=_build_periods(_SAFETECH_METRICS, ("OK", "OK", "OK", "OK", "OK")),
)

# StableFinance - healthy finance company
_STABLEFINANCE_METRICS: tuple[MetricsSeed, ...] = tuple(
    MetricsSeed(
        total_debt=80 * _SCALE,
        ebitda=45 * _SCALE,  # Leverage: 1.78x
        interest_expense=8 * _SCALE,
        current_assets=100 * _SCALE,
        current_liabilities=50 * _SCALE,
    )
    for _ in range(5)
)

STABLEFINANCE_PROFILE: DealProfile = DealProfile(
    deal=DealSeed(
        name="StableFinance Credit Line",
        borrower="StableFinance Inc",
        sector="Finance",
        region="North America",
        commitment_cents=350_000_000_00,
        currency="USD",
        maturity_iso="2027-09-30",
    ),
    covenants=(
        CovenantSeed(
            name="Leverage Ratio",
            formula="total_debt / ebitda",
            threshold_scaled=4 * _SCALE,
            direction="<=",
            frequency="QUARTERLY",
        ),
    ),
    periods=_build_periods(_STABLEFINANCE_METRICS, ("OK", "OK", "OK", "OK", "OK")),
)

# GreenHealth - healthy healthcare
_GREENHEALTH_METRICS: tuple[MetricsSeed, ...] = tuple(
    MetricsSeed(
        total_debt=60 * _SCALE,
        ebitda=40 * _SCALE,  # Leverage: 1.5x
        interest_expense=5 * _SCALE,
        current_assets=80 * _SCALE,
        current_liabilities=35 * _SCALE,
    )
    for _ in range(5)
)

GREENHEALTH_PROFILE: DealProfile = DealProfile(
    deal=DealSeed(
        name="GreenHealth Revolver",
        borrower="GreenHealth Systems",
        sector="Healthcare",
        region="Europe",
        commitment_cents=200_000_000_00,
        currency="EUR",
        maturity_iso="2028-12-31",
    ),
    covenants=(
        CovenantSeed(
            name="Leverage Ratio",
            formula="total_debt / ebitda",
            threshold_scaled=4 * _SCALE,
            direction="<=",
            frequency="QUARTERLY",
        ),
    ),
    periods=_build_periods(_GREENHEALTH_METRICS, ("OK", "OK", "OK", "OK", "OK")),
)

# PrimeTech - another safe tech company
_PRIMETECH_METRICS: tuple[MetricsSeed, ...] = tuple(
    MetricsSeed(
        total_debt=50 * _SCALE,
        ebitda=35 * _SCALE,  # Leverage: 1.43x
        interest_expense=4 * _SCALE,
        current_assets=70 * _SCALE,
        current_liabilities=30 * _SCALE,
    )
    for _ in range(5)
)

PRIMETECH_PROFILE: DealProfile = DealProfile(
    deal=DealSeed(
        name="PrimeTech Term Loan",
        borrower="PrimeTech Solutions",
        sector="Technology",
        region="Asia",
        commitment_cents=180_000_000_00,
        currency="USD",
        maturity_iso="2027-03-31",
    ),
    covenants=(
        CovenantSeed(
            name="Leverage Ratio",
            formula="total_debt / ebitda",
            threshold_scaled=4 * _SCALE,
            direction="<=",
            frequency="QUARTERLY",
        ),
    ),
    periods=_build_periods(_PRIMETECH_METRICS, ("OK", "OK", "OK", "OK", "OK")),
)


# =============================================================================
# Additional Risky Profiles (have breaches - high debt ratios)
# =============================================================================

# StruggleTech - high debt, breaching covenants
_STRUGGLETECH_METRICS: tuple[MetricsSeed, ...] = tuple(
    MetricsSeed(
        total_debt=250 * _SCALE,
        ebitda=45 * _SCALE,  # Leverage: 5.56x - BREACH
        interest_expense=25 * _SCALE,
        current_assets=60 * _SCALE,
        current_liabilities=55 * _SCALE,
    )
    for _ in range(5)
)

STRUGGLETECH_PROFILE: DealProfile = DealProfile(
    deal=DealSeed(
        name="StruggleTech Rescue Facility",
        borrower="StruggleTech Corp",
        sector="Technology",
        region="North America",
        commitment_cents=450_000_000_00,
        currency="USD",
        maturity_iso="2025-12-31",
    ),
    covenants=(
        CovenantSeed(
            name="Leverage Ratio",
            formula="total_debt / ebitda",
            threshold_scaled=4 * _SCALE,
            direction="<=",
            frequency="QUARTERLY",
        ),
    ),
    periods=_build_periods(
        _STRUGGLETECH_METRICS, ("BREACH", "BREACH", "BREACH", "BREACH", "BREACH")
    ),
)

# RiskyFinance - distressed finance company
_RISKYFINANCE_METRICS: tuple[MetricsSeed, ...] = tuple(
    MetricsSeed(
        total_debt=300 * _SCALE,
        ebitda=50 * _SCALE,  # Leverage: 6.0x - BREACH
        interest_expense=30 * _SCALE,
        current_assets=55 * _SCALE,
        current_liabilities=60 * _SCALE,
    )
    for _ in range(5)
)

RISKYFINANCE_PROFILE: DealProfile = DealProfile(
    deal=DealSeed(
        name="RiskyFinance DIP Loan",
        borrower="RiskyFinance Holdings",
        sector="Finance",
        region="Europe",
        commitment_cents=600_000_000_00,
        currency="EUR",
        maturity_iso="2025-06-30",
    ),
    covenants=(
        CovenantSeed(
            name="Leverage Ratio",
            formula="total_debt / ebitda",
            threshold_scaled=4 * _SCALE,
            direction="<=",
            frequency="QUARTERLY",
        ),
    ),
    periods=_build_periods(
        _RISKYFINANCE_METRICS, ("BREACH", "BREACH", "BREACH", "BREACH", "BREACH")
    ),
)

# CrisisHealth - distressed healthcare
_CRISISHEALTH_METRICS: tuple[MetricsSeed, ...] = tuple(
    MetricsSeed(
        total_debt=220 * _SCALE,
        ebitda=40 * _SCALE,  # Leverage: 5.5x - BREACH
        interest_expense=22 * _SCALE,
        current_assets=45 * _SCALE,
        current_liabilities=50 * _SCALE,
    )
    for _ in range(5)
)

CRISISHEALTH_PROFILE: DealProfile = DealProfile(
    deal=DealSeed(
        name="CrisisHealth Emergency Credit",
        borrower="CrisisHealth Network",
        sector="Healthcare",
        region="Asia",
        commitment_cents=350_000_000_00,
        currency="USD",
        maturity_iso="2025-09-30",
    ),
    covenants=(
        CovenantSeed(
            name="Leverage Ratio",
            formula="total_debt / ebitda",
            threshold_scaled=4 * _SCALE,
            direction="<=",
            frequency="QUARTERLY",
        ),
    ),
    periods=_build_periods(
        _CRISISHEALTH_METRICS, ("BREACH", "BREACH", "BREACH", "BREACH", "BREACH")
    ),
)

# FailingTech - another distressed tech
_FAILINGTECH_METRICS: tuple[MetricsSeed, ...] = tuple(
    MetricsSeed(
        total_debt=280 * _SCALE,
        ebitda=48 * _SCALE,  # Leverage: 5.83x - BREACH
        interest_expense=28 * _SCALE,
        current_assets=50 * _SCALE,
        current_liabilities=52 * _SCALE,
    )
    for _ in range(5)
)

FAILINGTECH_PROFILE: DealProfile = DealProfile(
    deal=DealSeed(
        name="FailingTech Bridge Loan",
        borrower="FailingTech Industries",
        sector="Technology",
        region="Europe",
        commitment_cents=500_000_000_00,
        currency="EUR",
        maturity_iso="2025-03-31",
    ),
    covenants=(
        CovenantSeed(
            name="Leverage Ratio",
            formula="total_debt / ebitda",
            threshold_scaled=4 * _SCALE,
            direction="<=",
            frequency="QUARTERLY",
        ),
    ),
    periods=_build_periods(
        _FAILINGTECH_METRICS, ("BREACH", "BREACH", "BREACH", "BREACH", "BREACH")
    ),
)


# All profiles for seeding (12 total: 6 safe, 6 risky)
ALL_PROFILES: tuple[DealProfile, ...] = (
    # Original 4 (2 safe, 2 risky)
    TECHCORP_PROFILE,
    FINANCEGROUP_PROFILE,
    HEALTHCARE_PROFILE,
    CLOUDTECH_PROFILE,
    # Additional safe profiles
    SAFETECH_PROFILE,
    STABLEFINANCE_PROFILE,
    GREENHEALTH_PROFILE,
    PRIMETECH_PROFILE,
    # Additional risky profiles
    STRUGGLETECH_PROFILE,
    RISKYFINANCE_PROFILE,
    CRISISHEALTH_PROFILE,
    FAILINGTECH_PROFILE,
)


__all__ = [
    "ALL_PROFILES",
    "CLOUDTECH_PROFILE",
    "CRISISHEALTH_PROFILE",
    "FAILINGTECH_PROFILE",
    "FINANCEGROUP_PROFILE",
    "GREENHEALTH_PROFILE",
    "HEALTHCARE_PROFILE",
    "PRIMETECH_PROFILE",
    "RISKYFINANCE_PROFILE",
    "SAFETECH_PROFILE",
    "STABLEFINANCE_PROFILE",
    "STRUGGLETECH_PROFILE",
    "TECHCORP_PROFILE",
    "CovenantSeed",
    "DealProfile",
    "DealSeed",
    "MetricsSeed",
    "PeriodSeed",
]
