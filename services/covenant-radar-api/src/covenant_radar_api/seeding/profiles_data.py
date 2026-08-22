"""Seed data: the four detailed core deal profiles."""

from __future__ import annotations

from covenant_radar_api.seeding.profiles import (
    _SCALE,
    CovenantSeed,
    DealProfile,
    DealSeed,
    MetricsSeed,
    _build_periods,
)

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


__all__ = [
    "CLOUDTECH_PROFILE",
    "FINANCEGROUP_PROFILE",
    "HEALTHCARE_PROFILE",
    "TECHCORP_PROFILE",
]
