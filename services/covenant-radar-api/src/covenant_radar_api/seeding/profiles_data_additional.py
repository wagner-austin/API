"""Seed data: additional safe and risky profiles, and the combined roster."""

from __future__ import annotations

from covenant_radar_api.seeding.profiles import (
    _SCALE,
    CovenantSeed,
    DealProfile,
    DealSeed,
    MetricsSeed,
    _build_periods,
)
from covenant_radar_api.seeding.profiles_data import (
    CLOUDTECH_PROFILE,
    FINANCEGROUP_PROFILE,
    HEALTHCARE_PROFILE,
    TECHCORP_PROFILE,
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
    "CRISISHEALTH_PROFILE",
    "FAILINGTECH_PROFILE",
    "GREENHEALTH_PROFILE",
    "PRIMETECH_PROFILE",
    "RISKYFINANCE_PROFILE",
    "SAFETECH_PROFILE",
    "STABLEFINANCE_PROFILE",
    "STRUGGLETECH_PROFILE",
]
