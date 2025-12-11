"""Synthetic data generator for covenant-radar-api training.

Generates hundreds of deals with realistic financial variations
for training XGBoost breach prediction models.

All functions are pure - no IO, deterministic with seed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.random import Generator

from .profiles import (
    CovenantSeed,
    DealProfile,
    DealSeed,
    MetricsSeed,
    PeriodSeed,
)

# =============================================================================
# Constants
# =============================================================================

_SCALE: int = 1_000_000

_SECTORS: tuple[Literal["Technology", "Finance", "Healthcare"], ...] = (
    "Technology",
    "Finance",
    "Healthcare",
)

_REGIONS: tuple[Literal["North America", "Europe", "Asia"], ...] = (
    "North America",
    "Europe",
    "Asia",
)

_CURRENCIES: tuple[str, ...] = ("USD", "EUR", "GBP", "JPY", "CHF")

_COMPANY_PREFIXES: tuple[str, ...] = (
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Omega",
    "Prime",
    "Core",
    "Nova",
    "Apex",
    "Peak",
    "Summit",
    "Horizon",
    "Vertex",
    "Zenith",
    "Atlas",
    "Titan",
    "Eagle",
    "Phoenix",
    "Sterling",
    "Global",
    "United",
    "Pacific",
    "Atlantic",
    "Nordic",
    "Alpine",
)

_COMPANY_SUFFIXES: tuple[str, ...] = (
    "Corp",
    "Inc",
    "LLC",
    "Holdings",
    "Group",
    "Partners",
    "Systems",
    "Solutions",
    "Industries",
    "Ventures",
    "Capital",
    "Finance",
    "Tech",
    "Health",
    "Medical",
)

_LOAN_TYPES: tuple[str, ...] = (
    "Senior Credit Facility",
    "Term Loan B",
    "Revolving Credit",
    "Asset Based Loan",
    "Bridge Loan",
    "Mezzanine Facility",
    "Unitranche",
    "Second Lien",
    "DIP Facility",
    "Working Capital",
)

# Period dates for 5 quarters back
_PERIOD_DATES: tuple[tuple[str, str], ...] = (
    ("2024-01-01", "2024-03-31"),
    ("2023-10-01", "2023-12-31"),
    ("2023-07-01", "2023-09-30"),
    ("2023-04-01", "2023-06-30"),
    ("2023-01-01", "2023-03-31"),
)


# =============================================================================
# Risk Profile Parameters
# =============================================================================


class RiskParams:
    """Parameters for generating different risk profiles."""

    # Healthy companies: leverage 1.0-2.5x
    HEALTHY_LEVERAGE_MIN = 1.0
    HEALTHY_LEVERAGE_MAX = 2.5

    # Stressed companies: leverage 3.0-4.0x (near threshold)
    STRESSED_LEVERAGE_MIN = 3.0
    STRESSED_LEVERAGE_MAX = 4.0

    # Distressed companies: leverage 4.5-8.0x (above threshold)
    DISTRESSED_LEVERAGE_MIN = 4.5
    DISTRESSED_LEVERAGE_MAX = 8.0

    # Base EBITDA ranges (in millions, before scaling)
    EBITDA_MIN = 20
    EBITDA_MAX = 100

    # Interest coverage ratios
    HEALTHY_COVERAGE_MIN = 4.0
    HEALTHY_COVERAGE_MAX = 10.0
    STRESSED_COVERAGE_MIN = 2.0
    STRESSED_COVERAGE_MAX = 3.5
    DISTRESSED_COVERAGE_MIN = 1.0
    DISTRESSED_COVERAGE_MAX = 2.0

    # Current ratio (current_assets / current_liabilities)
    HEALTHY_CURRENT_MIN = 1.5
    HEALTHY_CURRENT_MAX = 3.0
    STRESSED_CURRENT_MIN = 1.1
    STRESSED_CURRENT_MAX = 1.4
    DISTRESSED_CURRENT_MIN = 0.6
    DISTRESSED_CURRENT_MAX = 1.0


# =============================================================================
# Generator Functions
# =============================================================================


def _generate_company_name(rng: Generator, idx: int) -> tuple[str, str]:
    """Generate a unique company name and borrower name.

    Args:
        rng: Numpy random generator.
        idx: Index for uniqueness.

    Returns:
        Tuple of (deal_name, borrower_name).
    """
    prefix = _COMPANY_PREFIXES[rng.integers(0, len(_COMPANY_PREFIXES))]
    suffix = _COMPANY_SUFFIXES[rng.integers(0, len(_COMPANY_SUFFIXES))]
    loan_type = _LOAN_TYPES[rng.integers(0, len(_LOAN_TYPES))]

    borrower = f"{prefix}{suffix} {idx}"
    deal_name = f"{prefix}{suffix} {loan_type}"

    return deal_name, borrower


def _generate_metrics_for_profile(
    rng: Generator,
    risk_profile: Literal["healthy", "stressed", "distressed"],
    base_ebitda: float,
) -> MetricsSeed:
    """Generate a single period's metrics based on risk profile.

    Args:
        rng: Numpy random generator.
        risk_profile: One of 'healthy', 'stressed', 'distressed'.
        base_ebitda: Base EBITDA value in millions.

    Returns:
        MetricsSeed with all financial metrics.
    """
    # Add some random variation to EBITDA (+/- 15%)
    ebitda_variation = 1.0 + rng.uniform(-0.15, 0.15)
    ebitda = int(base_ebitda * ebitda_variation * _SCALE)

    # Calculate debt based on target leverage
    if risk_profile == "healthy":
        leverage = rng.uniform(RiskParams.HEALTHY_LEVERAGE_MIN, RiskParams.HEALTHY_LEVERAGE_MAX)
        coverage = rng.uniform(RiskParams.HEALTHY_COVERAGE_MIN, RiskParams.HEALTHY_COVERAGE_MAX)
        current = rng.uniform(RiskParams.HEALTHY_CURRENT_MIN, RiskParams.HEALTHY_CURRENT_MAX)
    elif risk_profile == "stressed":
        leverage = rng.uniform(RiskParams.STRESSED_LEVERAGE_MIN, RiskParams.STRESSED_LEVERAGE_MAX)
        coverage = rng.uniform(RiskParams.STRESSED_COVERAGE_MIN, RiskParams.STRESSED_COVERAGE_MAX)
        current = rng.uniform(RiskParams.STRESSED_CURRENT_MIN, RiskParams.STRESSED_CURRENT_MAX)
    else:  # distressed
        leverage = rng.uniform(
            RiskParams.DISTRESSED_LEVERAGE_MIN, RiskParams.DISTRESSED_LEVERAGE_MAX
        )
        coverage = rng.uniform(
            RiskParams.DISTRESSED_COVERAGE_MIN, RiskParams.DISTRESSED_COVERAGE_MAX
        )
        current = rng.uniform(RiskParams.DISTRESSED_CURRENT_MIN, RiskParams.DISTRESSED_CURRENT_MAX)

    total_debt = int(leverage * ebitda)
    interest_expense = int(ebitda / coverage)

    # Current assets/liabilities (scaled to debt level)
    current_liabilities = int(total_debt * rng.uniform(0.15, 0.25))
    current_assets = int(current_liabilities * current)

    return MetricsSeed(
        total_debt=total_debt,
        ebitda=ebitda,
        interest_expense=interest_expense,
        current_assets=current_assets,
        current_liabilities=current_liabilities,
    )


def _determine_status(
    metrics: MetricsSeed,
    leverage_threshold: float = 4.0,
    tolerance: float = 0.1,
) -> Literal["OK", "NEAR_BREACH", "BREACH"]:
    """Determine covenant status based on leverage ratio.

    Args:
        metrics: Financial metrics for the period.
        leverage_threshold: Max allowed leverage (default 4.0x).
        tolerance: Fraction of threshold for near-breach zone.

    Returns:
        Covenant status: OK, NEAR_BREACH, or BREACH.
    """
    if metrics["ebitda"] <= 0:
        return "BREACH"

    leverage = metrics["total_debt"] / metrics["ebitda"]
    near_breach_zone = leverage_threshold * (1 - tolerance)

    if leverage > leverage_threshold:
        return "BREACH"
    if leverage >= near_breach_zone:
        return "NEAR_BREACH"
    return "OK"


def _generate_periods_for_profile(
    rng: Generator,
    risk_profile: Literal["healthy", "stressed", "distressed"],
    base_ebitda: float,
    trend: Literal["improving", "stable", "deteriorating"],
) -> tuple[PeriodSeed, ...]:
    """Generate 5 quarters of periods with trend.

    Args:
        rng: Numpy random generator.
        risk_profile: Base risk profile.
        base_ebitda: Starting EBITDA value.
        trend: Financial trend direction.

    Returns:
        Tuple of 5 PeriodSeed objects.
    """
    periods: list[PeriodSeed] = []

    # Compute period profiles based on trend and risk profile
    period_profiles: tuple[
        Literal["healthy", "stressed", "distressed"],
        Literal["healthy", "stressed", "distressed"],
        Literal["healthy", "stressed", "distressed"],
        Literal["healthy", "stressed", "distressed"],
        Literal["healthy", "stressed", "distressed"],
    ]
    if trend == "improving":
        if risk_profile == "healthy":
            period_profiles = ("healthy", "healthy", "healthy", "healthy", "healthy")
        elif risk_profile == "stressed":
            period_profiles = ("stressed", "stressed", "stressed", "healthy", "healthy")
        else:  # distressed
            period_profiles = ("distressed", "distressed", "stressed", "stressed", "healthy")
    elif trend == "stable":
        period_profiles = (risk_profile, risk_profile, risk_profile, risk_profile, risk_profile)
    else:  # deteriorating
        if risk_profile == "healthy":
            period_profiles = ("healthy", "healthy", "stressed", "stressed", "distressed")
        elif risk_profile == "stressed":
            period_profiles = ("stressed", "stressed", "distressed", "distressed", "distressed")
        else:  # distressed
            period_profiles = (
                "distressed",
                "distressed",
                "distressed",
                "distressed",
                "distressed",
            )

    for i, (dates, period_risk) in enumerate(zip(_PERIOD_DATES, period_profiles, strict=True)):
        # Adjust EBITDA based on time (older periods have different base)
        time_factor = 1.0 + (i * 0.02 * (1 if trend == "improving" else -1))
        adjusted_ebitda = base_ebitda * time_factor

        metrics = _generate_metrics_for_profile(rng, period_risk, adjusted_ebitda)
        status = _determine_status(metrics)

        periods.append(
            PeriodSeed(
                start_iso=dates[0],
                end_iso=dates[1],
                metrics=metrics,
                expected_status=status,
            )
        )

    return tuple(periods)


def generate_synthetic_profile(
    rng: Generator,
    idx: int,
    risk_profile: Literal["healthy", "stressed", "distressed"],
    trend: Literal["improving", "stable", "deteriorating"],
) -> DealProfile:
    """Generate a complete synthetic deal profile.

    Args:
        rng: Numpy random generator.
        idx: Unique index for naming.
        risk_profile: Overall risk level.
        trend: Financial trajectory.

    Returns:
        Complete DealProfile with deal, covenants, and periods.
    """
    deal_name, borrower = _generate_company_name(rng, idx)
    sector = _SECTORS[rng.integers(0, len(_SECTORS))]
    region = _REGIONS[rng.integers(0, len(_REGIONS))]
    currency = _CURRENCIES[rng.integers(0, len(_CURRENCIES))]

    # Commitment amount between $50M and $1B
    commitment = int(rng.integers(50, 1000)) * 1_000_000_00

    # Maturity date 1-5 years out
    years_out = rng.integers(1, 6)
    maturity = f"202{4 + years_out}-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}"

    # Base EBITDA for this company
    base_ebitda = float(rng.integers(RiskParams.EBITDA_MIN, RiskParams.EBITDA_MAX + 1))

    deal = DealSeed(
        name=deal_name,
        borrower=borrower,
        sector=sector,
        region=region,
        commitment_cents=commitment,
        currency=currency,
        maturity_iso=maturity,
    )

    # Standard leverage covenant
    covenants: tuple[CovenantSeed, ...] = (
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
    )

    periods = _generate_periods_for_profile(rng, risk_profile, base_ebitda, trend)

    return DealProfile(
        deal=deal,
        covenants=covenants,
        periods=periods,
    )


def generate_synthetic_profiles(
    n_deals: int,
    random_seed: int = 42,
    healthy_ratio: float = 0.5,
    stressed_ratio: float = 0.25,
) -> tuple[DealProfile, ...]:
    """Generate a batch of synthetic deal profiles.

    Creates a realistic distribution of companies across risk profiles
    and trends for training breach prediction models.

    Args:
        n_deals: Total number of deals to generate.
        random_seed: Seed for reproducibility.
        healthy_ratio: Fraction of healthy companies (default 0.5).
        stressed_ratio: Fraction of stressed companies (default 0.25).

    Returns:
        Tuple of DealProfile objects.
    """
    rng = np.random.default_rng(random_seed)

    # Calculate counts
    n_healthy = int(n_deals * healthy_ratio)
    n_stressed = int(n_deals * stressed_ratio)
    n_distressed = n_deals - n_healthy - n_stressed

    profiles: list[DealProfile] = []
    idx = 1

    # Generate healthy companies
    trends: tuple[Literal["improving", "stable", "deteriorating"], ...] = (
        "improving",
        "stable",
        "deteriorating",
    )

    for i in range(n_healthy):
        trend = trends[i % 3]
        profiles.append(generate_synthetic_profile(rng, idx, "healthy", trend))
        idx += 1

    # Generate stressed companies
    for i in range(n_stressed):
        trend = trends[i % 3]
        profiles.append(generate_synthetic_profile(rng, idx, "stressed", trend))
        idx += 1

    # Generate distressed companies
    for i in range(n_distressed):
        trend = trends[i % 3]
        profiles.append(generate_synthetic_profile(rng, idx, "distressed", trend))
        idx += 1

    # Shuffle the list using rng-generated order
    indices: list[int] = list(range(len(profiles)))
    rng.shuffle(indices)
    shuffled: list[DealProfile] = [profiles[i] for i in indices]

    return tuple(shuffled)


def count_breach_labels(profiles: tuple[DealProfile, ...]) -> dict[str, int | float]:
    """Count breach labels across all profiles for dataset statistics.

    Args:
        profiles: Generated profiles to analyze.

    Returns:
        Dict with counts: n_deals, n_breach, n_near_breach, n_ok, n_periods, breach_rate.
    """
    n_breach = 0
    n_near_breach = 0
    n_ok = 0

    for profile in profiles:
        for period in profile["periods"]:
            status = period["expected_status"]
            if status == "BREACH":
                n_breach += 1
            elif status == "NEAR_BREACH":
                n_near_breach += 1
            else:
                n_ok += 1

    total = n_breach + n_near_breach + n_ok
    breach_rate: float = n_breach / total if total > 0 else 0.0

    result: dict[str, int | float] = {
        "n_deals": len(profiles),
        "n_periods": total,
        "n_breach": n_breach,
        "n_near_breach": n_near_breach,
        "n_ok": n_ok,
        "breach_rate": breach_rate,
    }
    return result


__all__ = [
    "RiskParams",
    "count_breach_labels",
    "generate_synthetic_profile",
    "generate_synthetic_profiles",
]
