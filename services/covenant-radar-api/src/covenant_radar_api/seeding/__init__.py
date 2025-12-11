"""Seeding module for covenant-radar-api.

Provides functionality to populate the database with demo data for
testing and demonstration purposes.

Usage:
    from covenant_radar_api.seeding import seed_database, ALL_PROFILES

    conn = psycopg.connect(dsn)
    result = seed_database(conn, ALL_PROFILES)

    # For synthetic data (hundreds of deals):
    from covenant_radar_api.seeding import generate_synthetic_profiles

    profiles = generate_synthetic_profiles(n_deals=200, random_seed=42)
    result = seed_database(conn, profiles)
"""

from __future__ import annotations

from .profiles import (
    ALL_PROFILES,
    CLOUDTECH_PROFILE,
    FINANCEGROUP_PROFILE,
    HEALTHCARE_PROFILE,
    TECHCORP_PROFILE,
    CovenantSeed,
    DealProfile,
    DealSeed,
    MetricsSeed,
    PeriodSeed,
)
from .runner import (
    SeedResult,
    seed_database,
    seed_database_with_defaults,
    seed_database_with_synthetic,
)
from .synthetic import (
    RiskParams,
    count_breach_labels,
    generate_synthetic_profile,
    generate_synthetic_profiles,
)

__all__ = [
    "ALL_PROFILES",
    "CLOUDTECH_PROFILE",
    "FINANCEGROUP_PROFILE",
    "HEALTHCARE_PROFILE",
    "TECHCORP_PROFILE",
    "CovenantSeed",
    "DealProfile",
    "DealSeed",
    "MetricsSeed",
    "PeriodSeed",
    "RiskParams",
    "SeedResult",
    "count_breach_labels",
    "generate_synthetic_profile",
    "generate_synthetic_profiles",
    "seed_database",
    "seed_database_with_defaults",
    "seed_database_with_synthetic",
]
