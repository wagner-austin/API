"""Runner for seeding database with demo data.

This module orchestrates the seeding process, using generators to
create domain objects and repositories to persist them.
"""

from __future__ import annotations

from typing import TypedDict

from covenant_persistence import (
    ConnectionProtocol,
    CovenantRepository,
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
    PostgresCovenantRepository,
    PostgresCovenantResultRepository,
    PostgresDealRepository,
    PostgresMeasurementRepository,
)

from . import _test_hooks
from .generators import (
    generate_covenant,
    generate_covenant_result,
    generate_deal,
    generate_measurements,
)
from .profiles import DealProfile


class SeedResult(TypedDict, total=True):
    """Result of seeding operation."""

    deals_created: int
    covenants_created: int
    measurements_created: int
    results_created: int


def _seed_deal(
    profile: DealProfile,
    deal_repo: DealRepository,
    cov_repo: CovenantRepository,
    meas_repo: MeasurementRepository,
    result_repo: CovenantResultRepository,
) -> tuple[int, int, int, int]:
    """Seed a single deal profile.

    Returns:
        Tuple of (deals, covenants, measurements, results) created.
    """
    deal_id = _test_hooks.uuid_generator()
    deal = generate_deal(deal_id, profile["deal"])
    deal_repo.create(deal)

    cov_ids: list[str] = []
    for cov_seed in profile["covenants"]:
        cov_id = _test_hooks.uuid_generator()
        covenant = generate_covenant(cov_id, deal_id, cov_seed)
        cov_repo.create(covenant)
        cov_ids.append(cov_id)

    measurements_count = 0
    results_count = 0

    for period in profile["periods"]:
        measurements = generate_measurements(
            deal_id,
            period["start_iso"],
            period["end_iso"],
            period["metrics"],
        )
        measurements_count += meas_repo.add_many(list(measurements))

        for cov_id in cov_ids:
            result = generate_covenant_result(cov_id, period)
            result_repo.save(result)
            results_count += 1

    return (1, len(cov_ids), measurements_count, results_count)


def seed_database(
    conn: ConnectionProtocol,
    profiles: tuple[DealProfile, ...],
) -> SeedResult:
    """Seed the database with demo data from profiles.

    Args:
        conn: Database connection to use.
        profiles: Tuple of DealProfile to seed.

    Returns:
        SeedResult with counts of created entities.
    """
    deal_repo: DealRepository = PostgresDealRepository(conn)
    cov_repo: CovenantRepository = PostgresCovenantRepository(conn)
    meas_repo: MeasurementRepository = PostgresMeasurementRepository(conn)
    result_repo: CovenantResultRepository = PostgresCovenantResultRepository(conn)

    total_deals = 0
    total_covenants = 0
    total_measurements = 0
    total_results = 0

    for profile in profiles:
        deals, covenants, measurements, results = _seed_deal(
            profile,
            deal_repo,
            cov_repo,
            meas_repo,
            result_repo,
        )
        total_deals += deals
        total_covenants += covenants
        total_measurements += measurements
        total_results += results

    conn.commit()

    return SeedResult(
        deals_created=total_deals,
        covenants_created=total_covenants,
        measurements_created=total_measurements,
        results_created=total_results,
    )


def seed_database_with_defaults(conn: ConnectionProtocol) -> SeedResult:
    """Seed the database with all default profiles.

    Args:
        conn: Database connection to use.

    Returns:
        SeedResult with counts of created entities.
    """
    from .profiles import ALL_PROFILES

    return seed_database(conn, ALL_PROFILES)


def seed_database_with_synthetic(
    conn: ConnectionProtocol,
    n_deals: int = 200,
    random_seed: int = 42,
    healthy_ratio: float = 0.5,
    stressed_ratio: float = 0.25,
) -> SeedResult:
    """Seed the database with synthetic generated profiles.

    Generates a realistic distribution of deals across risk profiles
    for training breach prediction models.

    Args:
        conn: Database connection to use.
        n_deals: Number of deals to generate (default 200).
        random_seed: Random seed for reproducibility.
        healthy_ratio: Fraction of healthy companies (default 0.5).
        stressed_ratio: Fraction of stressed companies (default 0.25).

    Returns:
        SeedResult with counts of created entities.
    """
    from .synthetic import generate_synthetic_profiles

    profiles = generate_synthetic_profiles(
        n_deals=n_deals,
        random_seed=random_seed,
        healthy_ratio=healthy_ratio,
        stressed_ratio=stressed_ratio,
    )

    return seed_database(conn, profiles)


__all__ = [
    "SeedResult",
    "seed_database",
    "seed_database_with_defaults",
    "seed_database_with_synthetic",
]
