"""Tests for seeding runner module."""

from __future__ import annotations

import pytest
from covenant_persistence.testing import InMemoryConnection, InMemoryStore

from covenant_radar_api.seeding import _test_hooks
from covenant_radar_api.seeding.profiles import (
    ALL_PROFILES,
    FINANCEGROUP_PROFILE,
    TECHCORP_PROFILE,
    DealProfile,
)
from covenant_radar_api.seeding.runner import (
    seed_database,
    seed_database_with_defaults,
    seed_database_with_synthetic,
)

# =============================================================================
# Fixtures
# =============================================================================


def _make_in_memory_store() -> InMemoryStore:
    """Create fresh in-memory store."""
    return InMemoryStore()


def _make_in_memory_conn(in_memory_store: InMemoryStore) -> InMemoryConnection:
    """Create in-memory connection."""
    return InMemoryConnection(in_memory_store)


in_memory_store = pytest.fixture(_make_in_memory_store)
in_memory_conn = pytest.fixture(_make_in_memory_conn)


# =============================================================================
# Tests
# =============================================================================


class TestSeedDatabase:
    """Tests for seed_database function."""

    def test_seeds_single_profile(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test seeding with single profile creates correct counts."""
        # Use predictable UUIDs
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        profiles: tuple[DealProfile, ...] = (TECHCORP_PROFILE,)
        result = seed_database(in_memory_conn, profiles)

        assert result["deals_created"] == 1
        # TechCorp has 2 covenants
        assert result["covenants_created"] == 2
        # 5 periods * 5 metrics = 25 measurements
        assert result["measurements_created"] == 25
        # 5 periods * 2 covenants = 10 results
        assert result["results_created"] == 10

    def test_seeds_multiple_profiles(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test seeding with multiple profiles."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        profiles: tuple[DealProfile, ...] = (TECHCORP_PROFILE, FINANCEGROUP_PROFILE)
        result = seed_database(in_memory_conn, profiles)

        assert result["deals_created"] == 2
        # TechCorp: 2, FinanceGroup: 1
        assert result["covenants_created"] == 3
        # 2 deals * 5 periods * 5 metrics = 50
        assert result["measurements_created"] == 50

    def test_seeds_empty_profiles(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test seeding with empty profiles tuple."""
        profiles: tuple[DealProfile, ...] = ()
        result = seed_database(in_memory_conn, profiles)

        assert result["deals_created"] == 0
        assert result["covenants_created"] == 0
        assert result["measurements_created"] == 0
        assert result["results_created"] == 0

    def test_creates_deals_in_store(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test deals are created in store."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        profiles: tuple[DealProfile, ...] = (TECHCORP_PROFILE,)
        seed_database(in_memory_conn, profiles)

        # First UUID is the deal
        assert "uuid-1" in in_memory_store.deals
        deal = in_memory_store.deals["uuid-1"]
        assert deal["name"] == "TechCorp Senior Credit Facility"
        assert deal["sector"] == "Technology"

    def test_creates_covenants_in_store(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test covenants are created in store."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        profiles: tuple[DealProfile, ...] = (TECHCORP_PROFILE,)
        seed_database(in_memory_conn, profiles)

        # First UUID is deal, next 2 are covenants
        assert "uuid-2" in in_memory_store.covenants
        assert "uuid-3" in in_memory_store.covenants

        cov = in_memory_store.covenants["uuid-2"]
        assert cov["name"] == "Leverage Ratio"

    def test_creates_measurements_in_store(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test measurements are created in store."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        profiles: tuple[DealProfile, ...] = (TECHCORP_PROFILE,)
        seed_database(in_memory_conn, profiles)

        # 5 periods * 5 metrics = 25
        assert len(in_memory_store.measurements) == 25

        # Check first measurement has deal_id = uuid-1 (the deal)
        m = in_memory_store.measurements[0]
        assert m["deal_id"]["value"] == "uuid-1"

    def test_creates_results_in_store(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test covenant results are created in store."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        profiles: tuple[DealProfile, ...] = (TECHCORP_PROFILE,)
        seed_database(in_memory_conn, profiles)

        # 5 periods * 2 covenants = 10
        assert len(in_memory_store.covenant_results) == 10

    def test_results_have_correct_status(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test results have expected statuses from profile."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        profiles: tuple[DealProfile, ...] = (FINANCEGROUP_PROFILE,)
        seed_database(in_memory_conn, profiles)

        # FinanceGroup: BREACH, NEAR_BREACH, OK, OK, OK
        statuses = [r["status"] for r in in_memory_store.covenant_results]
        assert "BREACH" in statuses
        assert "NEAR_BREACH" in statuses
        assert "OK" in statuses


class TestSeedDatabaseWithDefaults:
    """Tests for seed_database_with_defaults function."""

    def test_seeds_all_default_profiles(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test seeding with all default profiles."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        result = seed_database_with_defaults(in_memory_conn)

        assert result["deals_created"] == len(ALL_PROFILES)
        assert result["deals_created"] == 12

    def test_creates_all_deals(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test all deals are created in store."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        seed_database_with_defaults(in_memory_conn)

        assert len(in_memory_store.deals) == 12

    def test_total_covenants_created(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test correct total covenants created."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        result = seed_database_with_defaults(in_memory_conn)

        # Original 4: TechCorp (2), FinanceGroup (1), HealthCare (1), CloudTech (1) = 5
        # New 8: SafeTech (1), StableFinance (1), GreenHealth (1), PrimeTech (1),
        #        StruggleTech (1), RiskyFinance (1), CrisisHealth (1), FailingTech (1) = 8
        # Total = 13
        assert result["covenants_created"] == 13

    def test_total_measurements_created(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test correct total measurements created."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        result = seed_database_with_defaults(in_memory_conn)

        # 12 deals * 5 periods * 5 metrics = 300
        assert result["measurements_created"] == 300

    def test_total_results_created(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test correct total results created."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        result = seed_database_with_defaults(in_memory_conn)

        # Original 4 profiles: TechCorp=10, FinanceGroup=5, HealthCare=5, CloudTech=5 = 25
        # New 8 profiles: each has 1 covenant * 5 periods = 5 results per profile = 40
        # Total: 25 + 40 = 65
        assert result["results_created"] == 65


class TestSeedResultTypedDict:
    """Tests for SeedResult TypedDict structure."""

    def test_result_has_all_fields(
        self,
        in_memory_conn: InMemoryConnection,
    ) -> None:
        """Test SeedResult has all required fields."""
        # Empty profiles so no UUIDs needed
        profiles: tuple[DealProfile, ...] = ()
        result = seed_database(in_memory_conn, profiles)

        assert "deals_created" in result
        assert "covenants_created" in result
        assert "measurements_created" in result
        assert "results_created" in result

    def test_result_values_can_do_arithmetic(
        self,
        in_memory_conn: InMemoryConnection,
    ) -> None:
        """Test SeedResult values support integer arithmetic."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        profiles: tuple[DealProfile, ...] = (TECHCORP_PROFILE,)
        result = seed_database(in_memory_conn, profiles)

        # Verify integer behavior by doing arithmetic operations
        total = (
            result["deals_created"]
            + result["covenants_created"]
            + result["measurements_created"]
            + result["results_created"]
        )
        # TechCorp: 1 deal + 2 covenants + 25 measurements + 10 results = 38
        assert total == 38


class TestSeedDatabaseWithSynthetic:
    """Tests for seed_database_with_synthetic function."""

    def test_seeds_synthetic_profiles(
        self,
        in_memory_conn: InMemoryConnection,
        in_memory_store: InMemoryStore,
    ) -> None:
        """Test seeding with synthetic profiles creates entities."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        result = seed_database_with_synthetic(
            in_memory_conn,
            n_deals=5,
            random_seed=42,
            healthy_ratio=0.6,
            stressed_ratio=0.2,
        )

        assert result["deals_created"] == 5
        # Each deal has 2 covenants
        assert result["covenants_created"] == 10
        # Each deal has 5 periods * 5 metrics = 25 measurements
        assert result["measurements_created"] == 125
        # Each deal has 5 periods * 2 covenants = 10 results
        assert result["results_created"] == 50

    def test_seeds_with_default_ratios(
        self,
        in_memory_conn: InMemoryConnection,
    ) -> None:
        """Test seeding with default ratios."""
        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        _test_hooks.uuid_generator = fake_uuid

        result = seed_database_with_synthetic(
            in_memory_conn,
            n_deals=10,
            random_seed=42,
        )

        assert result["deals_created"] == 10
