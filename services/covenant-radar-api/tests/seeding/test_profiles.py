"""Tests for seeding profiles module."""

from __future__ import annotations

from covenant_radar_api.seeding.profiles import (
    ALL_PROFILES,
    CLOUDTECH_PROFILE,
    FINANCEGROUP_PROFILE,
    HEALTHCARE_PROFILE,
    TECHCORP_PROFILE,
)


class TestProfiles:
    """Tests for seed profile definitions."""

    def test_all_profiles_contains_twelve_profiles(self) -> None:
        """Test ALL_PROFILES has exactly 12 profiles."""
        assert len(ALL_PROFILES) == 12

    def test_all_profiles_tuple_contains_expected_profiles(self) -> None:
        """Test ALL_PROFILES contains all named profiles."""
        assert TECHCORP_PROFILE in ALL_PROFILES
        assert FINANCEGROUP_PROFILE in ALL_PROFILES
        assert HEALTHCARE_PROFILE in ALL_PROFILES
        assert CLOUDTECH_PROFILE in ALL_PROFILES


class TestTechCorpProfile:
    """Tests for TechCorp profile."""

    def test_deal_has_correct_sector(self) -> None:
        """Test TechCorp is in Technology sector."""
        assert TECHCORP_PROFILE["deal"]["sector"] == "Technology"

    def test_deal_has_correct_region(self) -> None:
        """Test TechCorp is in North America."""
        assert TECHCORP_PROFILE["deal"]["region"] == "North America"

    def test_has_two_covenants(self) -> None:
        """Test TechCorp has 2 covenants."""
        assert len(TECHCORP_PROFILE["covenants"]) == 2

    def test_has_five_periods(self) -> None:
        """Test TechCorp has 5 periods."""
        assert len(TECHCORP_PROFILE["periods"]) == 5

    def test_all_periods_are_ok(self) -> None:
        """Test all TechCorp periods have OK status."""
        for period in TECHCORP_PROFILE["periods"]:
            assert period["expected_status"] == "OK"

    def test_leverage_covenant_threshold(self) -> None:
        """Test leverage covenant has 4x threshold."""
        leverage_cov = TECHCORP_PROFILE["covenants"][0]
        assert leverage_cov["name"] == "Leverage Ratio"
        assert leverage_cov["threshold_scaled"] == 4_000_000
        assert leverage_cov["direction"] == "<="

    def test_interest_coverage_covenant_threshold(self) -> None:
        """Test interest coverage covenant has 2x threshold."""
        coverage_cov = TECHCORP_PROFILE["covenants"][1]
        assert coverage_cov["name"] == "Interest Coverage"
        assert coverage_cov["threshold_scaled"] == 2_000_000
        assert coverage_cov["direction"] == ">="


class TestFinanceGroupProfile:
    """Tests for FinanceGroup profile."""

    def test_deal_has_correct_sector(self) -> None:
        """Test FinanceGroup is in Finance sector."""
        assert FINANCEGROUP_PROFILE["deal"]["sector"] == "Finance"

    def test_deal_has_correct_region(self) -> None:
        """Test FinanceGroup is in Europe."""
        assert FINANCEGROUP_PROFILE["deal"]["region"] == "Europe"

    def test_has_one_covenant(self) -> None:
        """Test FinanceGroup has 1 covenant."""
        assert len(FINANCEGROUP_PROFILE["covenants"]) == 1

    def test_first_period_is_breach(self) -> None:
        """Test first period has BREACH status."""
        assert FINANCEGROUP_PROFILE["periods"][0]["expected_status"] == "BREACH"

    def test_second_period_is_near_breach(self) -> None:
        """Test second period has NEAR_BREACH status."""
        assert FINANCEGROUP_PROFILE["periods"][1]["expected_status"] == "NEAR_BREACH"


class TestHealthCareProfile:
    """Tests for HealthCare profile."""

    def test_deal_has_correct_sector(self) -> None:
        """Test HealthCare is in Healthcare sector."""
        assert HEALTHCARE_PROFILE["deal"]["sector"] == "Healthcare"

    def test_deal_has_correct_region(self) -> None:
        """Test HealthCare is in Asia."""
        assert HEALTHCARE_PROFILE["deal"]["region"] == "Asia"

    def test_has_current_ratio_covenant(self) -> None:
        """Test HealthCare has current ratio covenant."""
        cov = HEALTHCARE_PROFILE["covenants"][0]
        assert cov["name"] == "Current Ratio"
        assert cov["formula"] == "current_assets / current_liabilities"
        assert cov["direction"] == ">="

    def test_second_period_is_breach(self) -> None:
        """Test second period has BREACH status."""
        assert HEALTHCARE_PROFILE["periods"][1]["expected_status"] == "BREACH"


class TestCloudTechProfile:
    """Tests for CloudTech profile."""

    def test_deal_has_correct_sector(self) -> None:
        """Test CloudTech is in Technology sector."""
        assert CLOUDTECH_PROFILE["deal"]["sector"] == "Technology"

    def test_deal_has_correct_region(self) -> None:
        """Test CloudTech is in Europe."""
        assert CLOUDTECH_PROFILE["deal"]["region"] == "Europe"

    def test_first_two_periods_near_breach(self) -> None:
        """Test first two periods have NEAR_BREACH status."""
        assert CLOUDTECH_PROFILE["periods"][0]["expected_status"] == "NEAR_BREACH"
        assert CLOUDTECH_PROFILE["periods"][1]["expected_status"] == "NEAR_BREACH"


class TestPeriodStructure:
    """Tests for period data structure."""

    def test_periods_have_required_metrics(self) -> None:
        """Test all periods have all required metrics."""
        for profile in ALL_PROFILES:
            for period in profile["periods"]:
                metrics = period["metrics"]
                assert "total_debt" in metrics
                assert "ebitda" in metrics
                assert "interest_expense" in metrics
                assert "current_assets" in metrics
                assert "current_liabilities" in metrics

    def test_periods_have_valid_dates(self) -> None:
        """Test all periods have valid ISO date strings."""
        for profile in ALL_PROFILES:
            for period in profile["periods"]:
                start = period["start_iso"]
                end = period["end_iso"]
                # Basic ISO date format check
                assert len(start) == 10
                assert len(end) == 10
                assert start[4] == "-"
                assert start[7] == "-"
                assert end[4] == "-"
                assert end[7] == "-"

    def test_metrics_are_positive(self) -> None:
        """Test all metric values are positive."""
        for profile in ALL_PROFILES:
            for period in profile["periods"]:
                metrics = period["metrics"]
                assert metrics["total_debt"] > 0
                assert metrics["ebitda"] > 0
                assert metrics["interest_expense"] > 0
                assert metrics["current_assets"] > 0
                assert metrics["current_liabilities"] > 0


class TestDealStructure:
    """Tests for deal data structure."""

    def test_deals_have_required_fields(self) -> None:
        """Test all deals have required fields."""
        for profile in ALL_PROFILES:
            deal = profile["deal"]
            assert "name" in deal
            assert "borrower" in deal
            assert "sector" in deal
            assert "region" in deal
            assert "commitment_cents" in deal
            assert "currency" in deal
            assert "maturity_iso" in deal

    def test_deals_have_valid_commitment(self) -> None:
        """Test all deals have positive commitment amounts."""
        for profile in ALL_PROFILES:
            assert profile["deal"]["commitment_cents"] > 0

    def test_deals_have_valid_currency(self) -> None:
        """Test all deals have valid currency codes."""
        valid_currencies = {"USD", "EUR", "GBP", "JPY"}
        for profile in ALL_PROFILES:
            assert profile["deal"]["currency"] in valid_currencies


class TestCovenantStructure:
    """Tests for covenant data structure."""

    def test_covenants_have_required_fields(self) -> None:
        """Test all covenants have required fields."""
        for profile in ALL_PROFILES:
            for cov in profile["covenants"]:
                assert "name" in cov
                assert "formula" in cov
                assert "threshold_scaled" in cov
                assert "direction" in cov
                assert "frequency" in cov

    def test_covenants_have_valid_direction(self) -> None:
        """Test all covenants have valid direction."""
        for profile in ALL_PROFILES:
            for cov in profile["covenants"]:
                assert cov["direction"] in ("<=", ">=")

    def test_covenants_have_valid_frequency(self) -> None:
        """Test all covenants have valid frequency."""
        for profile in ALL_PROFILES:
            for cov in profile["covenants"]:
                assert cov["frequency"] in ("QUARTERLY", "ANNUAL")

    def test_covenants_have_positive_threshold(self) -> None:
        """Test all covenants have positive thresholds."""
        for profile in ALL_PROFILES:
            for cov in profile["covenants"]:
                assert cov["threshold_scaled"] > 0
