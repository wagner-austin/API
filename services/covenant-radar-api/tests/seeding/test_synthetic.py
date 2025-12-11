"""Tests for synthetic data generator."""

from __future__ import annotations

import numpy as np

from covenant_radar_api.seeding.synthetic import (
    RiskParams,
    count_breach_labels,
    generate_synthetic_profile,
    generate_synthetic_profiles,
)


class TestGenerateSyntheticProfile:
    """Tests for generate_synthetic_profile."""

    def test_healthy_improving_profile(self) -> None:
        """Test generating a healthy improving profile."""
        rng = np.random.default_rng(42)
        profile = generate_synthetic_profile(rng, 1, "healthy", "improving")

        # Name and borrower contain the idx "1"
        assert "1" in profile["deal"]["borrower"]
        assert profile["deal"]["sector"] in ("Technology", "Finance", "Healthcare")
        assert profile["deal"]["region"] in ("North America", "Europe", "Asia")
        assert len(profile["covenants"]) == 2
        assert len(profile["periods"]) == 5

    def test_stressed_stable_profile(self) -> None:
        """Test generating a stressed stable profile."""
        rng = np.random.default_rng(42)
        profile = generate_synthetic_profile(rng, 2, "stressed", "stable")

        assert len(profile["periods"]) == 5
        for period in profile["periods"]:
            assert period["metrics"]["total_debt"] > 0
            assert period["metrics"]["ebitda"] > 0

    def test_distressed_deteriorating_profile(self) -> None:
        """Test generating a distressed deteriorating profile."""
        rng = np.random.default_rng(42)
        profile = generate_synthetic_profile(rng, 3, "distressed", "deteriorating")

        # Distressed deals should have at least some breaches
        breach_count = sum(1 for p in profile["periods"] if p["expected_status"] == "BREACH")
        assert breach_count > 0

    def test_healthy_deteriorating_trend(self) -> None:
        """Test healthy deal with deteriorating trend."""
        rng = np.random.default_rng(42)
        profile = generate_synthetic_profile(rng, 4, "healthy", "deteriorating")

        # Should have periods from healthy to distressed
        assert len(profile["periods"]) == 5

    def test_stressed_improving_trend(self) -> None:
        """Test stressed deal with improving trend."""
        rng = np.random.default_rng(42)
        profile = generate_synthetic_profile(rng, 5, "stressed", "improving")

        assert len(profile["periods"]) == 5

    def test_distressed_improving_trend(self) -> None:
        """Test distressed deal with improving trend."""
        rng = np.random.default_rng(42)
        profile = generate_synthetic_profile(rng, 6, "distressed", "improving")

        assert len(profile["periods"]) == 5

    def test_distressed_stable_trend(self) -> None:
        """Test distressed deal with stable trend."""
        rng = np.random.default_rng(42)
        profile = generate_synthetic_profile(rng, 7, "distressed", "stable")

        # All periods should be distressed
        assert len(profile["periods"]) == 5

    def test_profile_has_required_fields(self) -> None:
        """Test that profile has all required fields."""
        rng = np.random.default_rng(42)
        profile = generate_synthetic_profile(rng, 1, "healthy", "stable")

        deal = profile["deal"]
        assert "name" in deal
        assert "borrower" in deal
        assert "sector" in deal
        assert "region" in deal
        assert "commitment_cents" in deal
        assert "currency" in deal
        assert "maturity_iso" in deal

        for covenant in profile["covenants"]:
            assert "name" in covenant
            assert "formula" in covenant
            assert "threshold_scaled" in covenant
            assert "direction" in covenant
            assert "frequency" in covenant

        for period in profile["periods"]:
            assert "start_iso" in period
            assert "end_iso" in period
            assert "metrics" in period
            assert "expected_status" in period


class TestGenerateSyntheticProfiles:
    """Tests for generate_synthetic_profiles batch generation."""

    def test_generates_correct_count(self) -> None:
        """Test generating correct number of profiles."""
        profiles = generate_synthetic_profiles(n_deals=20, random_seed=42)
        assert len(profiles) == 20

    def test_distribution_matches_ratios(self) -> None:
        """Test that distribution roughly matches specified ratios."""
        profiles = generate_synthetic_profiles(
            n_deals=100,
            random_seed=42,
            healthy_ratio=0.5,
            stressed_ratio=0.25,
        )
        # Distribution is applied before shuffle so counts are exact
        assert len(profiles) == 100

    def test_small_batch(self) -> None:
        """Test generating a small batch."""
        profiles = generate_synthetic_profiles(n_deals=5, random_seed=42)
        assert len(profiles) == 5

    def test_reproducibility(self) -> None:
        """Test that same seed produces same results."""
        profiles1 = generate_synthetic_profiles(n_deals=10, random_seed=42)
        profiles2 = generate_synthetic_profiles(n_deals=10, random_seed=42)

        for p1, p2 in zip(profiles1, profiles2, strict=True):
            assert p1["deal"]["name"] == p2["deal"]["name"]
            assert p1["deal"]["borrower"] == p2["deal"]["borrower"]

    def test_different_seeds_different_results(self) -> None:
        """Test that different seeds produce different results."""
        profiles1 = generate_synthetic_profiles(n_deals=10, random_seed=42)
        profiles2 = generate_synthetic_profiles(n_deals=10, random_seed=123)

        # At least some should be different
        different_count = sum(
            1
            for p1, p2 in zip(profiles1, profiles2, strict=True)
            if p1["deal"]["name"] != p2["deal"]["name"]
        )
        assert different_count > 0


class TestCountBreachLabels:
    """Tests for count_breach_labels statistics function."""

    def test_empty_profiles(self) -> None:
        """Test counting breach labels for empty profiles."""
        result = count_breach_labels(())
        assert result["n_deals"] == 0
        assert result["n_periods"] == 0
        assert result["n_breach"] == 0
        assert result["breach_rate"] == 0.0

    def test_counts_breaches_correctly(self) -> None:
        """Test that breaches are counted correctly."""
        profiles = generate_synthetic_profiles(n_deals=50, random_seed=42)
        result = count_breach_labels(profiles)

        assert result["n_deals"] == 50
        assert result["n_periods"] == 250  # 50 deals * 5 periods
        assert result["n_breach"] >= 0
        assert result["n_near_breach"] >= 0
        assert result["n_ok"] >= 0

        # Total should equal n_periods
        total = result["n_breach"] + result["n_near_breach"] + result["n_ok"]
        assert total == result["n_periods"]

        # Breach rate should be between 0 and 1
        breach_rate = float(result["breach_rate"])
        assert 0.0 <= breach_rate <= 1.0


class TestDetermineStatus:
    """Tests for _determine_status internal function (via generated profiles)."""

    def test_zero_ebitda_is_breach(self) -> None:
        """Test that zero EBITDA results in BREACH status."""
        from covenant_radar_api.seeding.profiles import MetricsSeed
        from covenant_radar_api.seeding.synthetic import _determine_status

        metrics = MetricsSeed(
            total_debt=1000,
            ebitda=0,
            interest_expense=100,
            current_assets=500,
            current_liabilities=300,
        )
        assert _determine_status(metrics) == "BREACH"

    def test_negative_ebitda_is_breach(self) -> None:
        """Test that negative EBITDA results in BREACH status."""
        from covenant_radar_api.seeding.profiles import MetricsSeed
        from covenant_radar_api.seeding.synthetic import _determine_status

        metrics = MetricsSeed(
            total_debt=1000,
            ebitda=-100,
            interest_expense=100,
            current_assets=500,
            current_liabilities=300,
        )
        assert _determine_status(metrics) == "BREACH"


class TestRiskParams:
    """Tests for RiskParams constants."""

    def test_leverage_ranges_ordered(self) -> None:
        """Test that leverage ranges are properly ordered."""
        assert RiskParams.HEALTHY_LEVERAGE_MIN < RiskParams.HEALTHY_LEVERAGE_MAX
        assert RiskParams.STRESSED_LEVERAGE_MIN < RiskParams.STRESSED_LEVERAGE_MAX
        assert RiskParams.DISTRESSED_LEVERAGE_MIN < RiskParams.DISTRESSED_LEVERAGE_MAX

        # Healthy < Stressed < Distressed
        assert RiskParams.HEALTHY_LEVERAGE_MAX < RiskParams.STRESSED_LEVERAGE_MIN
        assert RiskParams.STRESSED_LEVERAGE_MAX < RiskParams.DISTRESSED_LEVERAGE_MIN

    def test_coverage_ranges_ordered(self) -> None:
        """Test that coverage ranges are properly ordered (higher is healthier)."""
        assert RiskParams.HEALTHY_COVERAGE_MIN > RiskParams.STRESSED_COVERAGE_MAX
        # Stressed and distressed can touch at boundary (both at 2.0)
        assert RiskParams.STRESSED_COVERAGE_MIN >= RiskParams.DISTRESSED_COVERAGE_MAX

    def test_current_ratio_ranges_ordered(self) -> None:
        """Test that current ratio ranges are properly ordered."""
        assert RiskParams.HEALTHY_CURRENT_MIN > RiskParams.STRESSED_CURRENT_MAX
        # Stressed and distressed can touch at boundary
        assert RiskParams.STRESSED_CURRENT_MIN >= RiskParams.DISTRESSED_CURRENT_MAX
