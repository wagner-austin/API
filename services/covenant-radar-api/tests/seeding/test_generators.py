"""Tests for seeding generators module."""

from __future__ import annotations

from covenant_radar_api.seeding.generators import (
    calculate_placeholder_value,
    generate_covenant,
    generate_covenant_result,
    generate_deal,
    generate_measurements,
)
from covenant_radar_api.seeding.profiles import (
    CovenantSeed,
    DealSeed,
    MetricsSeed,
    PeriodSeed,
)


class TestGenerateDeal:
    """Tests for generate_deal function."""

    def test_generates_deal_with_correct_id(self) -> None:
        """Test generated deal has correct ID."""
        seed = DealSeed(
            name="Test Deal",
            borrower="Test Corp",
            sector="Technology",
            region="North America",
            commitment_cents=100_000_00,
            currency="USD",
            maturity_iso="2025-12-31",
        )
        deal = generate_deal("test-deal-id", seed)
        assert deal["id"]["value"] == "test-deal-id"

    def test_generates_deal_with_correct_name(self) -> None:
        """Test generated deal has correct name."""
        seed = DealSeed(
            name="My Deal Name",
            borrower="Test Corp",
            sector="Finance",
            region="Europe",
            commitment_cents=200_000_00,
            currency="EUR",
            maturity_iso="2026-06-30",
        )
        deal = generate_deal("id-123", seed)
        assert deal["name"] == "My Deal Name"

    def test_generates_deal_with_all_fields(self) -> None:
        """Test generated deal has all required fields."""
        seed = DealSeed(
            name="Full Deal",
            borrower="Full Corp",
            sector="Healthcare",
            region="Asia",
            commitment_cents=500_000_00,
            currency="USD",
            maturity_iso="2027-03-31",
        )
        deal = generate_deal("full-id", seed)

        assert deal["id"]["value"] == "full-id"
        assert deal["name"] == "Full Deal"
        assert deal["borrower"] == "Full Corp"
        assert deal["sector"] == "Healthcare"
        assert deal["region"] == "Asia"
        assert deal["commitment_amount_cents"] == 500_000_00
        assert deal["currency"] == "USD"
        assert deal["maturity_date_iso"] == "2027-03-31"


class TestGenerateCovenant:
    """Tests for generate_covenant function."""

    def test_generates_covenant_with_correct_ids(self) -> None:
        """Test generated covenant has correct IDs."""
        seed = CovenantSeed(
            name="Test Covenant",
            formula="a / b",
            threshold_scaled=1_000_000,
            direction="<=",
            frequency="QUARTERLY",
        )
        covenant = generate_covenant("cov-id", "deal-id", seed)
        assert covenant["id"]["value"] == "cov-id"
        assert covenant["deal_id"]["value"] == "deal-id"

    def test_generates_covenant_with_le_direction(self) -> None:
        """Test generated covenant with <= direction."""
        seed = CovenantSeed(
            name="LE Covenant",
            formula="x / y",
            threshold_scaled=2_000_000,
            direction="<=",
            frequency="QUARTERLY",
        )
        covenant = generate_covenant("c1", "d1", seed)
        assert covenant["threshold_direction"] == "<="

    def test_generates_covenant_with_ge_direction(self) -> None:
        """Test generated covenant with >= direction."""
        seed = CovenantSeed(
            name="GE Covenant",
            formula="x / y",
            threshold_scaled=3_000_000,
            direction=">=",
            frequency="ANNUAL",
        )
        covenant = generate_covenant("c2", "d2", seed)
        assert covenant["threshold_direction"] == ">="

    def test_generates_covenant_with_all_fields(self) -> None:
        """Test generated covenant has all required fields."""
        seed = CovenantSeed(
            name="Full Covenant",
            formula="total_debt / ebitda",
            threshold_scaled=4_000_000,
            direction="<=",
            frequency="QUARTERLY",
        )
        covenant = generate_covenant("cov-full", "deal-full", seed)

        assert covenant["id"]["value"] == "cov-full"
        assert covenant["deal_id"]["value"] == "deal-full"
        assert covenant["name"] == "Full Covenant"
        assert covenant["formula"] == "total_debt / ebitda"
        assert covenant["threshold_value_scaled"] == 4_000_000
        assert covenant["threshold_direction"] == "<="
        assert covenant["frequency"] == "QUARTERLY"


class TestGenerateMeasurements:
    """Tests for generate_measurements function."""

    def test_generates_five_measurements(self) -> None:
        """Test generates exactly 5 measurements."""
        metrics = MetricsSeed(
            total_debt=100,
            ebitda=50,
            interest_expense=10,
            current_assets=80,
            current_liabilities=40,
        )
        measurements = generate_measurements("d1", "2024-01-01", "2024-03-31", metrics)
        assert len(measurements) == 5

    def test_measurements_have_correct_deal_id(self) -> None:
        """Test all measurements have correct deal ID."""
        metrics = MetricsSeed(
            total_debt=100,
            ebitda=50,
            interest_expense=10,
            current_assets=80,
            current_liabilities=40,
        )
        measurements = generate_measurements("my-deal", "2024-01-01", "2024-03-31", metrics)
        for m in measurements:
            assert m["deal_id"]["value"] == "my-deal"

    def test_measurements_have_correct_period(self) -> None:
        """Test all measurements have correct period dates."""
        metrics = MetricsSeed(
            total_debt=100,
            ebitda=50,
            interest_expense=10,
            current_assets=80,
            current_liabilities=40,
        )
        measurements = generate_measurements("d1", "2023-07-01", "2023-09-30", metrics)
        for m in measurements:
            assert m["period_start_iso"] == "2023-07-01"
            assert m["period_end_iso"] == "2023-09-30"

    def test_measurements_have_all_metric_names(self) -> None:
        """Test measurements include all metric types."""
        metrics = MetricsSeed(
            total_debt=100,
            ebitda=50,
            interest_expense=10,
            current_assets=80,
            current_liabilities=40,
        )
        measurements = generate_measurements("d1", "2024-01-01", "2024-03-31", metrics)
        metric_names = {m["metric_name"] for m in measurements}
        expected = {
            "total_debt",
            "ebitda",
            "interest_expense",
            "current_assets",
            "current_liabilities",
        }
        assert metric_names == expected

    def test_measurements_have_correct_values(self) -> None:
        """Test measurements have correct metric values."""
        metrics = MetricsSeed(
            total_debt=1000,
            ebitda=500,
            interest_expense=100,
            current_assets=800,
            current_liabilities=400,
        )
        measurements = generate_measurements("d1", "2024-01-01", "2024-03-31", metrics)
        value_map = {m["metric_name"]: m["metric_value_scaled"] for m in measurements}
        assert value_map["total_debt"] == 1000
        assert value_map["ebitda"] == 500
        assert value_map["interest_expense"] == 100
        assert value_map["current_assets"] == 800
        assert value_map["current_liabilities"] == 400


class TestCalculatePlaceholderValue:
    """Tests for calculate_placeholder_value function."""

    def test_ok_status_returns_low_value(self) -> None:
        """Test OK status returns value below thresholds."""
        value = calculate_placeholder_value("OK")
        assert value == 2_000_000

    def test_near_breach_returns_medium_value(self) -> None:
        """Test NEAR_BREACH status returns near-threshold value."""
        value = calculate_placeholder_value("NEAR_BREACH")
        assert value == 3_800_000

    def test_breach_returns_high_value(self) -> None:
        """Test BREACH status returns above-threshold value."""
        value = calculate_placeholder_value("BREACH")
        assert value == 5_000_000

    def test_ok_less_than_near_breach(self) -> None:
        """Test OK value is less than NEAR_BREACH value."""
        ok_val = calculate_placeholder_value("OK")
        near_val = calculate_placeholder_value("NEAR_BREACH")
        assert ok_val < near_val

    def test_near_breach_less_than_breach(self) -> None:
        """Test NEAR_BREACH value is less than BREACH value."""
        near_val = calculate_placeholder_value("NEAR_BREACH")
        breach_val = calculate_placeholder_value("BREACH")
        assert near_val < breach_val


class TestGenerateCovenantResult:
    """Tests for generate_covenant_result function."""

    def test_generates_result_with_correct_covenant_id(self) -> None:
        """Test generated result has correct covenant ID."""
        period = PeriodSeed(
            start_iso="2024-01-01",
            end_iso="2024-03-31",
            metrics=MetricsSeed(
                total_debt=100,
                ebitda=50,
                interest_expense=10,
                current_assets=80,
                current_liabilities=40,
            ),
            expected_status="OK",
        )
        result = generate_covenant_result("cov-123", period)
        assert result["covenant_id"]["value"] == "cov-123"

    def test_generates_result_with_correct_period(self) -> None:
        """Test generated result has correct period dates."""
        period = PeriodSeed(
            start_iso="2023-07-01",
            end_iso="2023-09-30",
            metrics=MetricsSeed(
                total_debt=100,
                ebitda=50,
                interest_expense=10,
                current_assets=80,
                current_liabilities=40,
            ),
            expected_status="BREACH",
        )
        result = generate_covenant_result("c1", period)
        assert result["period_start_iso"] == "2023-07-01"
        assert result["period_end_iso"] == "2023-09-30"

    def test_generates_result_with_ok_status(self) -> None:
        """Test generated result with OK status."""
        period = PeriodSeed(
            start_iso="2024-01-01",
            end_iso="2024-03-31",
            metrics=MetricsSeed(
                total_debt=100,
                ebitda=50,
                interest_expense=10,
                current_assets=80,
                current_liabilities=40,
            ),
            expected_status="OK",
        )
        result = generate_covenant_result("c1", period)
        assert result["status"] == "OK"
        assert result["calculated_value_scaled"] == 2_000_000

    def test_generates_result_with_near_breach_status(self) -> None:
        """Test generated result with NEAR_BREACH status."""
        period = PeriodSeed(
            start_iso="2024-01-01",
            end_iso="2024-03-31",
            metrics=MetricsSeed(
                total_debt=100,
                ebitda=50,
                interest_expense=10,
                current_assets=80,
                current_liabilities=40,
            ),
            expected_status="NEAR_BREACH",
        )
        result = generate_covenant_result("c1", period)
        assert result["status"] == "NEAR_BREACH"
        assert result["calculated_value_scaled"] == 3_800_000

    def test_generates_result_with_breach_status(self) -> None:
        """Test generated result with BREACH status."""
        period = PeriodSeed(
            start_iso="2024-01-01",
            end_iso="2024-03-31",
            metrics=MetricsSeed(
                total_debt=100,
                ebitda=50,
                interest_expense=10,
                current_assets=80,
                current_liabilities=40,
            ),
            expected_status="BREACH",
        )
        result = generate_covenant_result("c1", period)
        assert result["status"] == "BREACH"
        assert result["calculated_value_scaled"] == 5_000_000
