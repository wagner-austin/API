"""Tests for covenant_domain.rules module."""

from __future__ import annotations

import pytest

from covenant_domain.models import Covenant, CovenantId, DealId, Measurement
from covenant_domain.rules import (
    _build_metrics_for_period,
    classify_status,
    evaluate_all_covenants_for_period,
    evaluate_covenant_for_period,
)


class TestClassifyStatus:
    def test_lte_ok(self) -> None:
        result = classify_status(
            threshold_value_scaled=3_500_000,
            threshold_direction="<=",
            calculated_value_scaled=3_000_000,
            tolerance_ratio_scaled=100_000,
        )
        assert result == "OK"

    def test_lte_near_breach(self) -> None:
        result = classify_status(
            threshold_value_scaled=3_500_000,
            threshold_direction="<=",
            calculated_value_scaled=3_400_000,
            tolerance_ratio_scaled=100_000,
        )
        assert result == "NEAR_BREACH"

    def test_lte_breach(self) -> None:
        result = classify_status(
            threshold_value_scaled=3_500_000,
            threshold_direction="<=",
            calculated_value_scaled=4_000_000,
            tolerance_ratio_scaled=100_000,
        )
        assert result == "BREACH"

    def test_gte_ok(self) -> None:
        result = classify_status(
            threshold_value_scaled=2_000_000,
            threshold_direction=">=",
            calculated_value_scaled=3_000_000,
            tolerance_ratio_scaled=100_000,
        )
        assert result == "OK"

    def test_gte_near_breach(self) -> None:
        result = classify_status(
            threshold_value_scaled=2_000_000,
            threshold_direction=">=",
            calculated_value_scaled=2_100_000,
            tolerance_ratio_scaled=100_000,
        )
        assert result == "NEAR_BREACH"

    def test_gte_breach(self) -> None:
        result = classify_status(
            threshold_value_scaled=2_000_000,
            threshold_direction=">=",
            calculated_value_scaled=1_500_000,
            tolerance_ratio_scaled=100_000,
        )
        assert result == "BREACH"

    def test_lte_exactly_at_threshold_is_ok(self) -> None:
        result = classify_status(
            threshold_value_scaled=3_500_000,
            threshold_direction="<=",
            calculated_value_scaled=3_500_000,
            tolerance_ratio_scaled=100_000,
        )
        assert result == "NEAR_BREACH"

    def test_gte_exactly_at_threshold_is_near_breach(self) -> None:
        result = classify_status(
            threshold_value_scaled=2_000_000,
            threshold_direction=">=",
            calculated_value_scaled=2_000_000,
            tolerance_ratio_scaled=100_000,
        )
        assert result == "NEAR_BREACH"

    def test_zero_tolerance_at_threshold_is_ok(self) -> None:
        # With zero tolerance and value exactly at threshold for "<=", it's OK (not greater than)
        result = classify_status(
            threshold_value_scaled=3_500_000,
            threshold_direction="<=",
            calculated_value_scaled=3_500_000,
            tolerance_ratio_scaled=0,
        )
        assert result == "OK"

    def test_zero_tolerance_over_threshold_is_breach(self) -> None:
        result = classify_status(
            threshold_value_scaled=3_500_000,
            threshold_direction="<=",
            calculated_value_scaled=3_500_001,
            tolerance_ratio_scaled=0,
        )
        assert result == "BREACH"


class TestBuildMetricsForPeriod:
    def test_single_metric(self) -> None:
        measurements = [
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="total_debt",
                metric_value_scaled=100_000_000,
            )
        ]
        result = _build_metrics_for_period(measurements, "2024-01-01", "2024-03-31")
        assert result == {"total_debt": 100_000_000}

    def test_multiple_metrics(self) -> None:
        measurements = [
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="total_debt",
                metric_value_scaled=100_000_000,
            ),
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="ebitda",
                metric_value_scaled=50_000_000,
            ),
        ]
        result = _build_metrics_for_period(measurements, "2024-01-01", "2024-03-31")
        assert result == {"total_debt": 100_000_000, "ebitda": 50_000_000}

    def test_filters_by_period(self) -> None:
        measurements = [
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="total_debt",
                metric_value_scaled=100_000_000,
            ),
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-04-01",
                period_end_iso="2024-06-30",
                metric_name="total_debt",
                metric_value_scaled=200_000_000,
            ),
        ]
        result = _build_metrics_for_period(measurements, "2024-01-01", "2024-03-31")
        assert result == {"total_debt": 100_000_000}

    def test_duplicate_metric_raises(self) -> None:
        measurements = [
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="total_debt",
                metric_value_scaled=100_000_000,
            ),
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="total_debt",
                metric_value_scaled=200_000_000,
            ),
        ]
        with pytest.raises(ValueError) as exc_info:
            _build_metrics_for_period(measurements, "2024-01-01", "2024-03-31")
        assert "Duplicate metric total_debt" in str(exc_info.value)

    def test_empty_result_for_no_matching_period(self) -> None:
        measurements = [
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="total_debt",
                metric_value_scaled=100_000_000,
            )
        ]
        result = _build_metrics_for_period(measurements, "2024-04-01", "2024-06-30")
        assert result == {}


class TestEvaluateCovenantForPeriod:
    def test_ok_status(self) -> None:
        covenant = Covenant(
            id=CovenantId(value="cov-1"),
            deal_id=DealId(value="deal-1"),
            name="Debt to EBITDA",
            formula="total_debt / ebitda",
            threshold_value_scaled=3_500_000,
            threshold_direction="<=",
            frequency="QUARTERLY",
        )
        measurements = [
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="total_debt",
                metric_value_scaled=100_000_000,
            ),
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="ebitda",
                metric_value_scaled=50_000_000,
            ),
        ]
        result = evaluate_covenant_for_period(
            covenant=covenant,
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            measurements=measurements,
            tolerance_ratio_scaled=100_000,
        )
        assert result["covenant_id"]["value"] == "cov-1"
        assert result["period_start_iso"] == "2024-01-01"
        assert result["period_end_iso"] == "2024-03-31"
        assert result["calculated_value_scaled"] == 2_000_000
        assert result["status"] == "OK"

    def test_breach_status(self) -> None:
        covenant = Covenant(
            id=CovenantId(value="cov-1"),
            deal_id=DealId(value="deal-1"),
            name="Debt to EBITDA",
            formula="total_debt / ebitda",
            threshold_value_scaled=3_500_000,
            threshold_direction="<=",
            frequency="QUARTERLY",
        )
        measurements = [
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="total_debt",
                metric_value_scaled=400_000_000,
            ),
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="ebitda",
                metric_value_scaled=100_000_000,
            ),
        ]
        result = evaluate_covenant_for_period(
            covenant=covenant,
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            measurements=measurements,
            tolerance_ratio_scaled=100_000,
        )
        assert result["calculated_value_scaled"] == 4_000_000
        assert result["status"] == "BREACH"

    def test_missing_metric_raises(self) -> None:
        covenant = Covenant(
            id=CovenantId(value="cov-1"),
            deal_id=DealId(value="deal-1"),
            name="Debt to EBITDA",
            formula="total_debt / ebitda",
            threshold_value_scaled=3_500_000,
            threshold_direction="<=",
            frequency="QUARTERLY",
        )
        measurements = [
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="total_debt",
                metric_value_scaled=100_000_000,
            ),
        ]
        with pytest.raises(KeyError):
            evaluate_covenant_for_period(
                covenant=covenant,
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                measurements=measurements,
                tolerance_ratio_scaled=100_000,
            )


class TestEvaluateAllCovenantsForPeriod:
    def test_multiple_covenants(self) -> None:
        covenants = [
            Covenant(
                id=CovenantId(value="cov-1"),
                deal_id=DealId(value="deal-1"),
                name="Debt to EBITDA",
                formula="total_debt / ebitda",
                threshold_value_scaled=3_500_000,
                threshold_direction="<=",
                frequency="QUARTERLY",
            ),
            Covenant(
                id=CovenantId(value="cov-2"),
                deal_id=DealId(value="deal-1"),
                name="Interest Coverage",
                formula="ebitda / interest_expense",
                threshold_value_scaled=2_000_000,
                threshold_direction=">=",
                frequency="QUARTERLY",
            ),
        ]
        measurements = [
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="total_debt",
                metric_value_scaled=100_000_000,
            ),
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="ebitda",
                metric_value_scaled=50_000_000,
            ),
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="interest_expense",
                metric_value_scaled=10_000_000,
            ),
        ]
        results = evaluate_all_covenants_for_period(
            covenants=covenants,
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            measurements=measurements,
            tolerance_ratio_scaled=100_000,
        )
        assert len(results) == 2
        assert results[0]["covenant_id"]["value"] == "cov-1"
        assert results[0]["status"] == "OK"
        assert results[1]["covenant_id"]["value"] == "cov-2"
        assert results[1]["status"] == "OK"

    def test_empty_covenants(self) -> None:
        results = evaluate_all_covenants_for_period(
            covenants=[],
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            measurements=[],
            tolerance_ratio_scaled=100_000,
        )
        assert results == []

    def test_preserves_order(self) -> None:
        covenants = [
            Covenant(
                id=CovenantId(value="cov-b"),
                deal_id=DealId(value="deal-1"),
                name="Covenant B",
                formula="a",
                threshold_value_scaled=10_000_000,
                threshold_direction="<=",
                frequency="QUARTERLY",
            ),
            Covenant(
                id=CovenantId(value="cov-a"),
                deal_id=DealId(value="deal-1"),
                name="Covenant A",
                formula="a",
                threshold_value_scaled=10_000_000,
                threshold_direction="<=",
                frequency="QUARTERLY",
            ),
        ]
        measurements = [
            Measurement(
                deal_id=DealId(value="deal-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                metric_name="a",
                metric_value_scaled=5_000_000,
            ),
        ]
        results = evaluate_all_covenants_for_period(
            covenants=covenants,
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            measurements=measurements,
            tolerance_ratio_scaled=100_000,
        )
        assert results[0]["covenant_id"]["value"] == "cov-b"
        assert results[1]["covenant_id"]["value"] == "cov-a"
