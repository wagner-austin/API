"""Tests for covenant_domain.features module."""

from __future__ import annotations

import pytest

from covenant_domain.features import (
    FEATURE_ORDER,
    _count_near_breaches,
    _safe_divide,
    classify_risk_tier,
    extract_features,
)
from covenant_domain.models import CovenantId, CovenantResult, Deal, DealId


class TestFeatureOrder:
    def test_feature_order_length(self) -> None:
        assert len(FEATURE_ORDER) == 8

    def test_feature_order_contents(self) -> None:
        assert "debt_to_ebitda" in FEATURE_ORDER
        assert "interest_cover" in FEATURE_ORDER
        assert "current_ratio" in FEATURE_ORDER
        assert "leverage_change_1p" in FEATURE_ORDER
        assert "leverage_change_4p" in FEATURE_ORDER
        assert "sector_encoded" in FEATURE_ORDER
        assert "region_encoded" in FEATURE_ORDER
        assert "near_breach_count_4p" in FEATURE_ORDER


class TestClassifyRiskTier:
    def test_low_risk(self) -> None:
        assert classify_risk_tier(0.0) == "LOW"
        assert classify_risk_tier(0.1) == "LOW"
        assert classify_risk_tier(0.29) == "LOW"

    def test_medium_risk(self) -> None:
        assert classify_risk_tier(0.3) == "MEDIUM"
        assert classify_risk_tier(0.5) == "MEDIUM"
        assert classify_risk_tier(0.69) == "MEDIUM"

    def test_high_risk(self) -> None:
        assert classify_risk_tier(0.7) == "HIGH"
        assert classify_risk_tier(0.9) == "HIGH"
        assert classify_risk_tier(1.0) == "HIGH"


class TestCountNearBreaches:
    def test_no_near_breaches(self) -> None:
        results = [
            CovenantResult(
                covenant_id=CovenantId(value="cov-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                calculated_value_scaled=3_000_000,
                status="OK",
            ),
            CovenantResult(
                covenant_id=CovenantId(value="cov-1"),
                period_start_iso="2023-10-01",
                period_end_iso="2023-12-31",
                calculated_value_scaled=3_000_000,
                status="OK",
            ),
        ]
        count = _count_near_breaches(results, 4)
        assert count == 0

    def test_some_near_breaches(self) -> None:
        results = [
            CovenantResult(
                covenant_id=CovenantId(value="cov-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                calculated_value_scaled=3_400_000,
                status="NEAR_BREACH",
            ),
            CovenantResult(
                covenant_id=CovenantId(value="cov-1"),
                period_start_iso="2023-10-01",
                period_end_iso="2023-12-31",
                calculated_value_scaled=3_000_000,
                status="OK",
            ),
            CovenantResult(
                covenant_id=CovenantId(value="cov-1"),
                period_start_iso="2023-07-01",
                period_end_iso="2023-09-30",
                calculated_value_scaled=3_450_000,
                status="NEAR_BREACH",
            ),
        ]
        count = _count_near_breaches(results, 4)
        assert count == 2

    def test_breach_not_counted(self) -> None:
        results = [
            CovenantResult(
                covenant_id=CovenantId(value="cov-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                calculated_value_scaled=4_000_000,
                status="BREACH",
            ),
        ]
        count = _count_near_breaches(results, 4)
        assert count == 0

    def test_respects_periods_limit(self) -> None:
        results = [
            CovenantResult(
                covenant_id=CovenantId(value="cov-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                calculated_value_scaled=3_400_000,
                status="NEAR_BREACH",
            ),
            CovenantResult(
                covenant_id=CovenantId(value="cov-1"),
                period_start_iso="2023-10-01",
                period_end_iso="2023-12-31",
                calculated_value_scaled=3_400_000,
                status="NEAR_BREACH",
            ),
            CovenantResult(
                covenant_id=CovenantId(value="cov-1"),
                period_start_iso="2023-07-01",
                period_end_iso="2023-09-30",
                calculated_value_scaled=3_400_000,
                status="NEAR_BREACH",
            ),
            CovenantResult(
                covenant_id=CovenantId(value="cov-1"),
                period_start_iso="2023-04-01",
                period_end_iso="2023-06-30",
                calculated_value_scaled=3_400_000,
                status="NEAR_BREACH",
            ),
            CovenantResult(
                covenant_id=CovenantId(value="cov-1"),
                period_start_iso="2023-01-01",
                period_end_iso="2023-03-31",
                calculated_value_scaled=3_400_000,
                status="NEAR_BREACH",
            ),
        ]
        count = _count_near_breaches(results, 4)
        assert count == 4

    def test_empty_results(self) -> None:
        count = _count_near_breaches([], 4)
        assert count == 0


class TestSafeDivide:
    def test_normal_division(self) -> None:
        result = _safe_divide(10.0, 2.0)
        assert result == 5.0

    def test_division_by_zero_returns_default(self) -> None:
        result = _safe_divide(10.0, 0.0)
        assert result == 0.0

    def test_division_by_zero_custom_default(self) -> None:
        result = _safe_divide(10.0, 0.0, default=-1.0)
        assert result == -1.0


class TestExtractFeatures:
    def test_extract_features_basic(self) -> None:
        deal = Deal(
            id=DealId(value="deal-1"),
            name="Acme Corp Loan",
            borrower="Acme Corporation",
            sector="Technology",
            region="North America",
            commitment_amount_cents=100_000_000_00,
            currency="USD",
            maturity_date_iso="2028-12-31",
        )
        metrics_current: dict[str, int] = {
            "total_debt": 100_000_000,
            "ebitda": 50_000_000,
            "interest_expense": 10_000_000,
            "current_assets": 200_000_000,
            "current_liabilities": 100_000_000,
        }
        metrics_1p_ago: dict[str, int] = {
            "total_debt": 90_000_000,
            "ebitda": 45_000_000,
        }
        metrics_4p_ago: dict[str, int] = {
            "total_debt": 80_000_000,
            "ebitda": 40_000_000,
        }
        recent_results: list[CovenantResult] = []
        sector_encoder: dict[str, int] = {"Technology": 1, "Finance": 2}
        region_encoder: dict[str, int] = {"North America": 1, "Europe": 2}

        features = extract_features(
            deal=deal,
            metrics_current=metrics_current,
            metrics_1p_ago=metrics_1p_ago,
            metrics_4p_ago=metrics_4p_ago,
            recent_results=recent_results,
            sector_encoder=sector_encoder,
            region_encoder=region_encoder,
        )

        assert features["debt_to_ebitda"] == 2.0
        assert features["interest_cover"] == 5.0
        assert features["current_ratio"] == 2.0
        assert features["sector_encoded"] == 1
        assert features["region_encoded"] == 1
        assert features["near_breach_count_4p"] == 0

    def test_extract_features_with_near_breaches(self) -> None:
        deal = Deal(
            id=DealId(value="deal-1"),
            name="Test Loan",
            borrower="Test Corp",
            sector="Finance",
            region="Europe",
            commitment_amount_cents=50_000_000_00,
            currency="EUR",
            maturity_date_iso="2027-06-30",
        )
        metrics_current: dict[str, int] = {
            "total_debt": 100_000_000,
            "ebitda": 50_000_000,
            "interest_expense": 10_000_000,
            "current_assets": 150_000_000,
            "current_liabilities": 100_000_000,
        }
        metrics_1p_ago: dict[str, int] = {}
        metrics_4p_ago: dict[str, int] = {}
        recent_results = [
            CovenantResult(
                covenant_id=CovenantId(value="cov-1"),
                period_start_iso="2024-01-01",
                period_end_iso="2024-03-31",
                calculated_value_scaled=3_400_000,
                status="NEAR_BREACH",
            ),
            CovenantResult(
                covenant_id=CovenantId(value="cov-1"),
                period_start_iso="2023-10-01",
                period_end_iso="2023-12-31",
                calculated_value_scaled=3_450_000,
                status="NEAR_BREACH",
            ),
        ]
        sector_encoder: dict[str, int] = {"Technology": 1, "Finance": 2}
        region_encoder: dict[str, int] = {"North America": 1, "Europe": 2}

        features = extract_features(
            deal=deal,
            metrics_current=metrics_current,
            metrics_1p_ago=metrics_1p_ago,
            metrics_4p_ago=metrics_4p_ago,
            recent_results=recent_results,
            sector_encoder=sector_encoder,
            region_encoder=region_encoder,
        )

        assert features["sector_encoded"] == 2
        assert features["region_encoded"] == 2
        assert features["near_breach_count_4p"] == 2

    def test_extract_features_zero_denominator(self) -> None:
        deal = Deal(
            id=DealId(value="deal-1"),
            name="Test Loan",
            borrower="Test Corp",
            sector="Technology",
            region="North America",
            commitment_amount_cents=50_000_000_00,
            currency="USD",
            maturity_date_iso="2027-06-30",
        )
        metrics_current: dict[str, int] = {
            "total_debt": 100_000_000,
            "ebitda": 0,
            "interest_expense": 0,
            "current_assets": 150_000_000,
            "current_liabilities": 0,
        }
        metrics_1p_ago: dict[str, int] = {}
        metrics_4p_ago: dict[str, int] = {}
        recent_results: list[CovenantResult] = []
        sector_encoder: dict[str, int] = {"Technology": 1}
        region_encoder: dict[str, int] = {"North America": 1}

        features = extract_features(
            deal=deal,
            metrics_current=metrics_current,
            metrics_1p_ago=metrics_1p_ago,
            metrics_4p_ago=metrics_4p_ago,
            recent_results=recent_results,
            sector_encoder=sector_encoder,
            region_encoder=region_encoder,
        )

        assert features["debt_to_ebitda"] == 0.0
        assert features["interest_cover"] == 0.0
        assert features["current_ratio"] == 0.0

    def test_extract_features_missing_sector_raises(self) -> None:
        deal = Deal(
            id=DealId(value="deal-1"),
            name="Test Loan",
            borrower="Test Corp",
            sector="Unknown",
            region="North America",
            commitment_amount_cents=50_000_000_00,
            currency="USD",
            maturity_date_iso="2027-06-30",
        )
        metrics_current: dict[str, int] = {
            "total_debt": 100_000_000,
            "ebitda": 50_000_000,
            "interest_expense": 10_000_000,
            "current_assets": 150_000_000,
            "current_liabilities": 100_000_000,
        }
        sector_encoder: dict[str, int] = {"Technology": 1}
        region_encoder: dict[str, int] = {"North America": 1}

        with pytest.raises(KeyError):
            extract_features(
                deal=deal,
                metrics_current=metrics_current,
                metrics_1p_ago={},
                metrics_4p_ago={},
                recent_results=[],
                sector_encoder=sector_encoder,
                region_encoder=region_encoder,
            )

    def test_extract_features_missing_region_raises(self) -> None:
        deal = Deal(
            id=DealId(value="deal-1"),
            name="Test Loan",
            borrower="Test Corp",
            sector="Technology",
            region="Unknown",
            commitment_amount_cents=50_000_000_00,
            currency="USD",
            maturity_date_iso="2027-06-30",
        )
        metrics_current: dict[str, int] = {
            "total_debt": 100_000_000,
            "ebitda": 50_000_000,
            "interest_expense": 10_000_000,
            "current_assets": 150_000_000,
            "current_liabilities": 100_000_000,
        }
        sector_encoder: dict[str, int] = {"Technology": 1}
        region_encoder: dict[str, int] = {"North America": 1}

        with pytest.raises(KeyError):
            extract_features(
                deal=deal,
                metrics_current=metrics_current,
                metrics_1p_ago={},
                metrics_4p_ago={},
                recent_results=[],
                sector_encoder=sector_encoder,
                region_encoder=region_encoder,
            )

    def test_extract_features_missing_current_metric_raises(self) -> None:
        deal = Deal(
            id=DealId(value="deal-1"),
            name="Test Loan",
            borrower="Test Corp",
            sector="Technology",
            region="North America",
            commitment_amount_cents=50_000_000_00,
            currency="USD",
            maturity_date_iso="2027-06-30",
        )
        metrics_current: dict[str, int] = {
            "total_debt": 100_000_000,
        }
        sector_encoder: dict[str, int] = {"Technology": 1}
        region_encoder: dict[str, int] = {"North America": 1}

        with pytest.raises(KeyError):
            extract_features(
                deal=deal,
                metrics_current=metrics_current,
                metrics_1p_ago={},
                metrics_4p_ago={},
                recent_results=[],
                sector_encoder=sector_encoder,
                region_encoder=region_encoder,
            )

    def test_extract_features_leverage_change_calculation(self) -> None:
        deal = Deal(
            id=DealId(value="deal-1"),
            name="Test Loan",
            borrower="Test Corp",
            sector="Technology",
            region="North America",
            commitment_amount_cents=50_000_000_00,
            currency="USD",
            maturity_date_iso="2027-06-30",
        )
        metrics_current: dict[str, int] = {
            "total_debt": 100_000_000,
            "ebitda": 50_000_000,
            "interest_expense": 10_000_000,
            "current_assets": 150_000_000,
            "current_liabilities": 100_000_000,
        }
        metrics_1p_ago: dict[str, int] = {
            "total_debt": 75_000_000,
            "ebitda": 50_000_000,
        }
        metrics_4p_ago: dict[str, int] = {
            "total_debt": 50_000_000,
            "ebitda": 50_000_000,
        }
        sector_encoder: dict[str, int] = {"Technology": 1}
        region_encoder: dict[str, int] = {"North America": 1}

        features = extract_features(
            deal=deal,
            metrics_current=metrics_current,
            metrics_1p_ago=metrics_1p_ago,
            metrics_4p_ago=metrics_4p_ago,
            recent_results=[],
            sector_encoder=sector_encoder,
            region_encoder=region_encoder,
        )

        assert features["debt_to_ebitda"] == 2.0
        assert features["leverage_change_1p"] == 0.5
        assert features["leverage_change_4p"] == 1.0
