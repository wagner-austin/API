"""Tests for explainers types module.

Covers all TypedDicts and type exports.
"""

from __future__ import annotations

from platform_ml.explainers.types import ComputationalCost as PlatformComputationalCost
from platform_ml.explainers.types import ExplainerName as PlatformExplainerName

from covenant_ml.explainers.types import (
    ComputationalCost,
    ExplainerCapabilities,
    ExplainerName,
    ExplainResult,
    FeatureImportanceScore,
    GradientConfig,
    IntegratedGradientsConfig,
    PermutationConfig,
    RegressionExplainResult,
)


class TestExplainResult:
    """Tests for ExplainResult TypedDict."""

    def test_explainresult_has_all_required_fields(self) -> None:
        """ExplainResult contains all required result fields."""
        importance: FeatureImportanceScore = {
            "name": "feature_1",
            "importance": 0.5,
            "rank": 1,
        }
        result: ExplainResult = {
            "status": "complete",
            "backend": "xgboost",
            "explainer": ExplainerName.PERMUTATION,
            "n_samples_used": 100,
            "n_features": 10,
            "target_class": 1,
            "feature_importances": [importance],
            "duration_seconds": 1.5,
        }
        assert result["status"] == "complete"
        assert result["backend"] == "xgboost"
        assert result["explainer"] is ExplainerName.PERMUTATION
        assert result["n_samples_used"] == 100
        assert result["n_features"] == 10
        assert result["target_class"] == 1
        assert len(result["feature_importances"]) == 1
        assert result["duration_seconds"] == 1.5

    def test_explainresult_accepts_failed_status(self) -> None:
        """ExplainResult accepts failed status."""
        result: ExplainResult = {
            "status": "failed",
            "backend": "mlp",
            "explainer": ExplainerName.GRADIENT,
            "n_samples_used": 0,
            "n_features": 5,
            "target_class": 1,
            "feature_importances": [],
            "duration_seconds": 0.0,
        }
        assert result["status"] == "failed"
        assert result["feature_importances"] == []


class TestReExports:
    """Tests for re-exported types from platform_ml."""

    def test_computational_cost_is_platform_mls_enum(self) -> None:
        """The re-exported ComputationalCost is platform_ml's class, not a copy."""
        assert ComputationalCost is PlatformComputationalCost

    def test_explainer_name_is_platform_mls_enum(self) -> None:
        """The re-exported ExplainerName is platform_ml's class, not a copy."""
        assert ExplainerName is PlatformExplainerName

    def test_featureimportancescore_structure(self) -> None:
        """FeatureImportanceScore has name, importance, rank fields."""
        score: FeatureImportanceScore = {
            "name": "my_feature",
            "importance": 0.75,
            "rank": 2,
        }
        assert score["name"] == "my_feature"
        assert score["importance"] == 0.75
        assert score["rank"] == 2

    def test_explainercapabilities_structure(self) -> None:
        """ExplainerCapabilities has required capability flags."""
        caps: ExplainerCapabilities = {
            "requires_gradients": True,
            "requires_background_data": False,
            "computational_cost": ComputationalCost.MEDIUM,
        }
        assert caps["requires_gradients"] is True
        assert caps["requires_background_data"] is False
        assert caps["computational_cost"] is ComputationalCost.MEDIUM

    def test_permutationconfig_structure(self) -> None:
        """PermutationConfig has n_repeats and random_state."""
        config: PermutationConfig = {
            "n_repeats": 5,
            "random_state": 42,
        }
        assert config["n_repeats"] == 5
        assert config["random_state"] == 42

    def test_gradientconfig_structure(self) -> None:
        """GradientConfig has multiply_by_input and absolute_value."""
        config: GradientConfig = {
            "multiply_by_input": False,
            "absolute_value": True,
        }
        assert config["multiply_by_input"] is False
        assert config["absolute_value"] is True

    def test_integratedgradientsconfig_structure(self) -> None:
        """IntegratedGradientsConfig has n_steps and baseline_mode."""
        config: IntegratedGradientsConfig = {
            "n_steps": 100,
            "baseline_mode": "zeros",
        }
        assert config["n_steps"] == 100
        assert config["baseline_mode"] == "zeros"


class TestRegressionExplainResult:
    """Tests for RegressionExplainResult TypedDict."""

    def test_regression_explain_result_structure(self) -> None:
        """RegressionExplainResult has all required fields (no target_class)."""
        score: FeatureImportanceScore = {
            "name": "feat1",
            "importance": 0.5,
            "rank": 1,
        }
        result: RegressionExplainResult = {
            "status": "complete",
            "backend": "xgboost_reg",
            "explainer": ExplainerName.PERMUTATION,
            "n_samples_used": 100,
            "n_features": 5,
            "feature_importances": [score],
            "duration_seconds": 1.5,
        }
        assert result["status"] == "complete"
        assert result["backend"] == "xgboost_reg"
        assert result["explainer"] is ExplainerName.PERMUTATION
        assert result["n_samples_used"] == 100
        assert result["n_features"] == 5
        assert len(result["feature_importances"]) == 1
        assert result["duration_seconds"] == 1.5
