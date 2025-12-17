"""Tests for explainers types module.

Covers all TypedDicts and type exports.
"""

from __future__ import annotations

from covenant_ml.explainers.types import (
    ComputationalCost,
    ExplainerCapabilities,
    ExplainerConfigUnion,
    ExplainerName,
    ExplainRequestConfig,
    ExplainResult,
    FeatureImportanceScore,
    GradientConfig,
    GradientExplainConfig,
    IntegratedGradientsConfig,
    IntegratedGradientsExplainConfig,
    PermutationConfig,
    PermutationExplainConfig,
    ShapTreeExplainConfig,
    SupportedExplainer,
)


class TestExplainRequestConfig:
    """Tests for ExplainRequestConfig TypedDict."""

    def test_explainrequestconfig_has_all_required_fields(self) -> None:
        """ExplainRequestConfig has explainer, target_class, n_samples, random_state."""
        config: ExplainRequestConfig = {
            "explainer": "permutation",
            "target_class": 1,
            "n_samples": 100,
            "random_state": 42,
        }
        assert config["explainer"] == "permutation"
        assert config["target_class"] == 1
        assert config["n_samples"] == 100
        assert config["random_state"] == 42

    def test_explainrequestconfig_accepts_all_explainer_types(self) -> None:
        """ExplainRequestConfig accepts all SupportedExplainer values."""
        explainers: list[SupportedExplainer] = [
            "permutation",
            "gradient",
            "integrated_gradients",
            "shap_tree",
        ]
        for exp in explainers:
            config: ExplainRequestConfig = {
                "explainer": exp,
                "target_class": 1,
                "n_samples": 50,
                "random_state": 0,
            }
            assert config["explainer"] == exp


class TestPermutationExplainConfig:
    """Tests for PermutationExplainConfig TypedDict."""

    def test_permutationexplainconfig_has_n_repeats(self) -> None:
        """PermutationExplainConfig includes n_repeats field."""
        config: PermutationExplainConfig = {
            "explainer": "permutation",
            "target_class": 1,
            "n_samples": 100,
            "random_state": 42,
            "n_repeats": 10,
        }
        assert config["explainer"] == "permutation"
        assert config["n_repeats"] == 10


class TestGradientExplainConfig:
    """Tests for GradientExplainConfig TypedDict."""

    def test_gradientexplainconfig_has_gradient_options(self) -> None:
        """GradientExplainConfig includes multiply_by_input and absolute_value."""
        config: GradientExplainConfig = {
            "explainer": "gradient",
            "target_class": 1,
            "n_samples": 100,
            "random_state": 42,
            "multiply_by_input": True,
            "absolute_value": True,
        }
        assert config["explainer"] == "gradient"
        assert config["multiply_by_input"] is True
        assert config["absolute_value"] is True


class TestIntegratedGradientsExplainConfig:
    """Tests for IntegratedGradientsExplainConfig TypedDict."""

    def test_integratedgradientsexplainconfig_has_ig_options(self) -> None:
        """IntegratedGradientsExplainConfig includes n_steps and baseline_mode."""
        config: IntegratedGradientsExplainConfig = {
            "explainer": "integrated_gradients",
            "target_class": 1,
            "n_samples": 100,
            "random_state": 42,
            "n_steps": 50,
            "baseline_mode": "zeros",
        }
        assert config["explainer"] == "integrated_gradients"
        assert config["n_steps"] == 50
        assert config["baseline_mode"] == "zeros"

    def test_integratedgradientsexplainconfig_accepts_mean_baseline(self) -> None:
        """IntegratedGradientsExplainConfig accepts mean baseline mode."""
        config: IntegratedGradientsExplainConfig = {
            "explainer": "integrated_gradients",
            "target_class": 0,
            "n_samples": 50,
            "random_state": 123,
            "n_steps": 25,
            "baseline_mode": "mean",
        }
        assert config["baseline_mode"] == "mean"


class TestShapTreeExplainConfig:
    """Tests for ShapTreeExplainConfig TypedDict."""

    def test_shaptreeexplainconfig_is_minimal(self) -> None:
        """ShapTreeExplainConfig has no extra fields beyond base."""
        config: ShapTreeExplainConfig = {
            "explainer": "shap_tree",
            "target_class": 1,
            "n_samples": 200,
            "random_state": 99,
        }
        assert config["explainer"] == "shap_tree"
        assert config["target_class"] == 1


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
            "explainer": "permutation",
            "n_samples_used": 100,
            "n_features": 10,
            "target_class": 1,
            "feature_importances": [importance],
            "duration_seconds": 1.5,
        }
        assert result["status"] == "complete"
        assert result["backend"] == "xgboost"
        assert result["explainer"] == "permutation"
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
            "explainer": "gradient",
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

    def test_computationalcost_literal_values(self) -> None:
        """ComputationalCost literal includes expected values."""
        # These are valid values per the type definition
        low: ComputationalCost = "low"
        medium: ComputationalCost = "medium"
        high: ComputationalCost = "high"
        assert low == "low"
        assert medium == "medium"
        assert high == "high"

    def test_explainername_literal_values(self) -> None:
        """ExplainerName literal includes base explainer names."""
        perm: ExplainerName = "permutation"
        grad: ExplainerName = "gradient"
        ig: ExplainerName = "integrated_gradients"
        assert perm == "permutation"
        assert grad == "gradient"
        assert ig == "integrated_gradients"

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
            "computational_cost": "medium",
        }
        assert caps["requires_gradients"] is True
        assert caps["requires_background_data"] is False
        assert caps["computational_cost"] == "medium"

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


class TestExplainerConfigUnion:
    """Tests for ExplainerConfigUnion type alias."""

    def test_explainerconfigunion_accepts_permutation(self) -> None:
        """ExplainerConfigUnion accepts PermutationExplainConfig."""
        config: ExplainerConfigUnion = {
            "explainer": "permutation",
            "target_class": 1,
            "n_samples": 100,
            "random_state": 42,
            "n_repeats": 10,
        }
        assert config["explainer"] == "permutation"

    def test_explainerconfigunion_accepts_gradient(self) -> None:
        """ExplainerConfigUnion accepts GradientExplainConfig."""
        config: ExplainerConfigUnion = {
            "explainer": "gradient",
            "target_class": 1,
            "n_samples": 100,
            "random_state": 42,
            "multiply_by_input": True,
            "absolute_value": True,
        }
        assert config["explainer"] == "gradient"

    def test_explainerconfigunion_accepts_integrated_gradients(self) -> None:
        """ExplainerConfigUnion accepts IntegratedGradientsExplainConfig."""
        config: ExplainerConfigUnion = {
            "explainer": "integrated_gradients",
            "target_class": 1,
            "n_samples": 100,
            "random_state": 42,
            "n_steps": 50,
            "baseline_mode": "mean",
        }
        assert config["explainer"] == "integrated_gradients"

    def test_explainerconfigunion_accepts_shap_tree(self) -> None:
        """ExplainerConfigUnion accepts ShapTreeExplainConfig."""
        config: ExplainerConfigUnion = {
            "explainer": "shap_tree",
            "target_class": 1,
            "n_samples": 100,
            "random_state": 42,
        }
        assert config["explainer"] == "shap_tree"
