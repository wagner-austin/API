"""Tests for platform_ml.explainers.protocol module.

Achieves 100% statement and branch coverage by testing Protocol conformance
with concrete implementations. Uses real classes that implement the protocols.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from platform_ml.explainers.protocol import (
    FeatureExplainer,
    GradientModelProtocol,
    PredictorProtocol,
)
from platform_ml.explainers.types import (
    ExplainerCapabilities,
    ExplainerName,
    FeatureImportanceScore,
)

from .array_helpers import assert_close, get_float, make_float64_2d


class SimplePredictorModel:
    """Simple predictor that implements PredictorProtocol."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return uniform probabilities for binary classification.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Probabilities with shape (n_samples, 2).
        """
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.zeros((n_samples, 2), dtype=np.float64)
        proba[:, 0] = 0.3
        proba[:, 1] = 0.7
        return proba


class SimpleGradientModel:
    """Simple gradient model that implements GradientModelProtocol."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return uniform probabilities for binary classification.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Probabilities with shape (n_samples, 2).
        """
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.zeros((n_samples, 2), dtype=np.float64)
        proba[:, 0] = 0.4
        proba[:, 1] = 0.6
        return proba

    def compute_gradients(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Return simple gradients based on input.

        Args:
            x: Input features with shape (n_samples, n_features).
            target_class: Class index for gradient computation.

        Returns:
            Gradients with shape (n_samples, n_features).
        """
        n_samples = int(x.shape[0])
        n_features = int(x.shape[1])
        gradients: NDArray[np.float64] = np.zeros((n_samples, n_features), dtype=np.float64)
        for i in range(n_features):
            gradients[:, i] = float(i + 1) * 0.1 * float(target_class + 1)
        return gradients


class SimpleExplainer:
    """Simple explainer that implements FeatureExplainer protocol."""

    def explainer_name(self) -> ExplainerName:
        """Return explainer name.

        Returns:
            Literal "permutation".
        """
        return "permutation"

    def capabilities(self) -> ExplainerCapabilities:
        """Return explainer capabilities.

        Returns:
            Capabilities dict.
        """
        caps: ExplainerCapabilities = {
            "requires_gradients": False,
            "requires_background_data": False,
            "computational_cost": "low",
        }
        return caps

    def compute_importance(
        self,
        *,
        model: PredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
        target_class: int,
    ) -> list[FeatureImportanceScore]:
        """Compute dummy importance scores.

        Args:
            model: Model implementing PredictorProtocol.
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names.
            target_class: Class index.

        Returns:
            List of FeatureImportanceScore.
        """
        _ = model.predict_proba(x_data)

        results: list[FeatureImportanceScore] = []
        for rank, name in enumerate(feature_names):
            score: FeatureImportanceScore = {
                "name": name,
                "importance": 1.0 / float(rank + 1),
                "rank": rank + 1,
            }
            results.append(score)
        return results


def test_predictor_protocol_conformance() -> None:
    """Verify SimplePredictorModel conforms to PredictorProtocol."""
    model: PredictorProtocol = SimplePredictorModel()
    x = make_float64_2d([[1.0, 2.0], [3.0, 4.0]])
    proba = model.predict_proba(x)

    assert proba.shape == (2, 2)
    assert get_float(proba, 0, 0) == 0.3
    assert get_float(proba, 0, 1) == 0.7
    assert get_float(proba, 1, 0) == 0.3
    assert get_float(proba, 1, 1) == 0.7


def test_gradient_model_protocol_predict_proba() -> None:
    """Verify SimpleGradientModel.predict_proba conforms to GradientModelProtocol."""
    model: GradientModelProtocol = SimpleGradientModel()
    x = make_float64_2d([[1.0, 2.0, 3.0]])
    proba = model.predict_proba(x)

    assert proba.shape == (1, 2)
    assert get_float(proba, 0, 0) == 0.4
    assert get_float(proba, 0, 1) == 0.6


def test_gradient_model_protocol_compute_gradients() -> None:
    """Verify SimpleGradientModel.compute_gradients conforms to GradientModelProtocol."""
    model: GradientModelProtocol = SimpleGradientModel()
    x = make_float64_2d([[1.0, 2.0, 3.0]])
    grads = model.compute_gradients(x, target_class=1)

    assert grads.shape == (1, 3)
    # target_class=1 means multiplier is 2
    assert_close(get_float(grads, 0, 0), 0.2)  # (0+1) * 0.1 * 2
    assert_close(get_float(grads, 0, 1), 0.4)  # (1+1) * 0.1 * 2
    assert_close(get_float(grads, 0, 2), 0.6)  # (2+1) * 0.1 * 2


def test_gradient_model_protocol_compute_gradients_class_zero() -> None:
    """Verify compute_gradients with target_class=0."""
    model: GradientModelProtocol = SimpleGradientModel()
    x = make_float64_2d([[1.0, 2.0]])
    grads = model.compute_gradients(x, target_class=0)

    assert grads.shape == (1, 2)
    # target_class=0 means multiplier is 1
    assert get_float(grads, 0, 0) == 0.1  # (0+1) * 0.1 * 1
    assert get_float(grads, 0, 1) == 0.2  # (1+1) * 0.1 * 1


def test_feature_explainer_protocol_conformance() -> None:
    """Verify SimpleExplainer conforms to FeatureExplainer protocol."""
    explainer: FeatureExplainer = SimpleExplainer()

    assert explainer.explainer_name() == "permutation"

    caps = explainer.capabilities()
    assert caps["requires_gradients"] is False
    assert caps["requires_background_data"] is False
    assert caps["computational_cost"] == "low"


def test_feature_explainer_compute_importance() -> None:
    """Verify FeatureExplainer.compute_importance works correctly."""
    explainer: FeatureExplainer = SimpleExplainer()
    model: PredictorProtocol = SimplePredictorModel()
    x = make_float64_2d([[1.0, 2.0], [3.0, 4.0]])
    feature_names = ["feat_a", "feat_b"]

    importance = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=1,
    )

    assert len(importance) == 2
    assert importance[0]["name"] == "feat_a"
    assert importance[0]["importance"] == 1.0
    assert importance[0]["rank"] == 1
    assert importance[1]["name"] == "feat_b"
    assert importance[1]["importance"] == 0.5
    assert importance[1]["rank"] == 2
