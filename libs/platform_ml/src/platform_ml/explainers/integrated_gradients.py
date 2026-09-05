"""Integrated Gradients feature importance explainer.

Computes feature importance using the Integrated Gradients method from:
"Axiomatic Attribution for Deep Networks" (Sundararajan et al., 2017).

Algorithm:
1. Define a baseline (zeros or mean of data)
2. Create interpolated inputs between baseline and actual inputs
3. Compute gradients at each interpolation step
4. Integrate (average) gradients along the path
5. Multiply by (input - baseline) to get attributions

This method satisfies key axioms:
- Sensitivity: If input differs from baseline, attribution is non-zero
- Implementation Invariance: Same attributions for functionally equivalent models
- Completeness: Attributions sum to prediction difference from baseline
"""

from __future__ import annotations

from typing import Literal
from typing import Protocol as TypingProtocol

import numpy as np
from numpy.typing import NDArray

from .protocol import GradientModelProtocol, PredictorProtocol
from .ranking import rank_importances
from .types import (
    ExplainerCapabilities,
    ExplainerName,
    FeatureImportanceScore,
    IntegratedGradientsConfig,
)


class _ComputeGradientsCallable(TypingProtocol):
    """Protocol for compute_gradients method callable."""

    def __call__(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Compute gradients of output w.r.t. input."""
        ...


class _GradientModelWrapper:
    """Wrapper that implements GradientModelProtocol for runtime-validated models.

    Used to wrap a PredictorProtocol model that has been validated at runtime
    to have compute_gradients method, providing proper typing for internal use.
    """

    def __init__(self, model: PredictorProtocol) -> None:
        """Initialize wrapper with a model that has compute_gradients.

        Args:
            model: Model with predict_proba and compute_gradients methods.

        Note:
            Caller must validate model has compute_gradients before wrapping.
        """
        self._model = model
        # Store the method reference - caller has validated this exists
        # Assign directly to typed variable to avoid Any from getattr
        _attr_name = "compute_gradients"
        self._compute_grad_method: _ComputeGradientsCallable = getattr(model, _attr_name)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Delegate to wrapped model's predict_proba.

        Args:
            x: Input features.

        Returns:
            Class probabilities.
        """
        return self._model.predict_proba(x)

    def compute_gradients(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Delegate to wrapped model's compute_gradients.

        Args:
            x: Input features.
            target_class: Class index for gradients.

        Returns:
            Gradients array.
        """
        result: NDArray[np.float64] = self._compute_grad_method(x, target_class)
        return result


INTEGRATED_GRADIENTS_CAPABILITIES: ExplainerCapabilities = {
    "requires_gradients": True,
    "requires_background_data": True,
    "computational_cost": "high",
}


def _validate_inputs(
    x_data: NDArray[np.float64],
    feature_names: list[str],
) -> None:
    """Validate input data and feature names match.

    Args:
        x_data: Input features array.
        feature_names: List of feature names.

    Raises:
        ValueError: If dimensions don't match.
    """
    n_features = int(x_data.shape[1])
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match x_data columns ({n_features})"
        )


def _compute_baseline(
    x_data: NDArray[np.float64],
    baseline_mode: Literal["zeros", "mean"],
) -> NDArray[np.float64]:
    """Compute baseline for integrated gradients.

    Args:
        x_data: Input features with shape (n_samples, n_features).
        baseline_mode: How to compute baseline.

    Returns:
        Baseline with shape (1, n_features).
    """
    n_features = int(x_data.shape[1])

    baseline: NDArray[np.float64]
    if baseline_mode == "zeros":
        baseline = np.zeros((1, n_features), dtype=np.float64)
    else:
        # baseline_mode == "mean"
        baseline = np.mean(x_data, axis=0, keepdims=True).astype(np.float64)

    return baseline


def _compute_interpolated_inputs(
    x_sample: NDArray[np.float64],
    baseline: NDArray[np.float64],
    n_steps: int,
) -> NDArray[np.float64]:
    """Create interpolated inputs between baseline and sample.

    Args:
        x_sample: Single input sample with shape (1, n_features).
        baseline: Baseline with shape (1, n_features).
        n_steps: Number of interpolation steps.

    Returns:
        Interpolated inputs with shape (n_steps, n_features).
    """
    # Create alpha values from 0 to 1
    alphas: NDArray[np.float64] = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)

    # Interpolate: baseline + alpha * (input - baseline)
    diff: NDArray[np.float64] = x_sample - baseline
    # alphas has shape (n_steps,), diff has shape (1, n_features)
    # We need alphas reshaped to (n_steps, 1) for broadcasting
    alphas_reshaped: NDArray[np.float64] = alphas.reshape(-1, 1)
    interpolated: NDArray[np.float64] = baseline + alphas_reshaped * diff

    return interpolated.astype(np.float64)


def _compute_integrated_gradients_single(
    model: GradientModelProtocol,
    x_sample: NDArray[np.float64],
    baseline: NDArray[np.float64],
    target_class: int,
    n_steps: int,
) -> NDArray[np.float64]:
    """Compute integrated gradients for a single sample.

    Args:
        model: Model with compute_gradients method.
        x_sample: Single sample with shape (1, n_features).
        baseline: Baseline with shape (1, n_features).
        target_class: Class index for gradient computation.
        n_steps: Number of interpolation steps.

    Returns:
        Attributions with shape (n_features,).
    """
    # Get interpolated inputs
    interpolated = _compute_interpolated_inputs(x_sample, baseline, n_steps)

    # Compute gradients at each interpolation point
    gradients: NDArray[np.float64] = model.compute_gradients(interpolated, target_class)

    # Average gradients (Riemann sum approximation of integral)
    avg_gradients: NDArray[np.float64] = np.mean(gradients, axis=0).astype(np.float64)

    # Multiply by (input - baseline)
    diff: NDArray[np.float64] = (x_sample - baseline).flatten().astype(np.float64)
    attributions: NDArray[np.float64] = avg_gradients * diff

    return attributions


def _compute_all_attributions(
    model: GradientModelProtocol,
    x_data: NDArray[np.float64],
    baseline: NDArray[np.float64],
    target_class: int,
    n_steps: int,
) -> NDArray[np.float64]:
    """Compute integrated gradients for all samples.

    Args:
        model: Model with compute_gradients method.
        x_data: Input data with shape (n_samples, n_features).
        baseline: Baseline with shape (1, n_features).
        target_class: Class index for gradient computation.
        n_steps: Number of interpolation steps.

    Returns:
        Attributions with shape (n_samples, n_features).
    """
    n_samples = int(x_data.shape[0])
    n_features = int(x_data.shape[1])

    all_attributions: NDArray[np.float64] = np.zeros((n_samples, n_features), dtype=np.float64)

    for i in range(n_samples):
        x_sample: NDArray[np.float64] = x_data[i : i + 1, :].astype(np.float64)
        attributions = _compute_integrated_gradients_single(
            model=model,
            x_sample=x_sample,
            baseline=baseline,
            target_class=target_class,
            n_steps=n_steps,
        )
        all_attributions[i, :] = attributions

    return all_attributions


def _aggregate_attributions(
    attributions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Aggregate attributions across samples.

    Args:
        attributions: Attributions with shape (n_samples, n_features).

    Returns:
        Mean absolute attribution per feature with shape (n_features,).
    """
    n_samples = int(attributions.shape[0])
    # Use np.abs with type annotation, then sum/len for mean
    abs_attr: NDArray[np.float64] = np.abs(attributions)
    summed: NDArray[np.float64] = np.sum(abs_attr, axis=0)
    mean_attr: NDArray[np.float64] = summed / float(n_samples)
    return mean_attr


class IntegratedGradientsExplainer:
    """Integrated Gradients feature importance explainer.

    Computes feature importance using path-integrated gradients.
    More accurate than simple gradients but computationally expensive.

    Args:
        config: Configuration with n_steps and baseline_mode.

    Example:
        >>> config: IntegratedGradientsConfig = {"n_steps": 50, "baseline_mode": "zeros"}
        >>> explainer = IntegratedGradientsExplainer(config)
        >>> importance = explainer.compute_importance(
        ...     model=gradient_model,
        ...     x_data=x_test,
        ...     feature_names=["feat1", "feat2"],
        ...     target_class=1,
        ... )
    """

    def __init__(self, config: IntegratedGradientsConfig) -> None:
        """Initialize integrated gradients explainer.

        Args:
            config: Configuration with n_steps and baseline_mode.
        """
        self._n_steps = int(config["n_steps"])
        self._baseline_mode: Literal["zeros", "mean"] = config["baseline_mode"]

    def explainer_name(self) -> ExplainerName:
        """Return explainer name.

        Returns:
            Literal "integrated_gradients".
        """
        return "integrated_gradients"

    def capabilities(self) -> ExplainerCapabilities:
        """Return explainer capabilities.

        Returns:
            Capabilities indicating gradients required, high cost.
        """
        return INTEGRATED_GRADIENTS_CAPABILITIES

    def compute_importance(
        self,
        *,
        model: PredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
        target_class: int,
    ) -> list[FeatureImportanceScore]:
        """Compute integrated gradients feature importance.

        Args:
            model: Model implementing PredictorProtocol. Must also have
                compute_gradients(x, target_class) method at runtime.
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names matching x_data columns.
            target_class: Class index for which to compute importance.

        Returns:
            List of FeatureImportanceScore sorted by rank.

        Raises:
            ValueError: If feature_names length doesn't match x_data columns.
            AttributeError: If model doesn't have compute_gradients method.
        """
        _validate_inputs(x_data, feature_names)

        # Validate model has compute_gradients at runtime
        if not hasattr(model, "compute_gradients"):
            raise AttributeError(
                f"Model {type(model).__name__} must have compute_gradients() method "
                "for integrated gradients explainer. Use a neural network backend (mlp, lstm)."
            )

        # Wrap model to provide proper GradientModelProtocol typing
        grad_model: GradientModelProtocol = _GradientModelWrapper(model)

        baseline = _compute_baseline(x_data, self._baseline_mode)

        attributions = _compute_all_attributions(
            model=grad_model,
            x_data=x_data,
            baseline=baseline,
            target_class=target_class,
            n_steps=self._n_steps,
        )

        importances = _aggregate_attributions(attributions)

        return rank_importances(feature_names, importances)


def create_integrated_gradients_explainer(
    config: IntegratedGradientsConfig,
) -> IntegratedGradientsExplainer:
    """Factory function to create an IntegratedGradientsExplainer.

    Args:
        config: Configuration with n_steps and baseline_mode.

    Returns:
        Configured IntegratedGradientsExplainer instance.
    """
    return IntegratedGradientsExplainer(config)


__all__ = [
    "INTEGRATED_GRADIENTS_CAPABILITIES",
    "IntegratedGradientsExplainer",
    "create_integrated_gradients_explainer",
]
