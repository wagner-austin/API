"""Classification explainer adapters: gradient, integrated gradients, SHAP tree."""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

import numpy as np
from cleargbm.types import GradientBoostingModel
from numpy.typing import NDArray
from platform_ml.explainers import (
    BoosterModelProtocol,
    ShapTreeWrapper,
    TreeModelProtocol,
)
from platform_ml.explainers.protocol import PredictorProtocol
from platform_ml.explainers.types import (
    ExplainerCapabilities,
    ExplainerName,
    FeatureImportanceScore,
)

from ..backends.cleargbm import try_extract_cleargbm_model
from .cleargbm_shap import ClearGBMShapWrapper


class ClearGBMPreparedProtocol(Protocol):
    """Protocol for ClearGBM prepared classifier with model access.

    Used to detect ClearGBM models and extract the underlying
    GradientBoostingModel for SHAP explanation.
    """

    @property
    def model(self) -> GradientBoostingModel:
        """Get the underlying ClearGBM model."""
        ...

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        ...


class _ComputeGradientsCallable(Protocol):
    """Protocol for compute_gradients method callable."""

    def __call__(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Compute gradients of output w.r.t. input."""
        ...


class _GradientModelWrapper:
    """Wrapper providing typed access to compute_gradients via string getattr.

    Used to wrap a PredictorProtocol model that has been validated at runtime
    to have compute_gradients method, providing proper typing for internal use.
    """

    _ATTR_NAME: str = "compute_gradients"

    def __init__(self, model: PredictorProtocol) -> None:
        """Initialize wrapper with a model that has compute_gradients.

        Args:
            model: Model with predict_proba and compute_gradients methods.

        Raises:
            AttributeError: If model doesn't have compute_gradients method.
        """
        if not hasattr(model, self._ATTR_NAME):
            raise AttributeError(
                f"Model {type(model).__name__} must have compute_gradients() method. "
                "Use a neural network backend (mlp, lstm)."
            )
        self._model = model
        # Use string variable to prevent linter from converting to direct access
        # Assign directly to protocol type to avoid Any from getattr
        self._compute_grad: _ComputeGradientsCallable = getattr(model, self._ATTR_NAME)

    def compute_gradients(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Compute gradients of output w.r.t. input.

        Args:
            x: Input features.
            target_class: Class index for gradients.

        Returns:
            Gradients array.
        """
        result: NDArray[np.float64] = self._compute_grad(x, target_class)
        return result


def _get_importance_from_pair(pair: tuple[int, float]) -> float:
    """Extract importance value from (index, importance) tuple.

    Args:
        pair: Tuple of (feature_index, importance_score).

    Returns:
        The importance score.
    """
    return pair[1]


def _rank_features(
    feature_names: list[str],
    importances: NDArray[np.float64],
) -> list[FeatureImportanceScore]:
    """Rank features by importance and return sorted list.

    Args:
        feature_names: List of feature names.
        importances: Array of importance scores with shape (n_features,).

    Returns:
        List of FeatureImportanceScore sorted by rank (most important first).
    """
    # Build list of (index, importance) using flat iterator with item()
    index_importance_pairs: list[tuple[int, float]] = []
    for i, imp in enumerate(importances.flat):
        imp_float: float = float(imp.item())
        index_importance_pairs.append((i, imp_float))

    # Sort by importance descending using typed helper
    sorted_pairs: list[tuple[int, float]] = sorted(
        index_importance_pairs,
        key=_get_importance_from_pair,
        reverse=True,
    )

    # Build ranked results
    results: list[FeatureImportanceScore] = []
    for rank_idx, pair in enumerate(sorted_pairs):
        feature_idx: int = pair[0]
        imp_value: float = pair[1]
        score: FeatureImportanceScore = {
            "name": feature_names[feature_idx],
            "importance": imp_value,
            "rank": rank_idx + 1,
        }
        results.append(score)

    return results


class _LocalExplanation(TypedDict, total=True):
    """Local explanation for a single sample from ShapTreeWrapper.

    Matches platform_ml.explainers.tree.LocalExplanation structure.
    """

    base_value: float
    values: list[float]
    feature_names: list[str]


class _GradientAdapter:
    """Adapter implementing gradient-based feature importance.

    Uses _GradientModelWrapper for typed access to compute_gradients.
    """

    def __init__(self, multiply_by_input: bool, absolute_value: bool) -> None:
        """Initialize adapter with config.

        Args:
            multiply_by_input: Whether to multiply gradients by input values.
            absolute_value: Whether to use absolute values of attributions.
        """
        self._multiply_by_input = multiply_by_input
        self._absolute_value = absolute_value

    def explainer_name(self) -> ExplainerName:
        """Return explainer name."""
        return "gradient"

    def capabilities(self) -> ExplainerCapabilities:
        """Return capabilities."""
        return {
            "requires_gradients": True,
            "requires_background_data": False,
            "computational_cost": "low",
        }

    def compute_importance(
        self,
        *,
        model: PredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
        target_class: int,
    ) -> list[FeatureImportanceScore]:
        """Compute feature importance using gradient-based attribution.

        Args:
            model: Model implementing PredictorProtocol with compute_gradients method.
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names.
            target_class: Class index for importance computation.

        Returns:
            List of FeatureImportanceScore sorted by importance.

        Raises:
            AttributeError: If model doesn't have compute_gradients method.
        """
        # Wrap model for typed gradient access
        grad_model = _GradientModelWrapper(model)
        grads: NDArray[np.float64] = grad_model.compute_gradients(x_data, target_class)

        # Apply gradient * input if configured
        if self._multiply_by_input:
            grads = grads * x_data

        # Aggregate across samples (mean)
        n_samples = int(x_data.shape[0])
        summed: NDArray[np.float64] = np.sum(grads, axis=0)
        mean_grads: NDArray[np.float64] = summed / float(n_samples)

        # Apply absolute value if configured
        if self._absolute_value:
            mean_grads = np.abs(mean_grads)

        return _rank_features(feature_names, mean_grads)


class _IntegratedGradientsAdapter:
    """Adapter implementing integrated gradients directly.

    Uses _GradientModelWrapper for typed access to compute_gradients.
    """

    def __init__(self, n_steps: int, baseline_mode: str) -> None:
        """Initialize adapter with config.

        Args:
            n_steps: Number of interpolation steps.
            baseline_mode: How to compute baseline ("zeros" or "mean").
        """
        self._n_steps = n_steps
        self._baseline_mode = baseline_mode

    def explainer_name(self) -> ExplainerName:
        """Return explainer name."""
        return "integrated_gradients"

    def capabilities(self) -> ExplainerCapabilities:
        """Return capabilities."""
        return {
            "requires_gradients": True,
            "requires_background_data": False,
            "computational_cost": "high",
        }

    def compute_importance(
        self,
        *,
        model: PredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
        target_class: int,
    ) -> list[FeatureImportanceScore]:
        """Compute feature importance using integrated gradients.

        Args:
            model: Model implementing PredictorProtocol with compute_gradients method.
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names.
            target_class: Class index for importance computation.

        Returns:
            List of FeatureImportanceScore sorted by importance.

        Raises:
            AttributeError: If model doesn't have compute_gradients method.
        """
        # Wrap model for typed gradient access
        grad_model = _GradientModelWrapper(model)

        n_samples = int(x_data.shape[0])
        n_features = int(x_data.shape[1])

        # Compute baseline
        baseline: NDArray[np.float64]
        if self._baseline_mode == "zeros":
            baseline = np.zeros((n_samples, n_features), dtype=np.float64)
        else:  # "mean"
            mean_vals: NDArray[np.float64] = np.mean(x_data, axis=0).astype(np.float64)
            baseline = np.tile(mean_vals, (n_samples, 1)).astype(np.float64)

        # Compute integrated gradients via Riemann sum
        integrated: NDArray[np.float64] = np.zeros((n_samples, n_features), dtype=np.float64)

        for step in range(self._n_steps):
            alpha = float(step) / float(self._n_steps)
            interpolated: NDArray[np.float64] = (baseline + alpha * (x_data - baseline)).astype(
                np.float64
            )
            grads: NDArray[np.float64] = grad_model.compute_gradients(interpolated, target_class)
            integrated = integrated + grads

        # Scale by (x - baseline) / n_steps
        diff: NDArray[np.float64] = (x_data - baseline).astype(np.float64)
        integrated = (integrated * diff / float(self._n_steps)).astype(np.float64)

        # Aggregate across samples (mean absolute value)
        abs_integrated: NDArray[np.float64] = np.abs(integrated)
        summed: NDArray[np.float64] = np.sum(abs_integrated, axis=0)
        mean_ig: NDArray[np.float64] = (summed / float(n_samples)).astype(np.float64)

        return _rank_features(feature_names, mean_ig)


class _ShapTreeAdapter:
    """Adapter to make ShapTreeWrapper conform to FeatureExplainer protocol.

    Wraps ShapTreeWrapper (for XGBoost/LightGBM) or ClearGBMShapWrapper
    (for ClearGBM) to provide compute_importance() method that returns
    aggregated SHAP values as FeatureImportanceScore list.

    Supports multiple tree-based backends:
    - XGBoost: Uses ShapTreeWrapper with shap.TreeExplainer
    - LightGBM: Uses ShapTreeWrapper with shap.TreeExplainer
    - ClearGBM: Uses ClearGBMShapWrapper with converted tree format
    """

    def __init__(self) -> None:
        """Initialize adapter (wrapper created per-model in compute_importance)."""
        pass

    def explainer_name(self) -> ExplainerName:
        """Return explainer name."""
        # Note: ExplainerName doesn't include "shap_tree", so we return closest match
        return "permutation"  # Placeholder - actual name tracked in registry

    def capabilities(self) -> ExplainerCapabilities:
        """Return capabilities."""
        return {
            "requires_gradients": False,
            "requires_background_data": False,
            "computational_cost": "medium",
        }

    def compute_importance(
        self,
        *,
        model: PredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
        target_class: int,
    ) -> list[FeatureImportanceScore]:
        """Compute feature importance using SHAP TreeExplainer.

        Automatically detects the model type and uses the appropriate
        SHAP wrapper:
        - ClearGBM models: ClearGBMShapWrapper
        - XGBoost/LightGBM models: ShapTreeWrapper

        Args:
            model: Tree model implementing PredictorProtocol.
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names.
            target_class: Class index (used to select SHAP values for that class).

        Returns:
            List of FeatureImportanceScore sorted by importance.
        """
        # Detect model type and create appropriate wrapper
        raw_explanations: list[_LocalExplanation]

        gbm_model = try_extract_cleargbm_model(model)
        if gbm_model is not None:
            # Use ClearGBM-specific SHAP wrapper
            wrapper = ClearGBMShapWrapper(gbm_model)
            cgbm_explanations = wrapper.explain_local(x_data, feature_names)
            # Convert to _LocalExplanation format
            raw_explanations = [
                _LocalExplanation(
                    base_value=exp["base_value"],
                    values=exp["values"],
                    feature_names=exp["feature_names"],
                )
                for exp in cgbm_explanations
            ]
        else:
            # SHAP introspects the native model, so any prepared form that
            # is a wrapper must hand over what it wraps. XGBoost's prepared
            # model is already native and yields None here.
            native = try_extract_native_tree_model(model)
            tree_model: TreeModelProtocol | BoosterModelProtocol = (
                model if native is None else native
            )
            tree_wrapper = ShapTreeWrapper(tree_model)
            tree_explanations = tree_wrapper.explain_local(x_data, feature_names)
            raw_explanations = [
                _LocalExplanation(
                    base_value=exp["base_value"],
                    values=exp["values"],
                    feature_names=exp["feature_names"],
                )
                for exp in tree_explanations
            ]

        # Aggregate SHAP values across samples (mean absolute value)
        n_features = len(feature_names)
        aggregated: NDArray[np.float64] = np.zeros(n_features, dtype=np.float64)

        sample_count: int = 0
        for raw_exp in raw_explanations:
            # Create typed dict from raw explanation
            exp: _LocalExplanation = {
                "base_value": raw_exp["base_value"],
                "values": raw_exp["values"],
                "feature_names": raw_exp["feature_names"],
            }
            values: list[float] = exp["values"]
            # Use enumerate to iterate and update in-place
            for feature_idx, val in enumerate(values):
                current: float = float(aggregated.flat[feature_idx].item())
                new_val: float = current + abs(val)
                aggregated[feature_idx] = new_val
            sample_count += 1

        # Average across samples
        for i in range(n_features):
            current_agg: float = float(aggregated.flat[i].item())
            aggregated[i] = current_agg / float(sample_count)

        return _rank_features(feature_names, aggregated)


@runtime_checkable
class _HasNativeModel(Protocol):
    """A prepared classifier that can surrender its native model handle.

    SHAP TreeExplainer reads a model's tree structure and rejects anything it
    does not recognise, so every backend whose prepared form is a wrapper has
    to expose what it wraps. Declaring that once here, rather than adding an
    extractor per backend, means a backend added later works by implementing
    the property.
    """

    @property
    def raw_model(self) -> TreeModelProtocol | BoosterModelProtocol:
        """The native model SHAP can introspect."""
        ...


def try_extract_native_tree_model(
    prepared: PredictorProtocol,
) -> TreeModelProtocol | BoosterModelProtocol | None:
    """Return the native model behind a prepared classifier, if it exposes one.

    Args:
        prepared: Prepared model from any backend.

    Returns:
        The native handle when the prepared model wraps one, else None. None
        means the prepared model is already native, as XGBoost's is.
    """
    if isinstance(prepared, _HasNativeModel):
        return prepared.raw_model
    return None


__all__ = [
    "ClearGBMPreparedProtocol",
    "_GradientAdapter",
    "_GradientModelWrapper",
    "_IntegratedGradientsAdapter",
    "_ShapTreeAdapter",
    "_get_importance_from_pair",
    "_rank_features",
]
