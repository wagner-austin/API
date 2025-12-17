"""Shapley value computation wrapper for tree-based models.

Implements the "Anti-Corruption Layer" pattern to safely interface with the
untyped 'shap' library while maintaining strict typing in the rest of the codebase.
"""

from __future__ import annotations

from typing import Protocol, TypedDict, TypeGuard

import numpy as np
from numpy.typing import NDArray

# -----------------------------------------------------------------------------
# Internal Protocols (The "Anti-Corruption" Interface)
# -----------------------------------------------------------------------------


class TreeModelProtocol(Protocol):
    """Protocol for tree-based models compatible with SHAP TreeExplainer.

    XGBoost, LightGBM, and sklearn tree models implement predict_proba.
    SHAP TreeExplainer uses internal tree structure, not this method directly.
    """

    def predict_proba(
        self,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict class probabilities."""
        ...


class ShapExplainerProtocol(Protocol):
    """Protocol for shap.TreeExplainer instance."""

    @property
    def expected_value(self) -> float | NDArray[np.float64]:
        """Base value (bias) of the model."""
        ...

    def shap_values(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
        tree_limit: int | None = None,
        approximate: bool = False,
        check_additivity: bool = True,
        from_call: bool = False,
    ) -> NDArray[np.float64] | list[NDArray[np.float64]]:
        """Compute Shapley values."""
        ...


class TreeExplainerConstructor(Protocol):
    """Protocol for the TreeExplainer constructor."""

    def __call__(
        self,
        model: TreeModelProtocol,
        data: NDArray[np.float64] | None = None,
        model_output: str = "raw",
        feature_perturbation: str = "interventional",
    ) -> ShapExplainerProtocol:
        """Create a new TreeExplainer."""
        ...


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _is_ndarray(ev: float | NDArray[np.float64]) -> TypeGuard[NDArray[np.float64]]:
    """Check if expected value is an ndarray.

    Args:
        ev: Value to check.

    Returns:
        True if ev is an ndarray with flat attribute.
    """
    return hasattr(ev, "flat")


def _extract_float_from_array(arr: NDArray[np.float64]) -> float:
    """Extract last float from array using flat iteration.

    Args:
        arr: Source array.

    Returns:
        Last float value in array.
    """
    last_val = 0.0
    for val in arr.flat:
        last_val = float(val.item())
    return last_val


def _extract_expected_value(ev: float | NDArray[np.float64]) -> float:
    """Extract a single float from expected_value (scalar or array).

    SHAP TreeExplainer.expected_value can be:
    - A scalar float (XGBoost binary classification)
    - A 1D array with 1 element
    - A 1D array with 2 elements [neg_class, pos_class]

    Args:
        ev: Expected value from TreeExplainer.

    Returns:
        Float value (last element if array, for positive class).
    """
    if _is_ndarray(ev):
        # TypeGuard narrows ev to NDArray[np.float64]
        return _extract_float_from_array(ev)

    # It's a scalar float
    return float(ev)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


class LocalExplanation(TypedDict):
    """Type-safe container for a single instance's explanation."""

    base_value: float
    feature_names: list[str]
    values: list[float]


class ShapTreeWrapper:
    """Type-safe wrapper for shap.TreeExplainer.

    Encapsulates the untyped 'shap' library interaction.
    """

    def __init__(self, model: TreeModelProtocol) -> None:
        """Initialize wrapper with a trained model.

        Args:
            model: Trained model (XGBoost, LightGBM, Sklearn tree, etc.)
                   Must be compatible with shap.TreeExplainer.
        """
        # Dynamic import to avoid top-level untyped import.
        # Strict pattern:
        # 1. Use __import__("module_name")
        # 2. Use getattr(mod, "ClassName") but assign directly to Protocol type
        shap_mod = __import__("shap")

        # Assign directly to Protocol type to override Any from getattr
        tree_explainer_cls: TreeExplainerConstructor = shap_mod.TreeExplainer

        # We assume 'margin' (log-odds) output for binary classification consistency
        self._explainer = tree_explainer_cls(model)

        # Handle scalar vs array expected_value
        ev = self._explainer.expected_value
        self._expected_value = _extract_expected_value(ev)

    def explain_local(
        self,
        x: NDArray[np.float64],
        feature_names: list[str],
    ) -> list[LocalExplanation]:
        """Compute local Shapley values for the provided instances.

        Args:
            x: Feature matrix (n_samples, n_features)
            feature_names: List of feature names corresponding to columns of x

        Returns:
            List of LocalExplanation objects, one per sample.

        Raises:
            ValueError: If feature count doesn't match x columns.
        """
        n_features = int(x.shape[1])
        if n_features != len(feature_names):
            raise ValueError(f"Feature count mismatch: x={n_features}, names={len(feature_names)}")

        # Compute values
        # shap_values returns (n_samples, n_features) for binary XGBoost
        raw_values = self._explainer.shap_values(x)

        # Handle list output (e.g. from sklearn models: [negative_vals, positive_vals])
        # Take positive class if list, otherwise use directly
        # SHAP TreeExplainer always returns (n_samples, n_features) shape arrays
        values_array: NDArray[np.float64] = (
            raw_values[-1] if isinstance(raw_values, list) else raw_values
        )

        results: list[LocalExplanation] = []
        n_samples = values_array.shape[0]

        for i in range(n_samples):
            # Convert row to strict float list
            row_vals: NDArray[np.float64] = values_array[i]
            # Ensure we iterate purely
            values_list: list[float] = [float(v) for v in row_vals.flat]

            explanation: LocalExplanation = {
                "base_value": self._expected_value,
                "feature_names": feature_names,
                "values": values_list,
            }
            results.append(explanation)

        return results
