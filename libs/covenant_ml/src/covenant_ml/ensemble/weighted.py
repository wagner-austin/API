"""Weighted ensemble for combining model predictions.

Provides a simple weighted average ensemble that combines predictions from
multiple models using learned or specified weights.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.ensemble.types import (
    EnsembleOOFData,
    EnsemblePrediction,
    EnsembleWeights,
    ModelOOFPredictions,
)


def validate_oof_data(oof_data: EnsembleOOFData) -> None:
    """Validate OOF data structure and consistency.

    Args:
        oof_data: OOF data to validate.

    Raises:
        ValueError: If data is invalid or inconsistent.
    """
    n_samples = oof_data["n_samples"]
    n_models = oof_data["n_models"]
    labels = oof_data["labels"]
    model_preds = oof_data["model_predictions"]

    if n_models < 2:
        raise ValueError(f"Ensemble requires at least 2 models, got {n_models}")

    if len(labels) != n_samples:
        raise ValueError(f"Labels length ({len(labels)}) does not match n_samples ({n_samples})")

    if len(model_preds) != n_models:
        raise ValueError(
            f"model_predictions length ({len(model_preds)}) does not match n_models ({n_models})"
        )

    for pred in model_preds:
        if len(pred["predictions"]) != n_samples:
            raise ValueError(
                f"Model {pred['model_name']} has {len(pred['predictions'])} predictions, "
                f"expected {n_samples}"
            )
        if len(pred["fold_indices"]) != n_samples:
            raise ValueError(
                f"Model {pred['model_name']} has {len(pred['fold_indices'])} fold_indices, "
                f"expected {n_samples}"
            )


def validate_weights(weights: EnsembleWeights, n_models: int) -> None:
    """Validate ensemble weights.

    Args:
        weights: Weights to validate.
        n_models: Expected number of models.

    Raises:
        ValueError: If weights are invalid.
    """
    if len(weights["weights"]) != n_models:
        raise ValueError(
            f"Weights length ({len(weights['weights'])}) does not match n_models ({n_models})"
        )

    if len(weights["model_names"]) != n_models:
        n_names = len(weights["model_names"])
        raise ValueError(f"model_names length ({n_names}) does not match n_models ({n_models})")

    weight_sum = float(np.sum(weights["weights"]))
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    if np.any(weights["weights"] < 0):
        raise ValueError("Weights must be non-negative")


def create_equal_weights(model_names: tuple[str, ...]) -> EnsembleWeights:
    """Create equal weights for all models.

    Args:
        model_names: Names of models in the ensemble.

    Returns:
        EnsembleWeights with equal weight for each model.

    Raises:
        ValueError: If fewer than 2 model names provided.
    """
    n_models = len(model_names)
    if n_models < 2:
        raise ValueError(f"Ensemble requires at least 2 models, got {n_models}")

    weight_value = 1.0 / n_models
    weights: NDArray[np.float64] = np.full(n_models, weight_value, dtype=np.float64)

    return EnsembleWeights(
        weights=weights,
        model_names=model_names,
    )


def create_oof_data(
    model_predictions: tuple[ModelOOFPredictions, ...],
    labels: NDArray[np.int64],
) -> EnsembleOOFData:
    """Create validated OOF data structure.

    Args:
        model_predictions: OOF predictions from each model.
        labels: True labels for all samples.

    Returns:
        Validated EnsembleOOFData.

    Raises:
        ValueError: If data is invalid.
    """
    if len(model_predictions) < 2:
        raise ValueError(f"Ensemble requires at least 2 models, got {len(model_predictions)}")

    n_samples = len(labels)
    n_models = len(model_predictions)

    oof_data = EnsembleOOFData(
        model_predictions=model_predictions,
        labels=labels,
        n_samples=n_samples,
        n_models=n_models,
    )

    validate_oof_data(oof_data)
    return oof_data


def compute_weighted_predictions(
    oof_data: EnsembleOOFData,
    weights: EnsembleWeights,
) -> EnsemblePrediction:
    """Compute weighted ensemble predictions.

    Args:
        oof_data: OOF predictions from all models.
        weights: Weights for each model.

    Returns:
        EnsemblePrediction with weighted predictions.

    Raises:
        ValueError: If weights don't match models.
    """
    validate_oof_data(oof_data)
    validate_weights(weights, oof_data["n_models"])

    n_samples = oof_data["n_samples"]
    n_models = oof_data["n_models"]

    # Stack predictions: shape (n_models, n_samples)
    pred_stack: NDArray[np.float64] = np.zeros((n_models, n_samples), dtype=np.float64)
    for i, pred in enumerate(oof_data["model_predictions"]):
        pred_stack[i, :] = pred["predictions"]

    # Compute weighted contributions: shape (n_models, n_samples)
    weight_array: NDArray[np.float64] = weights["weights"]
    contributions: NDArray[np.float64] = pred_stack * weight_array[:, np.newaxis]

    # Sum to get ensemble predictions: shape (n_samples,)
    ensemble_preds: NDArray[np.float64] = np.sum(contributions, axis=0)

    return EnsemblePrediction(
        predictions=ensemble_preds,
        weights=weights,
        model_contributions=contributions,
    )


def extract_prediction_matrix(oof_data: EnsembleOOFData) -> NDArray[np.float64]:
    """Extract predictions as a matrix for optimization.

    Args:
        oof_data: OOF data from all models.

    Returns:
        Prediction matrix of shape (n_models, n_samples).
    """
    n_samples = oof_data["n_samples"]
    n_models = oof_data["n_models"]

    pred_matrix: NDArray[np.float64] = np.zeros((n_models, n_samples), dtype=np.float64)
    for i, pred in enumerate(oof_data["model_predictions"]):
        pred_matrix[i, :] = pred["predictions"]

    return pred_matrix


__all__ = [
    "compute_weighted_predictions",
    "create_equal_weights",
    "create_oof_data",
    "extract_prediction_matrix",
    "validate_oof_data",
    "validate_weights",
]
