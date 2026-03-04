"""Test utilities for regression ensemble module.

Provides factory functions and test data generators for regression
ensemble tests. Reuses the fake minimize from ensemble.testing
since the scipy interface is identical.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.ensemble.regression_types import RegressionEnsembleOOFData
from covenant_ml.ensemble.types import ModelOOFPredictions


def make_regression_model_oof(
    name: str,
    predictions: tuple[float, ...],
) -> ModelOOFPredictions:
    """Create ModelOOFPredictions for regression testing.

    Args:
        name: Model name.
        predictions: Continuous prediction values.

    Returns:
        ModelOOFPredictions instance.
    """
    n_samples = len(predictions)
    preds: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    for i, v in enumerate(predictions):
        preds[i] = v

    fold_indices: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        fold_indices[i] = i % 3

    return ModelOOFPredictions(
        model_name=name,
        predictions=preds,
        fold_indices=fold_indices,
    )


def make_regression_oof_data(
    model_preds: tuple[tuple[str, tuple[float, ...]], ...],
    labels: tuple[float, ...],
) -> RegressionEnsembleOOFData:
    """Create RegressionEnsembleOOFData for testing.

    Args:
        model_preds: Tuple of (model_name, predictions) tuples.
        labels: True continuous labels.

    Returns:
        RegressionEnsembleOOFData instance.
    """
    preds = tuple(make_regression_model_oof(name, vals) for name, vals in model_preds)

    label_array: NDArray[np.float64] = np.zeros(len(labels), dtype=np.float64)
    for i, v in enumerate(labels):
        label_array[i] = v

    return RegressionEnsembleOOFData(
        model_predictions=preds,
        labels=label_array,
        n_samples=len(labels),
        n_models=len(model_preds),
    )


__all__ = [
    "make_regression_model_oof",
    "make_regression_oof_data",
]
