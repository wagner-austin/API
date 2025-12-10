"""XGBoost model prediction for covenant breach risk."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from covenant_domain.features import LoanFeatures
from numpy.typing import NDArray

from .types import Proba2DProtocol, XGBClassifierLoader, XGBModelProtocol


def _extract_feature_values(feat: LoanFeatures) -> tuple[float, ...]:
    """Extract feature values from LoanFeatures in FEATURE_ORDER."""
    return (
        feat["debt_to_ebitda"],
        feat["interest_cover"],
        feat["current_ratio"],
        feat["leverage_change_1p"],
        feat["leverage_change_4p"],
        float(feat["sector_encoded"]),
        float(feat["region_encoded"]),
        float(feat["near_breach_count_4p"]),
    )


_N_FEATURES = 8


def _features_to_array(features: Sequence[LoanFeatures]) -> NDArray[np.float64]:
    """
    Convert sequence of LoanFeatures TypedDicts to numpy array.

    Returns shape (n_samples, 8) with columns in FEATURE_ORDER.
    """
    n_samples = len(features)
    result = np.zeros((n_samples, _N_FEATURES), dtype=np.float64)

    for i, feat in enumerate(features):
        values = _extract_feature_values(feat)
        for j, val in enumerate(values):
            result[i, j] = val

    return result


def _extract_positive_class_probabilities(proba: Proba2DProtocol) -> list[float]:
    """Extract positive class (column 1) probabilities from predict_proba output."""
    # XGBoost returns shape (n_samples, 2) for binary classification
    # Column 1 is probability of positive class (breach)
    n_samples = proba.shape[0]
    result: list[float] = []
    for i in range(n_samples):
        result.append(float(proba[i, 1]))
    return result


def predict_probabilities(
    model: XGBModelProtocol,
    features: Sequence[LoanFeatures],
) -> list[float]:
    """
    Predict breach probabilities for loan features.

    Returns list of probabilities (0.0 to 1.0) for each input sample.
    Probability represents likelihood of covenant breach.
    """
    if len(features) == 0:
        return []

    x_array = _features_to_array(features)
    proba = model.predict_proba(x_array)
    return _extract_positive_class_probabilities(proba)


def load_model(model_path: str) -> XGBModelProtocol:
    """
    Load XGBoost model from file path.

    Uses dynamic import to avoid xgboost dependency at module level.
    """
    xgb = __import__("xgboost")
    classifier_loader: XGBClassifierLoader = xgb.XGBClassifier
    model = classifier_loader()
    model.load_model(model_path)
    return model


__all__ = [
    "load_model",
    "predict_probabilities",
]
