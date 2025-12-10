"""XGBoost model training for covenant breach prediction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import TrainConfig, XGBClassifierFactory, XGBModelProtocol


def train_model(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    config: TrainConfig,
) -> XGBModelProtocol:
    """
    Train XGBoost classifier for breach prediction.

    x_features: Feature matrix of shape (n_samples, n_features)
    y_labels: Target labels of shape (n_samples,) with values 0 or 1
    config: Training hyperparameters

    Returns trained XGBClassifier model.
    """
    xgb = __import__("xgboost")
    classifier_factory: XGBClassifierFactory = xgb.XGBClassifier

    model = classifier_factory(
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        random_state=config["random_state"],
        objective="binary:logistic",
        eval_metric="logloss",
    )

    model.fit(x_features, y_labels)
    return model


def save_model(model: XGBModelProtocol, path: str) -> None:
    """Save trained model to file path."""
    model.save_model(path)


__all__ = [
    "save_model",
    "train_model",
]
