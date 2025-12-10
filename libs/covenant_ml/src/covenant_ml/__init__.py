"""XGBoost wrapper for covenant breach risk prediction."""

from __future__ import annotations

from .predictor import load_model, predict_probabilities
from .trainer import save_model, train_model
from .types import (
    Proba2DProtocol,
    TrainConfig,
    XGBClassifierFactory,
    XGBClassifierLoader,
    XGBModelProtocol,
)

__all__ = [
    "Proba2DProtocol",
    "TrainConfig",
    "XGBClassifierFactory",
    "XGBClassifierLoader",
    "XGBModelProtocol",
    "load_model",
    "predict_probabilities",
    "save_model",
    "train_model",
]
