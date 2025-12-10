"""Type definitions for covenant ML training and prediction."""

from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray


class TrainConfig(TypedDict, total=True):
    """Configuration for XGBoost model training."""

    learning_rate: float
    max_depth: int
    n_estimators: int
    subsample: float
    colsample_bytree: float
    random_state: int


class Proba2DProtocol(Protocol):
    """Protocol for 2D probability array from predict_proba.

    predict_proba returns shape (n_samples, n_classes).
    For binary classification: (n_samples, 2).
    """

    @property
    def shape(self) -> tuple[int, int]: ...

    def __getitem__(self, idx: tuple[int, int]) -> float: ...


class XGBModelProtocol(Protocol):
    """Protocol for XGBoost classifier interface."""

    def fit(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
    ) -> XGBModelProtocol: ...

    def predict_proba(
        self,
        x_features: NDArray[np.float64],
    ) -> Proba2DProtocol: ...

    def save_model(self, fname: str) -> None: ...

    def load_model(self, fname: str) -> None: ...


class XGBClassifierFactory(Protocol):
    """Protocol for XGBClassifier constructor."""

    def __call__(
        self,
        *,
        learning_rate: float,
        max_depth: int,
        n_estimators: int,
        subsample: float,
        colsample_bytree: float,
        random_state: int,
        objective: str,
        eval_metric: str,
    ) -> XGBModelProtocol: ...


class XGBClassifierLoader(Protocol):
    """Protocol for XGBClassifier loader (no-arg constructor)."""

    def __call__(self) -> XGBModelProtocol: ...


__all__ = [
    "Proba2DProtocol",
    "TrainConfig",
    "XGBClassifierFactory",
    "XGBClassifierLoader",
    "XGBModelProtocol",
]
