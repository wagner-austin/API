"""Shared fixtures and helpers for test_trainer splits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.trainer_fit import (
    _XGBCoreProto,
)
from covenant_ml.types import (
    DMatrixFactory,
    DMatrixProtocol,
    XGBBoosterProtocol,
    XGBClassifierFactory,
    XGBModelProtocol,
    XGBParams,
)


def _make_training_data(
    n_samples: int = 20,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Create simple training data for binary classification."""
    x_features: NDArray[np.float64] = np.zeros((n_samples, 8), dtype=np.float64)
    y_labels: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)

    # Deterministic data generation based on index
    for i in range(n_samples):
        first_feat = ((i + seed) % 100) / 100.0
        for j in range(8):
            x_features[i, j] = (first_feat + j * 0.1) % 1.0
        # Label based on first feature (computed before assignment)
    y_labels[i] = 1 if first_feat > 0.5 else 0

    return x_features, y_labels


class _FakeCore:
    def __init__(self, available: bool) -> None:
        self._available = available

    def build_info(self) -> dict[str, bool]:
        return {"USE_CUDA": self._available}


class _FakeDMatrix:
    def set_info(self, *, feature_names: list[str] | None) -> None:
        _ = feature_names


class _FakeDMatrixFactory:
    def __call__(self, data: NDArray[np.float64]) -> DMatrixProtocol:
        _ = data
        return _FakeDMatrix()


class _FakeBooster:
    def save_model(self, fname: str) -> None:
        _ = fname

    def predict(self, data: DMatrixProtocol) -> NDArray[np.float32]:
        _ = data
        result: NDArray[np.float32] = np.zeros(1, dtype=np.float32)
        result[0] = 0.5
        return result


class _FakeXGBModel:
    def __init__(
        self,
        *,
        n_jobs: int,
        tree_method: str,
        device: str,
        reg_alpha: float,
        reg_lambda: float,
    ) -> None:
        self._params: XGBParams = XGBParams(
            n_jobs=n_jobs,
            tree_method=tree_method,
            device=device,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )
        self._booster = _FakeBooster()
        self._feature_importances: NDArray[np.float32] = np.zeros(2, dtype=np.float32)
        self._feature_importances[0] = 0.5
        self._feature_importances[1] = 0.5

    @property
    def feature_importances_(self) -> NDArray[np.float32]:
        return self._feature_importances

    def fit(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        *,
        verbose: bool = False,
    ) -> XGBModelProtocol:
        _ = x_features, y_labels, verbose
        return self

    def predict_proba(self, x_features: NDArray[np.float64]) -> NDArray[np.float64]:
        _ = x_features
        return np.zeros((1, 2), dtype=np.float64)

    def get_xgb_params(self) -> XGBParams:
        return self._params

    def save_model(self, fname: str) -> None:
        self._booster.save_model(fname)

    def load_model(self, fname: str) -> None:
        _ = fname

    def get_booster(self) -> XGBBoosterProtocol:
        return self._booster


class _FakeClassifierFactory:
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
        n_jobs: int,
        tree_method: str,
        device: str,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: float | None = None,
    ) -> XGBModelProtocol:
        _ = (
            learning_rate,
            max_depth,
            n_estimators,
            subsample,
            colsample_bytree,
            random_state,
            objective,
            eval_metric,
            n_jobs,
            tree_method,
            device,
            reg_alpha,
            reg_lambda,
            scale_pos_weight,
        )
        return _FakeXGBModel(
            n_jobs=n_jobs,
            tree_method=tree_method,
            device=device,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )


class _FakeXGBModule:
    def __init__(self, available: bool) -> None:
        self.core: _XGBCoreProto = _FakeCore(available)
        self.XGBClassifier: XGBClassifierFactory = _FakeClassifierFactory()
        self.DMatrix: DMatrixFactory = _FakeDMatrixFactory()


def _make_larger_data(
    n_samples: int = 100,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Create larger training data with balanced classes."""
    x_features: NDArray[np.float64] = np.zeros((n_samples, 8), dtype=np.float64)
    y_labels: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        first_feat = ((i + seed) % 100) / 100.0
        for j in range(8):
            x_features[i, j] = (first_feat + j * 0.1) % 1.0
        # 50% positive labels
        y_labels[i] = 1 if i % 2 == 0 else 0

    return x_features, y_labels


def _make_imbalanced_data(
    n_samples: int = 100,
    positive_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Create imbalanced training data for class weight testing."""
    x_features: NDArray[np.float64] = np.zeros((n_samples, 8), dtype=np.float64)
    y_labels: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)

    n_positive = int(n_samples * positive_ratio)
    for i in range(n_samples):
        first_feat = ((i + seed) % 100) / 100.0
        for j in range(8):
            x_features[i, j] = (first_feat + j * 0.1) % 1.0
        # Create imbalanced labels
        y_labels[i] = 1 if i < n_positive else 0

    return x_features, y_labels


def _make_regression_data(
    n_samples: int = 100,
    n_features: int = 8,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create regression data with a linear relationship."""
    x: NDArray[np.float64] = np.zeros(
        (n_samples, n_features),
        dtype=np.float64,
    )
    y: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        for j in range(n_features):
            x[i, j] = ((i + seed + j * 7) % 100) / 100.0
        # Linear target with some variation
        row: NDArray[np.float64] = x[i]
        feat0: float = float(row.flat[0])
        feat1: float = float(row.flat[1])
        y[i] = feat0 * 2.0 + feat1 * 0.5 + 1.0

    return x, y
