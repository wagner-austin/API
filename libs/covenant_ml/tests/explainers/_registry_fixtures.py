"""Shared fixtures and helpers for test_registry splits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from platform_ml.explainers import FeatureExplainer

from covenant_ml.backends.protocol import PreparedClassifier
from covenant_ml.explainers.registry import (
    ExplainerFactory,
)
from covenant_ml.types import XGBModelProtocol


class _FakePredictorWithGradients:
    """Fake predictor implementing PredictorProtocol with compute_gradients."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return uniform probabilities."""
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.full((n_samples, 2), 0.5, dtype=np.float64)
        return proba

    def compute_gradients(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Return gradients proportional to feature index."""
        n_samples = int(x.shape[0])
        n_features = int(x.shape[1])
        grads: NDArray[np.float64] = np.zeros((n_samples, n_features), dtype=np.float64)
        for f in range(n_features):
            # Gradient magnitude proportional to feature index
            grads[:, f] = float(f + 1) * 0.1
        return grads


class _FakePredictorNoGradients:
    """Fake predictor without compute_gradients method."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return uniform probabilities."""
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.full((n_samples, 2), 0.5, dtype=np.float64)
        return proba


def _create_tree_predictor() -> XGBModelProtocol:
    """Create a real XGBoost model for SHAP tree tests.

    Uses the covenant_ml xgboost backend for training, then loads
    the model directly via xgboost.XGBClassifier for SHAP compatibility.

    Requires SHAP >= 0.50 for XGBoost 3.x compatibility.

    Returns:
        XGBoost trained model implementing XGBModelProtocol.
    """
    import tempfile
    from pathlib import Path

    from covenant_ml.backends.protocol import ClassifierBackend
    from covenant_ml.backends.xgboost import create_xgboost_backend
    from covenant_ml.predictor import load_model
    from covenant_ml.types import TrainConfig

    rng = np.random.default_rng(42)
    x_train = rng.random((100, 4)).astype(np.float64)
    y_train = rng.integers(0, 2, size=100).astype(np.int64)
    feature_names = ["f0", "f1", "f2", "f3"]

    backend: ClassifierBackend = create_xgboost_backend()

    with tempfile.TemporaryDirectory() as tmp_dir:
        config: TrainConfig = {
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "random_state": 42,
            "early_stopping_rounds": 3,
            "device": "cpu",
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }

        outcome = backend.train(
            x_features=x_train,
            y_labels=y_train,
            feature_names=feature_names,
            config=config,
            output_dir=Path(tmp_dir),
            progress=None,
        )

        # Load model directly using XGBClassifier (not backend.load())
        # This creates a real XGBoost model that SHAP TreeExplainer can use
        return load_model(outcome["model_path"])


def _make_simple_explainer_factory() -> ExplainerFactory:
    """Create a factory that returns a simple explainer.

    Returns:
        Factory function that creates permutation explainer.
    """
    from platform_ml.explainers import PermutationConfig, create_permutation_explainer

    def factory() -> FeatureExplainer:
        config: PermutationConfig = {"n_repeats": 2, "random_state": 42}
        return create_permutation_explainer(config)

    return factory


def _create_cleargbm_prepared() -> PreparedClassifier:
    """Create a ClearGBM prepared classifier for tests.

    Trains a real native ``PyGbmModel`` via ``train_gradient_boosting``
    (matching the ClearGBM backend's own training path), wraps it in
    ``_ClearGBMPrepared``, and returns it. Bypasses save/load so the test
    exercises the wrapper's in-memory instance directly.

    Returns:
        PreparedClassifier wrapping a ClearGBM native model.
    """
    from cleargbm.ensemble import train_gradient_boosting
    from cleargbm.types import GradientBoostingConfig

    from covenant_ml.backends.cleargbm.backend import _ClearGBMPrepared

    rng = np.random.default_rng(42)
    x_train: NDArray[np.float64] = rng.random((100, 4)).astype(np.float64)
    y_train: NDArray[np.int64] = rng.integers(0, 2, size=100).astype(np.int64)

    feature_names: tuple[str, ...] = ("f0", "f1", "f2", "f3")

    config: GradientBoostingConfig = GradientBoostingConfig(
        n_estimators=5,
        max_depth=3,
        learning_rate=0.1,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        colsample_bytree=None,
        categorical_features=None,
        n_classes=None,
        max_bins=64,
        subsample=1.0,
        random_state=42,
        monotonic_constraints=None,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=1,
        early_stopping_rounds=10,
        growth_strategy="depth_wise",
        num_leaves=None,
        objective="binary_log_loss",
        scale_pos_weight=1.0,
    )

    native_model = train_gradient_boosting(
        x_train=x_train,
        y_train=y_train,
        x_val=None,
        y_val=None,
        config=config,
        feature_names=feature_names,
    )

    return _ClearGBMPrepared(native_model)
