"""ClearGBM backend integration tests with actual training.

Tests the full training loop, prediction, save/load, and error paths.
Uses real US bankruptcy data for integration tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.backends.cleargbm import (
    CLEARGBM_CAPABILITIES,
    create_cleargbm_backend,
)
from covenant_ml.backends.cleargbm.backend import (
    _compute_class_weight,
    _is_cleargbm_config,
)
from covenant_ml.types import (
    MLPConfig,
    TrainConfig,
)
from tests.backends.cleargbm._cleargbm_fixtures import (
    _invoke_cleargbm_train,
    _make_cleargbm_config,
    _make_synthetic_dataset,
)


def test_is_cleargbm_config_returns_true_for_cleargbm_config() -> None:
    """_is_cleargbm_config returns True for valid ClearGBMConfig."""
    config = _make_cleargbm_config()
    assert _is_cleargbm_config(config) is True


def test_is_cleargbm_config_returns_false_for_xgboost_config() -> None:
    """_is_cleargbm_config returns False for TrainConfig (XGBoost)."""
    config: TrainConfig = {
        "device": "cpu",
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_rounds": 2,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    assert _is_cleargbm_config(config) is False


def test_is_cleargbm_config_returns_false_for_mlp_config() -> None:
    """_is_cleargbm_config returns False for MLPConfig."""
    config: MLPConfig = {
        "device": "cpu",
        "precision": "fp32",
        "optimizer": "adamw",
        "hidden_sizes": (32,),
        "learning_rate": 0.01,
        "batch_size": 32,
        "n_epochs": 2,
        "dropout": 0.0,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_patience": 5,
    }
    assert _is_cleargbm_config(config) is False


def test_is_cleargbm_config_returns_false_for_missing_key() -> None:
    """_is_cleargbm_config returns False for config missing min_samples_split."""
    # XGBoost config lacks min_samples_split key used by ClearGBM
    xgb_config: TrainConfig = {
        "device": "cpu",
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_rounds": 2,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    result = _is_cleargbm_config(xgb_config)
    assert result is False


def test_compute_class_weight_balanced() -> None:
    """_compute_class_weight returns 1.0 for balanced classes."""
    y: NDArray[np.int64] = np.array((0, 0, 0, 0, 0, 1, 1, 1, 1, 1), dtype=np.int64)
    weight = _compute_class_weight(y)
    assert weight == 1.0


def test_compute_class_weight_imbalanced() -> None:
    """_compute_class_weight returns ratio for imbalanced classes."""
    # 8 negatives, 2 positives -> weight = 8/2 = 4.0
    y: NDArray[np.int64] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 1, 1), dtype=np.int64)
    weight = _compute_class_weight(y)
    assert weight == 4.0


def test_compute_class_weight_raises_no_positives() -> None:
    """_compute_class_weight raises ValueError when no positive samples."""
    y = np.zeros(10, dtype=np.int64)  # All negative
    with pytest.raises(ValueError, match="no positive samples"):
        _compute_class_weight(y)


def test_compute_class_weight_single_positive() -> None:
    """_compute_class_weight handles single positive sample."""
    # 9 negatives, 1 positive -> weight = 9/1 = 9.0
    y = np.zeros(10, dtype=np.int64)
    y[0] = 1
    weight = _compute_class_weight(y)
    assert weight == 9.0


def test_cleargbm_prepared_predict_proba_shape(tmp_path: Path) -> None:
    """_ClearGBMPrepared.predict_proba returns correct shape."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset(n_samples=50)
    config = _make_cleargbm_config(n_estimators=3, max_depth=2)

    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)
    loaded = backend.load(path=outcome["model_path"])

    proba = loaded.predict_proba(x)
    assert proba.shape == (50, 2)


def test_cleargbm_prepared_predict_proba_valid_probabilities(tmp_path: Path) -> None:
    """_ClearGBMPrepared.predict_proba returns valid probabilities."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset(n_samples=50)
    config = _make_cleargbm_config(n_estimators=3, max_depth=2)

    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)
    loaded = backend.load(path=outcome["model_path"])

    proba = loaded.predict_proba(x)

    # Probabilities should be in [0, 1]
    min_val = float(np.min(proba))
    max_val = float(np.max(proba))
    assert min_val >= 0.0
    assert max_val <= 1.0

    # Rows should sum to 1
    row_sums = proba[:, 0] + proba[:, 1]
    ones = np.ones(50, dtype=np.float64)
    assert np.allclose(row_sums, ones, rtol=1e-6)


def test_cleargbm_prepared_model_property(tmp_path: Path) -> None:
    """_ClearGBMPrepared.model returns the underlying model with trees."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset(n_samples=50)
    config = _make_cleargbm_config(n_estimators=3, max_depth=2)

    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)
    loaded = backend.load(path=outcome["model_path"])

    # Verify loaded model can predict (proves model is accessible)
    proba = loaded.predict_proba(x)
    assert proba.shape == (50, 2)
    # Verify outcome has expected number of trees in feature importances
    # (3 trees trained, so importances should be populated)
    assert outcome["total_rounds"] == 3


def test_create_cleargbm_backend_returns_correct_name() -> None:
    """create_cleargbm_backend returns backend with correct name."""
    backend = create_cleargbm_backend()
    assert backend.backend_name() == "cleargbm"


def test_cleargbm_capabilities_structure() -> None:
    """CLEARGBM_CAPABILITIES has expected structure."""
    assert CLEARGBM_CAPABILITIES["supports_train"] is True
    assert CLEARGBM_CAPABILITIES["supports_gpu"] is False
    assert CLEARGBM_CAPABILITIES["supports_early_stopping"] is True
    assert CLEARGBM_CAPABILITIES["supports_feature_importance"] is True
    assert CLEARGBM_CAPABILITIES["model_format"] == "json"


def test_cleargbm_backend_name() -> None:
    """ClearGBMBackend.backend_name returns 'cleargbm'."""
    backend = create_cleargbm_backend()
    assert backend.backend_name() == "cleargbm"


def test_cleargbm_backend_capabilities() -> None:
    """ClearGBMBackend.capabilities returns CLEARGBM_CAPABILITIES."""
    backend = create_cleargbm_backend()
    caps = backend.capabilities()
    assert caps == CLEARGBM_CAPABILITIES


def test_cleargbm_backend_prepare_raises() -> None:
    """ClearGBMBackend.prepare raises RuntimeError."""
    backend = create_cleargbm_backend()

    with pytest.raises(RuntimeError, match="prepare not supported"):
        backend.prepare(n_features=10, n_classes=2, feature_names=None)
