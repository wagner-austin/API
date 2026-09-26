"""Random Forest backend integration tests with actual sklearn training.

Tests the full training loop, prediction, and error paths using real US bankruptcy data.
Hyperparameter variations live in test_random_forest_hyperparams.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_ml import RequestedDevice, RequestedPrecision

from covenant_ml.backends.random_forest import (
    RANDOM_FOREST_CAPABILITIES,
    RandomForestBackend,
    create_random_forest_backend,
)
from covenant_ml.types import (
    BackendName,
    MLPConfig,
    OptimizerName,
    TrainProgress,
)
from tests.backends.random_forest._rf_fixtures import (
    _invoke_rf_train,
    _make_rf_config,
    _make_synthetic_dataset,
)

from ...conftest import load_us_bankruptcy_data


def test_rf_backend_train_returns_outcome(tmp_path: Path) -> None:
    """RandomForestBackend trains and returns TrainOutcome with all required fields."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=10, max_depth=5)

    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    # Verify outcome structure
    assert outcome["model_id"] == "random_forest"
    assert outcome["samples_total"] == len(y)
    assert outcome["samples_train"] > 0
    assert outcome["samples_val"] > 0
    assert outcome["samples_test"] > 0

    # Verify metrics exist and are reasonable
    assert 0.0 <= outcome["train_metrics"]["auc"] <= 1.0
    assert 0.0 <= outcome["val_metrics"]["auc"] <= 1.0
    assert 0.0 <= outcome["test_metrics"]["auc"] <= 1.0
    assert outcome["best_val_auc"] > 0.5  # Should beat random

    # Verify model was saved
    assert Path(outcome["model_path"]).exists()

    # Verify feature importances exist
    assert len(outcome["feature_importances"]) == len(names)
    assert outcome["feature_importances"][0]["rank"] == 1


def test_rf_backend_config_type_validation(tmp_path: Path) -> None:
    """RandomForestBackend raises on non-RandomForest config."""
    backend = create_random_forest_backend()
    x, y, names = _make_synthetic_dataset()

    # Try MLP config (wrong type)
    mlp_config: MLPConfig = {
        "device": RequestedDevice.CPU,
        "precision": RequestedPrecision.FP32,
        "optimizer": OptimizerName.ADAMW,
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

    with pytest.raises(RuntimeError, match="RandomForestBackend requires RandomForestConfig"):
        _invoke_rf_train(backend, x, y, names, mlp_config, tmp_path)


def test_rf_backend_predict_proba_after_train(tmp_path: Path) -> None:
    """RandomForestBackend trained model can predict probabilities."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=15)

    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    # Model should have learned and achieved reasonable AUC
    assert outcome["best_val_auc"] > 0.5
    # Verify loss decreased
    loss_initial = 0.693
    loss_final = outcome["val_metrics"]["loss"]
    assert loss_final < loss_initial, f"Loss should decrease: {loss_final} < {loss_initial}"


def test_rf_backend_evaluate_computes_metrics(tmp_path: Path) -> None:
    """RandomForestBackend.evaluate computes metrics using loaded model."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=15)
    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    # Load the trained model and evaluate
    loaded_model = backend.load(path=outcome["model_path"])
    metrics = backend.evaluate(model=loaded_model, x=x, y=y)

    # Metrics should be computed correctly
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["auc"] > 0.5  # Should beat random
    # Verify loss is reasonable
    assert metrics["loss"] > 0.0
    assert metrics["loss"] < 2.0


def test_rf_backend_prepare_raises() -> None:
    """RandomForestBackend.prepare raises RuntimeError (not supported)."""
    backend = create_random_forest_backend()

    with pytest.raises(RuntimeError, match="prepare not supported"):
        backend.prepare(n_features=10, n_classes=2, feature_names=None)


def test_rf_backend_save_raises(tmp_path: Path) -> None:
    """RandomForestBackend.save raises RuntimeError."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=5)
    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    # Load a trained model to pass to save
    loaded_model = backend.load(path=outcome["model_path"])

    with pytest.raises(RuntimeError, match="save not supported"):
        backend.save(model=loaded_model, path="/tmp/test.txt")


def test_rf_backend_load_and_predict(tmp_path: Path) -> None:
    """RandomForestBackend.load loads a trained model that can predict probabilities."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=10)
    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    # Load the trained model
    loaded_model = backend.load(path=outcome["model_path"])

    # Predict probabilities
    proba: NDArray[np.float64] = np.asarray(loaded_model.predict_proba(x), dtype=np.float64)

    # Verify shape and values
    n_samples = int(y.shape[0])
    assert proba.shape == (n_samples, 2), f"Expected shape ({n_samples}, 2), got {proba.shape}"
    min_val: float = float(np.min(proba))
    max_val: float = float(np.max(proba))
    assert min_val >= 0.0, "Probabilities must be >= 0"
    assert max_val <= 1.0, "Probabilities must be <= 1"
    # Probabilities should sum to 1 for each sample
    col0: NDArray[np.float64] = np.asarray(proba[:, 0], dtype=np.float64)
    col1: NDArray[np.float64] = np.asarray(proba[:, 1], dtype=np.float64)
    row_sums: NDArray[np.float64] = col0 + col1
    ones: NDArray[np.float64] = np.ones(n_samples, dtype=np.float64)
    assert np.allclose(row_sums, ones), "Probabilities should sum to 1"


def test_rf_backend_feature_importances_returns_none(tmp_path: Path) -> None:
    """RandomForestBackend.get_feature_importances returns None (provided via TrainOutcome)."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=5)
    outcome = _invoke_rf_train(backend, x, y, names, config, tmp_path)

    # Load trained model
    loaded_model = backend.load(path=outcome["model_path"])

    # get_feature_importances returns None (provided via TrainOutcome instead)
    result = backend.get_feature_importances(model=loaded_model, feature_names=names)
    assert result is None


def test_rf_backend_train_without_feature_names(tmp_path: Path) -> None:
    """RandomForestBackend generates feature names if not provided."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y = dataset["x"], dataset["y"]

    config = _make_rf_config(n_estimators=10)

    outcome = backend.train(
        x_features=x,
        y_labels=y,
        feature_names=None,  # Not provided
        config=config,
        output_dir=tmp_path,
        progress=None,
    )

    # Should generate f0, f1, f2, etc.
    assert outcome["feature_importances"][0]["name"].startswith("f")
    # Model should learn (val AUC beats random)
    assert outcome["best_val_auc"] > 0.5
    # Verify loss decreased from untrained baseline
    loss_initial = 0.693  # -log(0.5) for binary classification
    loss_final = outcome["val_metrics"]["loss"]
    assert loss_final < loss_initial, f"Loss should decrease: {loss_final} < {loss_initial}"


def test_rf_backend_raises_on_no_positive_samples(tmp_path: Path) -> None:
    """RandomForestBackend raises if training set has no positive samples."""
    backend = create_random_forest_backend()

    # Create dataset with no positives
    x = np.random.default_rng(42).standard_normal((100, 8)).astype(np.float64)
    y = np.zeros(100, dtype=np.int64)  # All negative
    names = [f"f{i}" for i in range(8)]

    config = _make_rf_config(n_estimators=5)

    with pytest.raises(ValueError, match="no positive samples"):
        _invoke_rf_train(backend, x, y, names, config, tmp_path)


def test_rf_backend_with_progress_callback(tmp_path: Path) -> None:
    """RandomForestBackend calls progress callback with training metrics."""
    backend = create_random_forest_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_rf_config(n_estimators=10)

    progress_reports: list[TrainProgress] = []

    def track_progress(p: TrainProgress) -> None:
        progress_reports.append(p)

    outcome = backend.train(
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=tmp_path,
        progress=track_progress,
    )

    # Should have exactly one progress report (at end of training)
    assert len(progress_reports) == 1
    assert progress_reports[0]["total_rounds"] == 1
    assert 0.0 <= progress_reports[0]["train_auc"] <= 1.0
    # Model should learn (val AUC beats random)
    assert outcome["best_val_auc"] > 0.5
    # Verify loss decreased from untrained baseline
    loss_initial = 0.693  # -log(0.5) for binary classification
    loss_final = outcome["val_metrics"]["loss"]
    assert loss_final < loss_initial, f"Loss should decrease: {loss_final} < {loss_initial}"


def test_rf_capabilities() -> None:
    """RANDOM_FOREST_CAPABILITIES has expected structure."""
    assert RANDOM_FOREST_CAPABILITIES["supports_train"] is True
    assert RANDOM_FOREST_CAPABILITIES["supports_gpu"] is False
    assert RANDOM_FOREST_CAPABILITIES["supports_early_stopping"] is False
    assert RANDOM_FOREST_CAPABILITIES["supports_feature_importance"] is True
    assert RANDOM_FOREST_CAPABILITIES["model_format"] == "joblib"


def test_rf_backend_name() -> None:
    """RandomForestBackend.backend_name returns BackendName.RANDOM_FOREST."""
    backend = create_random_forest_backend()
    assert backend.backend_name() is BackendName.RANDOM_FOREST


def test_rf_backend_capabilities() -> None:
    """RandomForestBackend.capabilities returns RANDOM_FOREST_CAPABILITIES."""
    backend = create_random_forest_backend()
    caps = backend.capabilities()
    assert caps == RANDOM_FOREST_CAPABILITIES


def test_rf_backend_class_instantiation() -> None:
    """RandomForestBackend can be instantiated directly."""
    backend = RandomForestBackend()
    assert backend.backend_name() is BackendName.RANDOM_FOREST
