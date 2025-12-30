"""Logistic Regression backend integration tests with actual sklearn training.

Tests the full training loop, prediction, and error paths using real US bankruptcy data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.backends.logreg import (
    LOGREG_CAPABILITIES,
    LogRegBackend,
    create_logreg_backend,
)
from covenant_ml.backends.protocol import ClassifierBackend
from covenant_ml.types import (
    ClassifierTrainConfig,
    LogRegConfig,
    MLPConfig,
    TrainOutcome,
    TrainProgress,
)

from ...conftest import load_us_bankruptcy_data

# Type aliases for LogReg config literals
LogRegSolver = Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
LogRegPenalty = Literal["l1", "l2", "elasticnet", "none"]


def _invoke_logreg_train(
    backend: ClassifierBackend,
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    names: list[str] | None,
    config: ClassifierTrainConfig,
    output_dir: Path,
) -> TrainOutcome:
    """Helper to invoke backend train (isolates .train() call for guard).

    Args:
        backend: Classifier backend to use.
        x: Feature matrix.
        y: Labels.
        names: Feature names.
        config: Training configuration.
        output_dir: Output directory for model artifacts.

    Returns:
        TrainOutcome from the training run.
    """
    return backend.train(
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=output_dir,
        progress=None,
    )


def _make_synthetic_dataset(
    n_samples: int = 100,
    n_features: int = 8,
    pos_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Create synthetic binary classification dataset for edge case tests.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        pos_ratio: Ratio of positive samples.
        seed: Random seed.

    Returns:
        Tuple of (features, labels, feature_names).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    n_pos = int(n_samples * pos_ratio)
    y = np.zeros(n_samples, dtype=np.int64)
    y[:n_pos] = 1
    rng.shuffle(y)
    feature_names = [f"f{i}" for i in range(n_features)]
    return x, y, feature_names


def _make_logreg_config(
    penalty: LogRegPenalty = "l2",
    c_value: float = 1.0,
    solver: LogRegSolver = "lbfgs",
    max_iter: int = 100,
) -> LogRegConfig:
    """Create LogReg config for testing.

    Args:
        penalty: Regularization penalty type.
        c_value: Inverse regularization strength.
        solver: Solver algorithm.
        max_iter: Maximum iterations.

    Returns:
        LogRegConfig for testing.
    """
    return {
        "solver": solver,
        "penalty": penalty,
        "C": c_value,
        "max_iter": max_iter,
        "tol": 1e-4,
        "l1_ratio": 0.5,
        "class_weight_balanced": True,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
    }


def test_logreg_backend_train_returns_outcome(tmp_path: Path) -> None:
    """LogRegBackend trains and returns TrainOutcome with all required fields."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config(penalty="l2", c_value=1.0)

    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    # Verify outcome structure
    assert outcome["model_id"] == "logreg"
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


def test_logreg_backend_config_type_validation(tmp_path: Path) -> None:
    """LogRegBackend raises on non-LogReg config."""
    backend = create_logreg_backend()
    x, y, names = _make_synthetic_dataset()

    # Try MLP config (wrong type)
    mlp_config: MLPConfig = {
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

    with pytest.raises(RuntimeError, match="LogRegBackend requires LogRegConfig"):
        _invoke_logreg_train(backend, x, y, names, mlp_config, tmp_path)


def test_logreg_backend_predict_proba_after_train(tmp_path: Path) -> None:
    """LogRegBackend trained model can predict probabilities."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config()

    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    # Model should have learned and achieved reasonable AUC
    assert outcome["best_val_auc"] > 0.5
    # Verify loss decreased
    loss_initial = 0.693
    loss_final = outcome["val_metrics"]["loss"]
    assert loss_final < loss_initial, f"Loss should decrease: {loss_final} < {loss_initial}"


def test_logreg_backend_evaluate_computes_metrics(tmp_path: Path) -> None:
    """LogRegBackend.evaluate computes metrics using loaded model."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config()
    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    # Load the trained model and evaluate
    loaded_model = backend.load(path=outcome["model_path"])
    metrics = backend.evaluate(model=loaded_model, x=x, y=y)

    # Metrics should be computed correctly
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["auc"] > 0.5  # Should beat random
    # Verify loss is reasonable
    assert metrics["loss"] > 0.0
    assert metrics["loss"] < 2.0  # Should be better than terrible


def test_logreg_backend_prepare_raises() -> None:
    """LogRegBackend.prepare raises RuntimeError (not supported for LogReg)."""
    backend = create_logreg_backend()

    with pytest.raises(RuntimeError, match="prepare not supported"):
        backend.prepare(n_features=10, n_classes=2, feature_names=None)


def test_logreg_backend_save_raises(tmp_path: Path) -> None:
    """LogRegBackend.save raises RuntimeError."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config()
    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    # Load a trained model to pass to save
    loaded_model = backend.load(path=outcome["model_path"])

    with pytest.raises(RuntimeError, match="save not supported"):
        backend.save(model=loaded_model, path="/tmp/test.txt")


def test_logreg_backend_load_and_predict(tmp_path: Path) -> None:
    """LogRegBackend.load loads a trained model that can predict probabilities."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config()
    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

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


def test_logreg_backend_feature_importances_returns_none(tmp_path: Path) -> None:
    """LogRegBackend.get_feature_importances returns None (provided via TrainOutcome)."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config()
    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    # Load trained model
    loaded_model = backend.load(path=outcome["model_path"])

    # get_feature_importances returns None (provided via TrainOutcome instead)
    result = backend.get_feature_importances(model=loaded_model, feature_names=names)
    assert result is None


def test_logreg_backend_train_without_feature_names(tmp_path: Path) -> None:
    """LogRegBackend generates feature names if not provided."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y = dataset["x"], dataset["y"]

    config = _make_logreg_config()

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


def test_logreg_backend_raises_on_no_positive_samples(tmp_path: Path) -> None:
    """LogRegBackend raises if training set has no positive samples."""
    backend = create_logreg_backend()

    # Create dataset with no positives
    x = np.random.default_rng(42).standard_normal((100, 8)).astype(np.float64)
    y = np.zeros(100, dtype=np.int64)  # All negative
    names = [f"f{i}" for i in range(8)]

    config = _make_logreg_config()

    with pytest.raises(ValueError, match="no positive samples"):
        _invoke_logreg_train(backend, x, y, names, config, tmp_path)


def test_logreg_backend_with_progress_callback(tmp_path: Path) -> None:
    """LogRegBackend calls progress callback with training metrics."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config()

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


def test_logreg_capabilities() -> None:
    """LOGREG_CAPABILITIES has expected structure."""
    assert LOGREG_CAPABILITIES["supports_train"] is True
    assert LOGREG_CAPABILITIES["supports_gpu"] is False
    assert LOGREG_CAPABILITIES["supports_early_stopping"] is False
    assert LOGREG_CAPABILITIES["supports_feature_importance"] is True
    assert LOGREG_CAPABILITIES["model_format"] == "joblib"


def test_logreg_backend_name() -> None:
    """LogRegBackend.backend_name returns 'logreg'."""
    backend = create_logreg_backend()
    assert backend.backend_name() == "logreg"


def test_logreg_backend_capabilities() -> None:
    """LogRegBackend.capabilities returns LOGREG_CAPABILITIES."""
    backend = create_logreg_backend()
    caps = backend.capabilities()
    assert caps == LOGREG_CAPABILITIES


def test_logreg_backend_class_instantiation() -> None:
    """LogRegBackend can be instantiated directly."""
    backend = LogRegBackend()
    assert backend.backend_name() == "logreg"


# =============================================================================
# Penalty Variations Tests
# =============================================================================


def test_logreg_backend_with_l1_penalty(tmp_path: Path) -> None:
    """LogRegBackend works with L1 penalty (saga solver)."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config(penalty="l1", solver="saga", max_iter=500)

    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    assert outcome["best_val_auc"] > 0.5
    assert Path(outcome["model_path"]).exists()


def test_logreg_backend_with_no_penalty(tmp_path: Path) -> None:
    """LogRegBackend works with no penalty."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config(penalty="none", solver="lbfgs")

    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    assert outcome["best_val_auc"] > 0.5


def test_logreg_backend_with_elasticnet_penalty(tmp_path: Path) -> None:
    """LogRegBackend works with ElasticNet penalty (saga solver)."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config(penalty="elasticnet", solver="saga", max_iter=500)
    config["l1_ratio"] = 0.5  # Mix of L1 and L2

    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    assert outcome["best_val_auc"] > 0.5


def test_logreg_backend_with_strong_regularization(tmp_path: Path) -> None:
    """LogRegBackend works with strong regularization (small C)."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config(c_value=0.01)  # Strong regularization

    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    # Should still produce valid output
    assert 0.0 <= outcome["best_val_auc"] <= 1.0
    assert Path(outcome["model_path"]).exists()


def test_logreg_backend_without_class_weight_balance(tmp_path: Path) -> None:
    """LogRegBackend works without class weight balancing."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config()
    config["class_weight_balanced"] = False

    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    assert outcome["best_val_auc"] > 0.5


def test_logreg_backend_feature_importance_ranking(tmp_path: Path) -> None:
    """LogRegBackend produces correctly ranked feature importances."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config()
    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    importances = outcome["feature_importances"]

    # Verify ranks are correct
    for i, feat in enumerate(importances):
        assert feat["rank"] == i + 1

    # Verify sorted by importance (descending)
    for i in range(len(importances) - 1):
        assert importances[i]["importance"] >= importances[i + 1]["importance"]


def test_logreg_backend_config_stored_in_outcome(tmp_path: Path) -> None:
    """LogRegBackend stores config in TrainOutcome."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config(penalty="l2", c_value=0.5)
    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    # Verify config matches what was passed in (compare against shared keys)
    assert outcome["config"]["random_state"] == config["random_state"]
    assert outcome["config"]["train_ratio"] == config["train_ratio"]


def test_logreg_backend_scale_pos_weight_computed(tmp_path: Path) -> None:
    """LogRegBackend computes and stores scale_pos_weight."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config()
    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    # Verify scale_pos_weight_computed is positive
    assert outcome["scale_pos_weight_computed"] > 0.0


def test_logreg_backend_single_round_training(tmp_path: Path) -> None:
    """LogRegBackend always reports single round training."""
    backend = create_logreg_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_logreg_config()
    outcome = _invoke_logreg_train(backend, x, y, names, config, tmp_path)

    # LogReg is single-round training
    assert outcome["best_round"] == 1
    assert outcome["total_rounds"] == 1
    assert outcome["early_stopped"] is False
