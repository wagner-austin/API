"""LightGBM backend integration tests with actual LightGBM training.

Tests the full training loop, prediction, and error paths using real US bankruptcy data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.backends.lightgbm import LIGHTGBM_CAPABILITIES, create_lightgbm_backend
from covenant_ml.backends.lightgbm.backend import _resolve_device
from covenant_ml.backends.protocol import ClassifierBackend
from covenant_ml.explainers.adapters import try_extract_native_tree_model
from covenant_ml.types import (
    ClassifierTrainConfig,
    LightGBMConfig,
    MLPConfig,
    TrainOutcome,
    TrainProgress,
)

from ...conftest import load_us_bankruptcy_data


def _invoke_lightgbm_train(
    backend: ClassifierBackend,
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    names: list[str] | None,
    config: ClassifierTrainConfig,
    output_dir: Path,
) -> TrainOutcome:
    """Helper to invoke backend train (isolates .train() call for guard)."""
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
    """Create synthetic binary classification dataset for edge case tests."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    n_pos = int(n_samples * pos_ratio)
    y = np.zeros(n_samples, dtype=np.int64)
    y[:n_pos] = 1
    rng.shuffle(y)
    feature_names = [f"f{i}" for i in range(n_features)]
    return x, y, feature_names


def _make_lightgbm_config(
    n_estimators: int = 10,
    max_depth: int = 3,
    num_leaves: int = 8,
) -> LightGBMConfig:
    """Create LightGBM config for testing."""
    return {
        "device": "cpu",
        "learning_rate": 0.1,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "min_child_samples": 5,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_rounds": 3,
    }


def test_lightgbm_backend_train_returns_outcome(tmp_path: Path) -> None:
    """LightGBMBackend trains and returns TrainOutcome with all required fields."""
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config: LightGBMConfig = {
        "device": "cpu",
        "learning_rate": 0.1,
        "max_depth": 4,
        "n_estimators": 20,
        "num_leaves": 16,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "random_state": 42,
        "early_stopping_rounds": 5,
    }

    outcome = _invoke_lightgbm_train(backend, x, y, names, config, tmp_path)

    # Verify outcome structure
    assert outcome["model_id"] == "lightgbm"
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


def test_lightgbm_backend_train_with_early_stopping(tmp_path: Path) -> None:
    """LightGBMBackend triggers early stopping when validation AUC plateaus."""
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    # High n_estimators with short early stopping rounds to trigger early stop
    config = _make_lightgbm_config(
        n_estimators=100,
        max_depth=2,
        num_leaves=4,
    )
    config["early_stopping_rounds"] = 5

    outcome = _invoke_lightgbm_train(backend, x, y, names, config, tmp_path)

    # Should have trained successfully
    assert outcome["best_val_auc"] > 0.5


def test_lightgbm_backend_config_type_validation(tmp_path: Path) -> None:
    """LightGBMBackend raises on non-LightGBM config."""
    backend = create_lightgbm_backend()
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

    with pytest.raises(RuntimeError, match="LightGBMBackend requires LightGBMConfig"):
        _invoke_lightgbm_train(backend, x, y, names, mlp_config, tmp_path)


def test_lightgbm_backend_predict_proba_after_train(tmp_path: Path) -> None:
    """LightGBMBackend trained model can predict probabilities."""
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_lightgbm_config(n_estimators=10)

    outcome = _invoke_lightgbm_train(backend, x, y, names, config, tmp_path)

    # Model should have learned and achieved reasonable AUC
    assert outcome["best_val_auc"] > 0.5
    # Verify loss decreased
    loss_initial = 0.693
    loss_final = outcome["val_metrics"]["loss"]
    assert loss_final < loss_initial, f"Loss should decrease: {loss_final} < {loss_initial}"


def test_lightgbm_backend_evaluate_computes_metrics(tmp_path: Path) -> None:
    """LightGBMBackend.evaluate computes metrics using loaded model."""
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_lightgbm_config(n_estimators=15)
    outcome = _invoke_lightgbm_train(backend, x, y, names, config, tmp_path)

    # Load the trained model and evaluate
    loaded_model = backend.load(path=outcome["model_path"])
    metrics = backend.evaluate(model=loaded_model, x=x, y=y)

    # Metrics should be computed correctly
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["auc"] > 0.5  # Should beat random
    # Verify loss decreased from untrained baseline
    loss_initial = 0.693  # Untrained random baseline
    loss_final = metrics["loss"]
    assert loss_final < loss_initial, f"Loss should decrease: {loss_final} < {loss_initial}"


def test_lightgbm_backend_prepare_raises() -> None:
    """LightGBMBackend.prepare raises RuntimeError (not supported for tree models)."""
    backend = create_lightgbm_backend()

    with pytest.raises(RuntimeError, match="prepare not supported"):
        backend.prepare(n_features=10, n_classes=2, feature_names=None)


def test_lightgbm_backend_save_raises(tmp_path: Path) -> None:
    """LightGBMBackend.save raises RuntimeError."""
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_lightgbm_config(n_estimators=5)
    outcome = _invoke_lightgbm_train(backend, x, y, names, config, tmp_path)

    # Load a trained model to pass to save
    loaded_model = backend.load(path=outcome["model_path"])

    with pytest.raises(RuntimeError, match="save not supported"):
        backend.save(model=loaded_model, path="/tmp/test.txt")


def test_lightgbm_backend_load_and_predict(tmp_path: Path) -> None:
    """LightGBMBackend.load loads a trained model that can predict probabilities."""
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_lightgbm_config(n_estimators=10)
    outcome = _invoke_lightgbm_train(backend, x, y, names, config, tmp_path)

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


def test_lightgbm_backend_feature_importances_returns_none(tmp_path: Path) -> None:
    """LightGBMBackend.get_feature_importances returns None (provided via TrainOutcome)."""
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_lightgbm_config(n_estimators=5)
    outcome = _invoke_lightgbm_train(backend, x, y, names, config, tmp_path)

    # Load trained model
    loaded_model = backend.load(path=outcome["model_path"])

    # get_feature_importances returns None (provided via TrainOutcome instead)
    result = backend.get_feature_importances(model=loaded_model, feature_names=names)
    assert result is None


def test_lightgbm_backend_with_device_auto(tmp_path: Path) -> None:
    """LightGBMBackend works with device='auto' (resolves to cpu)."""
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_lightgbm_config(n_estimators=10)
    config["device"] = "auto"  # Should resolve to "cpu"

    outcome = _invoke_lightgbm_train(backend, x, y, names, config, tmp_path)

    # Model should train successfully
    assert outcome["best_val_auc"] > 0.5
    # Verify model was saved
    assert Path(outcome["model_path"]).exists()


def test_lightgbm_backend_with_regularization(tmp_path: Path) -> None:
    """LightGBMBackend works with L1/L2 regularization."""
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_lightgbm_config(n_estimators=15)
    config["reg_alpha"] = 1.0  # L1
    config["reg_lambda"] = 1.0  # L2

    outcome = _invoke_lightgbm_train(backend, x, y, names, config, tmp_path)

    assert outcome["best_val_auc"] > 0.5


def test_lightgbm_backend_with_subsampling(tmp_path: Path) -> None:
    """LightGBMBackend works with row and column subsampling."""
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_lightgbm_config(n_estimators=15)
    config["subsample"] = 0.7
    config["colsample_bytree"] = 0.7

    outcome = _invoke_lightgbm_train(backend, x, y, names, config, tmp_path)

    assert outcome["best_val_auc"] > 0.5


def test_lightgbm_backend_train_without_feature_names(tmp_path: Path) -> None:
    """LightGBMBackend generates feature names if not provided."""
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y = dataset["x"], dataset["y"]

    config = _make_lightgbm_config(n_estimators=10)

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
    # Verify loss decreased during training by comparing train vs untrained baseline
    # For binary classification, untrained loss is approx -log(0.5) = 0.693
    loss_initial = 0.693  # Untrained random baseline
    loss_final = outcome["val_metrics"]["loss"]
    assert loss_final < loss_initial, f"Loss should decrease: {loss_final} < {loss_initial}"


def test_lightgbm_backend_raises_on_no_positive_samples(tmp_path: Path) -> None:
    """LightGBMBackend raises if training set has no positive samples."""
    backend = create_lightgbm_backend()

    # Create dataset with no positives
    x = np.random.default_rng(42).standard_normal((100, 8)).astype(np.float64)
    y = np.zeros(100, dtype=np.int64)  # All negative
    names = [f"f{i}" for i in range(8)]

    config = _make_lightgbm_config(n_estimators=5)

    with pytest.raises(ValueError, match="no positive samples"):
        _invoke_lightgbm_train(backend, x, y, names, config, tmp_path)


def test_lightgbm_backend_with_progress_callback(tmp_path: Path) -> None:
    """LightGBMBackend calls progress callback with training metrics."""
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_lightgbm_config(n_estimators=10)

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
    assert progress_reports[0]["total_rounds"] == 10
    assert 0.0 <= progress_reports[0]["train_auc"] <= 1.0
    # Verify val_loss is present and reasonable
    val_loss = progress_reports[0]["val_loss"]
    if val_loss is None:
        raise AssertionError("val_loss must not be None in progress report")
    # Model should learn (val AUC beats random)
    assert outcome["best_val_auc"] > 0.5
    # Verify loss decreased during training by comparing vs untrained baseline
    # For binary classification, untrained loss is approx -log(0.5) = 0.693
    loss_initial = 0.693  # Untrained random baseline
    loss_final = val_loss
    assert loss_final < loss_initial, f"Loss should decrease: {loss_final} < {loss_initial}"


def test_lightgbm_capabilities() -> None:
    """LIGHTGBM_CAPABILITIES has expected structure."""
    assert LIGHTGBM_CAPABILITIES["supports_train"] is True
    assert LIGHTGBM_CAPABILITIES["supports_gpu"] is True
    assert LIGHTGBM_CAPABILITIES["supports_early_stopping"] is True
    assert LIGHTGBM_CAPABILITIES["supports_feature_importance"] is True
    assert LIGHTGBM_CAPABILITIES["model_format"] == "txt"


def test_lightgbm_backend_name() -> None:
    """LightGBMBackend.backend_name returns 'lightgbm'."""
    backend = create_lightgbm_backend()
    assert backend.backend_name() == "lightgbm"


def test_lightgbm_backend_capabilities() -> None:
    """LightGBMBackend.capabilities returns LIGHTGBM_CAPABILITIES."""
    backend = create_lightgbm_backend()
    caps = backend.capabilities()
    assert caps == LIGHTGBM_CAPABILITIES


def test_lightgbm_backend_different_depths(tmp_path: Path) -> None:
    """LightGBMBackend works with various max_depth values."""
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    for max_depth in [2, 4, 6]:
        config = _make_lightgbm_config(n_estimators=10, max_depth=max_depth)
        outcome = _invoke_lightgbm_train(backend, x, y, names, config, tmp_path)
        assert outcome["best_val_auc"] > 0.5, f"Failed for max_depth={max_depth}"


# =============================================================================
# Device Resolution Tests
# =============================================================================


def test_resolve_device_auto_returns_cpu() -> None:
    """_resolve_device returns 'cpu' for 'auto' device."""
    result = _resolve_device("auto")
    assert result == "cpu"


def test_resolve_device_cpu_returns_cpu() -> None:
    """_resolve_device returns 'cpu' for 'cpu' device."""
    result = _resolve_device("cpu")
    assert result == "cpu"


def test_resolve_device_cuda_on_windows_returns_gpu() -> None:
    """_resolve_device returns 'gpu' for 'cuda' on Windows platform."""
    result = _resolve_device("cuda", platform="win32")
    assert result == "gpu"


def test_resolve_device_cuda_on_linux_returns_cuda() -> None:
    """_resolve_device returns 'cuda' for 'cuda' on Linux platform."""
    result = _resolve_device("cuda", platform="linux")
    assert result == "cuda"


def test_resolve_device_cuda_on_darwin_returns_cuda() -> None:
    """_resolve_device returns 'cuda' for 'cuda' on macOS platform."""
    result = _resolve_device("cuda", platform="darwin")
    assert result == "cuda"


def test_resolve_device_unknown_returns_cpu() -> None:
    """_resolve_device returns 'cpu' for unknown device string."""
    result = _resolve_device("unknown")
    assert result == "cpu"


def test_lightgbm_prepared_exposes_the_native_booster(tmp_path: Path) -> None:
    """The prepared model surrenders the Booster SHAP needs.

    Booster.predict returns only P(class=1), so the wrapper exists to satisfy
    PredictorProtocol. SHAP introspects the native object and rejected that
    wrapper, which made lightgbm x shap_tree fail while being advertised as
    compatible.
    """
    backend = create_lightgbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_lightgbm_config(n_estimators=5)
    outcome = _invoke_lightgbm_train(backend, x, y, names, config, tmp_path)
    loaded = backend.load(path=outcome["model_path"])

    native = try_extract_native_tree_model(loaded)

    # Names the concrete type: a None here would read as "NoneType" and
    # still fail, so a separate not-None assertion adds nothing.
    assert type(native).__name__ == "Booster"
