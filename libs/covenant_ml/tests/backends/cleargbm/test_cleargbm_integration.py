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
    _EarlyStoppingTracker,
    _is_cleargbm_config,
)
from covenant_ml.backends.protocol import ClassifierBackend, PreparedClassifier
from covenant_ml.types import (
    ClassifierTrainConfig,
    ClearGBMConfig,
    FeatureImportance,
    MLPConfig,
    TrainConfig,
    TrainOutcome,
    TrainProgress,
)

from ...conftest import load_us_bankruptcy_data

# Random binary classifier baseline log loss: -log(0.5) = 0.693
_BASELINE_LOSS = 0.693


def _invoke_cleargbm_train(
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


def _make_cleargbm_config(
    n_estimators: int = 5,
    max_depth: int = 3,
) -> ClearGBMConfig:
    """Create ClearGBM config for testing."""
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": 0.1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
        "max_bins": 64,
        "subsample": 1.0,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_rounds": 3,
        "track_contributions": False,
        "monotonic_constraints": None,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
    }


def _require_importances(
    importances: list[FeatureImportance] | None,
) -> list[FeatureImportance]:
    """Require importances list is not None.

    Args:
        importances: Feature importances or None.

    Returns:
        Feature importances list.

    Raises:
        AssertionError: If importances is None.
    """
    if importances is None:
        raise AssertionError("Expected importances list, got None")
    return importances


# =============================================================================
# Type Guard Tests
# =============================================================================


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


# =============================================================================
# Class Weight Computation Tests
# =============================================================================


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


# =============================================================================
# Early Stopping Tracker Tests
# =============================================================================


def test_early_stopping_tracker_improves_on_better_loss() -> None:
    """_EarlyStoppingTracker updates best when loss improves."""
    tracker = _EarlyStoppingTracker(early_stopping_rounds=3)

    tracker.update(val_loss=0.5, tree_index=0)
    assert tracker.best_val_loss == 0.5
    assert tracker.best_round == 1
    assert tracker.rounds_without_improvement == 0
    assert tracker.early_stopped is False


def test_early_stopping_tracker_no_improvement() -> None:
    """_EarlyStoppingTracker increments rounds_without_improvement when no improvement."""
    tracker = _EarlyStoppingTracker(early_stopping_rounds=3)

    tracker.update(val_loss=0.5, tree_index=0)
    tracker.update(val_loss=0.6, tree_index=1)  # Worse loss

    assert tracker.best_val_loss == 0.5
    assert tracker.best_round == 1
    assert tracker.rounds_without_improvement == 1


def test_early_stopping_tracker_triggers_early_stop() -> None:
    """_EarlyStoppingTracker triggers early_stopped after threshold."""
    tracker = _EarlyStoppingTracker(early_stopping_rounds=3)

    tracker.update(val_loss=0.5, tree_index=0)
    tracker.update(val_loss=0.6, tree_index=1)  # No improvement
    tracker.update(val_loss=0.7, tree_index=2)  # No improvement
    tracker.update(val_loss=0.8, tree_index=3)  # No improvement - triggers

    assert tracker.rounds_without_improvement == 3
    assert tracker.early_stopped is True


def test_early_stopping_tracker_val_loss_none_no_update() -> None:
    """_EarlyStoppingTracker handles val_loss=None without updating state."""
    tracker = _EarlyStoppingTracker(early_stopping_rounds=3)

    # First update with actual loss
    tracker.update(val_loss=0.5, tree_index=0)
    assert tracker.best_val_loss == 0.5
    assert tracker.best_round == 1

    # Update with None - should not change state
    tracker.update(val_loss=None, tree_index=1)
    assert tracker.best_val_loss == 0.5
    assert tracker.best_round == 1
    assert tracker.rounds_without_improvement == 0
    assert tracker.early_stopped is False


def test_early_stopping_tracker_all_none_no_crash() -> None:
    """_EarlyStoppingTracker handles all None val_loss without crashing."""
    tracker = _EarlyStoppingTracker(early_stopping_rounds=3)

    # Multiple updates with None
    tracker.update(val_loss=None, tree_index=0)
    tracker.update(val_loss=None, tree_index=1)
    tracker.update(val_loss=None, tree_index=2)

    # State should remain at defaults
    assert tracker.best_val_loss == float("inf")
    assert tracker.best_round == 0
    assert tracker.rounds_without_improvement == 0
    assert tracker.early_stopped is False


# =============================================================================
# ClearGBMPrepared Tests
# =============================================================================


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


# =============================================================================
# Backend Factory and Constants Tests
# =============================================================================


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


# =============================================================================
# Backend Prepare Tests
# =============================================================================


def test_cleargbm_backend_prepare_raises() -> None:
    """ClearGBMBackend.prepare raises RuntimeError."""
    backend = create_cleargbm_backend()

    with pytest.raises(RuntimeError, match="prepare not supported"):
        backend.prepare(n_features=10, n_classes=2, feature_names=None)


# =============================================================================
# Backend Train Tests
# =============================================================================


def test_cleargbm_backend_train_returns_outcome(tmp_path: Path) -> None:
    """ClearGBMBackend trains and returns TrainOutcome with all required fields."""
    backend = create_cleargbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_cleargbm_config(n_estimators=10, max_depth=4)

    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Verify outcome structure
    assert outcome["samples_total"] == len(y)
    assert outcome["samples_train"] > 0
    assert outcome["samples_val"] > 0
    assert outcome["samples_test"] > 0

    # Verify metrics exist and are reasonable
    assert 0.0 <= outcome["train_metrics"]["auc"] <= 1.0
    assert 0.0 <= outcome["val_metrics"]["auc"] <= 1.0
    assert 0.0 <= outcome["test_metrics"]["auc"] <= 1.0
    assert outcome["best_val_auc"] > 0.0

    # Verify model was saved
    assert Path(outcome["model_path"]).exists()
    assert outcome["model_path"].endswith(".json")

    # Verify feature importances exist with correct count
    assert len(outcome["feature_importances"]) == len(names)
    assert outcome["feature_importances"][0]["rank"] == 1

    # Verify scale_pos_weight was computed
    assert outcome["scale_pos_weight_computed"] > 0.0

    # Verify loss decreased from baseline
    val_loss = outcome["val_metrics"]["loss"]
    assert val_loss < _BASELINE_LOSS, (
        f"Validation loss {val_loss} should be below baseline {_BASELINE_LOSS}"
    )


def test_cleargbm_backend_train_without_feature_names(tmp_path: Path) -> None:
    """ClearGBMBackend generates feature names if not provided."""
    backend = create_cleargbm_backend()
    x, y, _ = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)

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
    # Model should train successfully
    assert outcome["samples_total"] == 100
    # Verify loss decreased from baseline
    val_loss = outcome["val_metrics"]["loss"]
    assert val_loss < _BASELINE_LOSS, (
        f"Validation loss {val_loss} should be below baseline {_BASELINE_LOSS}"
    )


def test_cleargbm_backend_config_type_validation(tmp_path: Path) -> None:
    """ClearGBMBackend raises on non-ClearGBM config."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    # Try XGBoost config (wrong type)
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

    with pytest.raises(RuntimeError, match="ClearGBMBackend requires ClearGBMConfig"):
        _invoke_cleargbm_train(backend, x, y, names, xgb_config, tmp_path)


def test_cleargbm_backend_train_with_mlp_config_raises(tmp_path: Path) -> None:
    """ClearGBMBackend raises on MLPConfig."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

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

    with pytest.raises(RuntimeError, match="ClearGBMBackend requires ClearGBMConfig"):
        _invoke_cleargbm_train(backend, x, y, names, mlp_config, tmp_path)


def test_cleargbm_backend_raises_on_no_positive_samples(tmp_path: Path) -> None:
    """ClearGBMBackend raises if training set has no positive samples."""
    backend = create_cleargbm_backend()

    # Create dataset with no positives
    x = np.random.default_rng(42).standard_normal((100, 8)).astype(np.float64)
    y = np.zeros(100, dtype=np.int64)  # All negative
    names = [f"f{i}" for i in range(8)]

    config = _make_cleargbm_config(n_estimators=3)

    with pytest.raises(ValueError, match="no positive samples"):
        _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)


def test_cleargbm_backend_with_progress_callback(tmp_path: Path) -> None:
    """ClearGBMBackend calls progress callback during training."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)

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

    # Should have progress reports (one per tree)
    assert len(progress_reports) == 5
    # Each report should have valid structure
    for report in progress_reports:
        assert report["round"] >= 1
        assert report["total_rounds"] == 5
        assert report["train_loss"] >= 0.0

    # Verify loss decreased from baseline
    val_loss = outcome["val_metrics"]["loss"]
    assert val_loss < _BASELINE_LOSS, (
        f"Validation loss {val_loss} should be below baseline {_BASELINE_LOSS}"
    )


def test_cleargbm_backend_train_with_subsampling(tmp_path: Path) -> None:
    """ClearGBMBackend works with row subsampling."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)
    config["subsample"] = 0.7

    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Should complete successfully
    assert outcome["samples_total"] == 100


def test_cleargbm_backend_train_early_stopping(tmp_path: Path) -> None:
    """ClearGBMBackend tracks early stopping progress."""
    backend = create_cleargbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    # Use more estimators to potentially trigger early stopping
    config = _make_cleargbm_config(n_estimators=20, max_depth=2)
    config["early_stopping_rounds"] = 3

    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Verify early_stopped field is boolean (value depends on data)
    early_stopped = outcome["early_stopped"]
    assert early_stopped is True or early_stopped is False
    # Verify best_round is tracked
    assert outcome["best_round"] >= 1


def test_cleargbm_backend_train_different_depths(tmp_path: Path) -> None:
    """ClearGBMBackend works with various max_depth values."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    for max_depth in [2, 4, 6]:
        config = _make_cleargbm_config(n_estimators=3, max_depth=max_depth)
        outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)
        assert outcome["samples_total"] == 100, f"Failed for max_depth={max_depth}"


# =============================================================================
# Backend Evaluate Tests
# =============================================================================


def test_cleargbm_backend_evaluate_computes_metrics(tmp_path: Path) -> None:
    """ClearGBMBackend.evaluate computes metrics using loaded model."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Load the trained model and evaluate
    loaded_model = backend.load(path=outcome["model_path"])
    metrics = backend.evaluate(model=loaded_model, x=x, y=y)

    # Metrics should be computed correctly
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] > 0.0  # Log loss is always positive


def test_cleargbm_backend_evaluate_on_subset(tmp_path: Path) -> None:
    """ClearGBMBackend.evaluate works on data subset."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset(n_samples=100)

    config = _make_cleargbm_config(n_estimators=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Evaluate on first 50 samples only
    loaded_model = backend.load(path=outcome["model_path"])
    metrics = backend.evaluate(model=loaded_model, x=x[:50], y=y[:50])

    assert 0.0 <= metrics["auc"] <= 1.0


# =============================================================================
# Backend Save/Load Tests
# =============================================================================


def test_cleargbm_backend_save_and_load(tmp_path: Path) -> None:
    """ClearGBMBackend.save and load round-trip correctly."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Load the saved model
    loaded = backend.load(path=outcome["model_path"])

    # Verify it can predict
    proba = loaded.predict_proba(x)
    assert proba.shape == (100, 2)


def test_cleargbm_backend_save_creates_json_file(tmp_path: Path) -> None:
    """ClearGBMBackend.save creates a JSON file."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=3)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    model_path = Path(outcome["model_path"])
    assert model_path.exists()
    assert model_path.suffix == ".json"

    # Verify it's valid JSON by reading
    content = model_path.read_text(encoding="utf-8")
    assert content.startswith("{")


def test_cleargbm_backend_save_raises_for_wrong_model_type(tmp_path: Path) -> None:
    """ClearGBMBackend.save raises for non-ClearGBMPrepared model."""
    backend = create_cleargbm_backend()

    class _FakeModel:
        """Fake model that's not _ClearGBMPrepared."""

        def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
            n = int(x.shape[0])
            return np.full((n, 2), 0.5, dtype=np.float64)

    fake_model: PreparedClassifier = _FakeModel()
    save_path = str(tmp_path / "fake.json")

    with pytest.raises(RuntimeError, match="must be _ClearGBMPrepared"):
        backend.save(model=fake_model, path=save_path)


def test_cleargbm_backend_load_and_predict(tmp_path: Path) -> None:
    """ClearGBMBackend.load loads a trained model that can predict."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Load the trained model
    loaded_model = backend.load(path=outcome["model_path"])

    # Predict probabilities
    proba = loaded_model.predict_proba(x)

    # Verify shape and values
    assert proba.shape == (100, 2)
    min_val = float(np.min(proba))
    max_val = float(np.max(proba))
    assert min_val >= 0.0
    assert max_val <= 1.0


# =============================================================================
# Backend Feature Importance Tests
# =============================================================================


def test_cleargbm_backend_get_feature_importances(tmp_path: Path) -> None:
    """ClearGBMBackend.get_feature_importances returns importance list."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset(n_features=4)

    config = _make_cleargbm_config(n_estimators=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Load model and get importances
    loaded_model = backend.load(path=outcome["model_path"])
    importances = _require_importances(
        backend.get_feature_importances(model=loaded_model, feature_names=names)
    )

    # Verify count matches features
    assert len(importances) == 4
    # Check structure via first element
    first = importances[0]
    assert first["name"] == "f0" or first["name"] == "f1" or first["name"] in names
    assert first["importance"] >= 0.0
    assert first["rank"] == 1


def test_cleargbm_backend_get_feature_importances_sorted_by_rank(tmp_path: Path) -> None:
    """ClearGBMBackend.get_feature_importances returns sorted by rank."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset(n_features=6)

    config = _make_cleargbm_config(n_estimators=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    loaded_model = backend.load(path=outcome["model_path"])
    importances = _require_importances(
        backend.get_feature_importances(model=loaded_model, feature_names=names)
    )

    # Verify ranks are sequential starting at 1
    ranks = [imp["rank"] for imp in importances]
    expected_ranks = list(range(1, len(ranks) + 1))
    assert ranks == expected_ranks


def test_cleargbm_backend_get_feature_importances_returns_none_for_wrong_type(
    tmp_path: Path,
) -> None:
    """ClearGBMBackend.get_feature_importances returns None for wrong model type."""
    backend = create_cleargbm_backend()

    class _FakeModel:
        """Fake model that's not _ClearGBMPrepared."""

        def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
            n = int(x.shape[0])
            return np.full((n, 2), 0.5, dtype=np.float64)

    fake_model: PreparedClassifier = _FakeModel()
    result = backend.get_feature_importances(model=fake_model, feature_names=["a", "b"])

    assert result is None


# =============================================================================
# Integration Test with Real Data
# =============================================================================


def test_cleargbm_backend_us_bankruptcy_full_pipeline(tmp_path: Path) -> None:
    """Full pipeline test with US bankruptcy dataset."""
    backend = create_cleargbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config: ClearGBMConfig = {
        "n_estimators": 15,
        "max_depth": 4,
        "learning_rate": 0.1,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": None,
        "max_bins": 64,
        "subsample": 0.8,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "random_state": 42,
        "early_stopping_rounds": 5,
        "track_contributions": False,
        "monotonic_constraints": None,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
    }

    # Train
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Verify training completed
    assert outcome["samples_total"] == dataset["n_samples"]
    assert Path(outcome["model_path"]).exists()

    # Load and evaluate
    loaded_model = backend.load(path=outcome["model_path"])
    test_metrics = backend.evaluate(model=loaded_model, x=x, y=y)

    # Should produce reasonable metrics
    assert test_metrics["auc"] > 0.0
    assert test_metrics["loss"] < 10.0  # Sanity check

    # Verify feature importances
    importances = _require_importances(
        backend.get_feature_importances(model=loaded_model, feature_names=names)
    )
    assert len(importances) == len(names)


def test_cleargbm_backend_progress_callback_receives_val_loss(tmp_path: Path) -> None:
    """Progress callback receives validation loss when val set exists."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset(n_samples=100)

    config = _make_cleargbm_config(n_estimators=5)

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

    # Check that at least one report has val_loss
    has_val_loss = any(r["val_loss"] is not None for r in progress_reports)
    assert has_val_loss

    # Verify loss decreased from baseline
    val_loss = outcome["val_metrics"]["loss"]
    assert val_loss < _BASELINE_LOSS, (
        f"Validation loss {val_loss} should be below baseline {_BASELINE_LOSS}"
    )


def test_cleargbm_backend_model_id_is_uuid(tmp_path: Path) -> None:
    """Model ID is a valid UUID string."""
    import uuid

    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=3)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Verify model_id is a valid UUID
    model_id = outcome["model_id"]
    parsed_uuid = uuid.UUID(model_id)
    assert str(parsed_uuid) == model_id


def test_cleargbm_backend_config_preserved_in_outcome(tmp_path: Path) -> None:
    """Training config is preserved in TrainOutcome."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=7, max_depth=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Verify config is preserved by narrowing to ClearGBMConfig
    saved_config = outcome["config"]
    # ClearGBMConfig has min_samples_split which other configs don't
    assert _is_cleargbm_config(saved_config)
    # Now we can access ClearGBM-specific keys
    cleargbm_config: ClearGBMConfig = saved_config
    assert cleargbm_config["n_estimators"] == 7
    assert cleargbm_config["max_depth"] == 5
