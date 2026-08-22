"""ClearGBM backend integration tests with actual training.

Tests the full training loop, prediction, save/load, and error paths.
Uses real US bankruptcy data for integration tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_float,
)

from covenant_ml.backends.cleargbm import (
    create_cleargbm_backend,
)
from covenant_ml.backends.cleargbm.config_resolution import (
    _resolve_max_features,
    _resolve_monotonic_constraints,
)
from covenant_ml.types import (
    MLPConfig,
    TrainConfig,
    TrainProgress,
)
from tests.backends.cleargbm._cleargbm_fixtures import (
    _BASELINE_LOSS,
    _invoke_cleargbm_train,
    _make_cleargbm_config,
    _make_synthetic_dataset,
)

from ...conftest import load_us_bankruptcy_data


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


def test_cleargbm_backend_progress_callback_is_noop_on_native_path(tmp_path: Path) -> None:
    """Passing a progress callback is accepted but never invoked on the native path.

    The Rust training loop is a single native call — it does not surface per-tree
    progress to Python. The wrapper's ``train()`` documents this and skips the
    callback rather than emitting synthetic ``TrainProgress`` events. This test
    guards the documented behavior: training must succeed to convergence even
    when a callback is present, and the callback must never fire.
    """
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

    # Documented no-op on the native path.
    assert progress_reports == []

    # Training still completes and beats the random baseline.
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


def test_cleargbm_backend_honors_reg_lambda(tmp_path: Path) -> None:
    """reg_lambda from the config reaches the trainer and changes the model.

    The backend hardcoded reg_lambda=0.0 until 2026-08-22, so cv_external's
    stated reg_lambda=1.0 trained unregularized. Heavy L2 must shrink leaf
    values, so predictions from the two settings cannot be identical.
    """
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    outputs: dict[float, str] = {}
    for reg_lambda in (0.0, 100.0):
        config = _make_cleargbm_config(n_estimators=5)
        config["reg_lambda"] = reg_lambda
        outcome = _invoke_cleargbm_train(
            backend, x, y, names, config, tmp_path / f"reg_{reg_lambda}"
        )
        outputs[reg_lambda] = Path(outcome["model_path"]).read_text(encoding="utf-8")

    assert outputs[0.0] != outputs[100.0]


def test_cleargbm_backend_rejects_unknown_constraint_feature(tmp_path: Path) -> None:
    """A monotonic constraint naming a nonexistent feature is an error,
    not a silently dropped constraint."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=3)
    config["monotonic_constraints"] = {"no_such_feature": 1}

    with pytest.raises(ValueError, match="monotonic_constraints name unknown features"):
        _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)


def test_resolve_monotonic_constraints_maps_names_to_columns() -> None:
    """Named constraints land on their columns; unnamed columns get 0."""
    resolved = _resolve_monotonic_constraints({"b": -1, "c": 1}, ("a", "b", "c"))
    assert resolved == (0, -1, 1)
    assert _resolve_monotonic_constraints(None, ("a", "b")) is None


def test_resolve_max_features_translates_each_form() -> None:
    """None passes through; ints pass through; fractions become counts."""
    assert _resolve_max_features(None, 10) is None
    assert _resolve_max_features(4, 10) == 4
    assert _resolve_max_features(0.5, 10) == 5
    assert _resolve_max_features(0.01, 10) == 1


def test_cleargbm_backend_applies_computed_class_weight(tmp_path: Path) -> None:
    """The auto-computed scale_pos_weight reaches training, not just the log.

    The backend computed and reported this weight while the core had no
    weighting mechanism at all — the saved model's own config is the proof
    the value now arrives: it must record the imbalance ratio of the
    training split, not 1.0.
    """
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    outcome = _invoke_cleargbm_train(
        backend, x, y, names, _make_cleargbm_config(n_estimators=3), tmp_path
    )

    model_json = narrow_json_to_dict(
        load_json_str(Path(outcome["model_path"]).read_text(encoding="utf-8"))
    )
    saved_config = narrow_json_to_dict(model_json["config"])
    saved_weight = narrow_json_to_float(saved_config["scale_pos_weight"])
    assert saved_weight == outcome["scale_pos_weight_computed"]
    assert saved_weight != 1.0


def test_cleargbm_backend_train_leaf_wise(tmp_path: Path) -> None:
    """ClearGBMBackend passes growth_strategy/num_leaves through to the core.

    A leaf-wise train must complete and produce a loadable model; a broken
    pass-through would surface as the Rust config validator rejecting the
    depth_wise-shaped pair.
    """
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)
    config["growth_strategy"] = "leaf_wise"
    config["num_leaves"] = 8

    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    assert outcome["samples_total"] == 100
    assert Path(outcome["model_path"]).exists()


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
