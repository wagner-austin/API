"""Tests for covenant_ml trainer module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml import save_model, train_model
from covenant_ml.testing import make_train_config, set_cuda_hook
from covenant_ml.trainer import (
    DataSplits,
    _resolve_device,
    _XGBCoreProto,
    _XGBModuleProto,
    stratified_split,
    train_model_with_validation,
)
from covenant_ml.types import (
    TrainProgress,
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


def test_make_train_config_includes_scale_weight() -> None:
    """Helper adds optional scale_pos_weight when provided."""
    config = make_train_config(scale_pos_weight=2.5)
    assert config["scale_pos_weight"] == 2.5


class _FakeCore:
    def __init__(self, available: bool) -> None:
        self._available = available

    def build_info(self) -> dict[str, bool]:
        return {"USE_CUDA": self._available}


class _FakeBooster:
    def save_model(self, fname: str) -> None:
        _ = fname


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


def test_train_model_returns_fitted_model() -> None:
    """train_model returns a model that can predict."""
    x_features, y_labels = _make_training_data()
    config = make_train_config(reg_alpha=1.0, reg_lambda=5.0)

    model = train_model(x_features, y_labels, config)
    proba = model.predict_proba(x_features)
    assert proba.shape == (20, 2)


def test_train_model_sets_cpu_parallel_params() -> None:
    """XGBoost model uses multi-core histogram training."""
    x_features, y_labels = _make_training_data()
    config = make_train_config(reg_alpha=1.0, reg_lambda=5.0)

    model = train_model(x_features, y_labels, config)
    params = model.get_xgb_params()

    expected_jobs = max(1, int(os.cpu_count() or 1))
    assert params["n_jobs"] == expected_jobs
    assert params["tree_method"] == "hist"
    assert params["device"] == "cpu"


def test_resolve_device_prefers_cuda_when_supported() -> None:
    """_resolve_device chooses cuda when xgboost reports support."""
    fake_module: _XGBModuleProto = _FakeXGBModule(True)
    resolved = _resolve_device("auto", fake_module)
    assert resolved == "cuda"


def test_resolve_device_rejects_cuda_when_unsupported() -> None:
    """_resolve_device raises if cuda requested without support."""
    fake_module: _XGBModuleProto = _FakeXGBModule(False)
    with pytest.raises(RuntimeError, match="CUDA requested"):
        _resolve_device("cuda", fake_module)


def test_cuda_hook_forces_cpu_when_disabled() -> None:
    """Hook path executes and can force CPU resolution."""
    fake_module: _XGBModuleProto = _FakeXGBModule(True)
    set_cuda_hook(lambda: False)
    try:
        resolved = _resolve_device("auto", fake_module)
        assert resolved == "cpu"
    finally:
        set_cuda_hook(None)


def test_cuda_hook_allows_cuda_request_when_supported() -> None:
    """Hook path allows explicit cuda when supported."""
    fake_module: _XGBModuleProto = _FakeXGBModule(True)
    set_cuda_hook(lambda: True)
    try:
        resolved = _resolve_device("cuda", fake_module)
        assert resolved == "cuda"
    finally:
        set_cuda_hook(None)


def test_train_model_produces_valid_probabilities() -> None:
    """Predicted probabilities are in valid range."""
    x_features, y_labels = _make_training_data()
    config = make_train_config(reg_alpha=1.0, reg_lambda=5.0)

    model = train_model(x_features, y_labels, config)
    proba = model.predict_proba(x_features)

    for i in range(proba.shape[0]):
        for j in range(proba.shape[1]):
            p = float(proba[i, j])
            assert 0.0 <= p <= 1.0


def test_train_model_deterministic_with_same_seed() -> None:
    """Training with same random_state produces identical models."""
    x_features, y_labels = _make_training_data()
    config = make_train_config(random_state=123, reg_alpha=1.0, reg_lambda=5.0)

    model1 = train_model(x_features, y_labels, config)
    model2 = train_model(x_features, y_labels, config)

    proba1 = model1.predict_proba(x_features)
    proba2 = model2.predict_proba(x_features)

    for i in range(proba1.shape[0]):
        assert float(proba1[i, 1]) == float(proba2[i, 1])


def test_save_model_creates_file() -> None:
    """save_model creates a file at the specified path."""
    x_features, y_labels = _make_training_data()
    config = make_train_config(reg_alpha=1.0, reg_lambda=5.0)

    model = train_model(x_features, y_labels, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir) / "model.json")
        save_model(model, model_path)
        assert Path(model_path).exists()
        assert Path(model_path).stat().st_size > 0


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


def test_stratified_split_creates_correct_sizes() -> None:
    """stratified_split creates splits with correct proportions."""
    x_features, y_labels = _make_larger_data(100)

    splits = stratified_split(
        x_features,
        y_labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # Due to stratified per-class splitting, exact sizes may vary slightly
    assert 68 <= splits.n_train <= 72
    assert 13 <= splits.n_val <= 17
    assert 13 <= splits.n_test <= 17
    assert splits.n_total == 100


def test_stratified_split_maintains_class_proportions() -> None:
    """Stratified split maintains class balance in each split."""
    x_features, y_labels = _make_larger_data(100)

    splits = stratified_split(
        x_features,
        y_labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # Original ratio is 50% positive
    original_ratio = float(np.sum(y_labels)) / len(y_labels)

    # Each split should have similar ratio (within tolerance)
    train_ratio = float(np.sum(splits.y_train)) / len(splits.y_train)
    val_ratio = float(np.sum(splits.y_val)) / len(splits.y_val)
    test_ratio = float(np.sum(splits.y_test)) / len(splits.y_test)

    assert abs(train_ratio - original_ratio) < 0.1
    assert abs(val_ratio - original_ratio) < 0.15
    assert abs(test_ratio - original_ratio) < 0.15


def test_stratified_split_raises_on_invalid_ratios() -> None:
    """stratified_split raises ValueError if ratios don't sum to 1.0."""
    x_features, y_labels = _make_larger_data(100)

    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        stratified_split(
            x_features,
            y_labels,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.2,  # Sum = 1.1
            random_state=42,
        )


def test_stratified_split_deterministic() -> None:
    """Same random_state produces same splits."""
    x_features, y_labels = _make_larger_data(100)

    splits1 = stratified_split(x_features, y_labels, 0.7, 0.15, 0.15, random_state=123)
    splits2 = stratified_split(x_features, y_labels, 0.7, 0.15, 0.15, random_state=123)

    assert np.array_equal(splits1.y_train, splits2.y_train)
    assert np.array_equal(splits1.y_val, splits2.y_val)
    assert np.array_equal(splits1.y_test, splits2.y_test)


def test_data_splits_properties() -> None:
    """DataSplits has correct property values."""
    x_train = np.zeros((70, 8), dtype=np.float64)
    y_train = np.zeros(70, dtype=np.int64)
    x_val = np.zeros((15, 8), dtype=np.float64)
    y_val = np.zeros(15, dtype=np.int64)
    x_test = np.zeros((15, 8), dtype=np.float64)
    y_test = np.zeros(15, dtype=np.int64)

    splits = DataSplits(x_train, y_train, x_val, y_val, x_test, y_test)

    assert splits.n_train == 70
    assert splits.n_val == 15
    assert splits.n_test == 15
    assert splits.n_total == 100


def test_train_model_with_validation_returns_outcome() -> None:
    """train_model_with_validation returns TrainOutcome with all fields."""
    x_features, y_labels = _make_larger_data(100)
    config = make_train_config(
        n_estimators=5,
        early_stopping_rounds=3,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = train_model_with_validation(
            x_features, y_labels, config, Path(tmpdir), feature_names=feature_names
        )

        assert len(outcome["model_id"]) == 36  # UUID format
        assert Path(outcome["model_path"]).exists()
        assert outcome["samples_total"] == 100
        # Sizes may vary slightly due to stratified splitting
        assert 68 <= outcome["samples_train"] <= 72
        assert 13 <= outcome["samples_val"] <= 17
        assert 13 <= outcome["samples_test"] <= 17
        assert "loss" in outcome["train_metrics"]
        assert "loss" in outcome["val_metrics"]
        assert "loss" in outcome["test_metrics"]


def test_train_model_with_validation_progress_callback() -> None:
    """Progress callback receives TrainProgress updates."""
    x_features, y_labels = _make_larger_data(100)
    config = make_train_config(
        n_estimators=5,
        early_stopping_rounds=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    progress_updates: list[TrainProgress] = []

    def callback(progress: TrainProgress) -> None:
        progress_updates.append(progress)

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        train_model_with_validation(
            x_features,
            y_labels,
            config,
            Path(tmpdir),
            feature_names=feature_names,
            progress_callback=callback,
        )

        # Should have received progress updates
        assert len(progress_updates) == 5  # n_estimators = 5

        # Check first update (TrainProgress is a TypedDict - use dict access)
        first = progress_updates[0]
        assert first["round"] == 1
        assert first["total_rounds"] == 5
        assert 0.0 <= first["train_auc"] <= 1.0
        first_val_auc = first["val_auc"]
        assert first_val_auc is None or 0.0 <= first_val_auc <= 1.0


def test_train_model_with_validation_early_stopping() -> None:
    """Training can stop early when validation doesn't improve."""
    # Create data where model will overfit quickly
    x_features, y_labels = _make_larger_data(60)
    config = make_train_config(
        learning_rate=0.5,
        max_depth=6,
        n_estimators=50,
        subsample=1.0,
        colsample_bytree=1.0,
        early_stopping_rounds=3,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = train_model_with_validation(
            x_features, y_labels, config, Path(tmpdir), feature_names=feature_names
        )

        # May or may not early stop depending on data
        # Just verify the fields are populated correctly
        assert outcome["total_rounds"] <= 50
        assert outcome["best_round"] >= 1
        assert 0.0 <= outcome["best_val_auc"] <= 1.0


def test_train_model_with_validation_zero_estimators() -> None:
    """train_model_with_validation raises RuntimeError with n_estimators=0."""
    x_features, y_labels = _make_larger_data(100)
    config = make_train_config(
        n_estimators=0,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=3,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        pytest.raises(RuntimeError, match=r"n_estimators must be >= 1"),
    ):
        train_model_with_validation(
            x_features, y_labels, config, Path(tmpdir), feature_names=feature_names
        )


def test_train_model_with_validation_metrics_valid() -> None:
    """All metrics in outcome are within valid ranges."""
    x_features, y_labels = _make_larger_data(100)
    config = make_train_config(
        n_estimators=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = train_model_with_validation(
            x_features, y_labels, config, Path(tmpdir), feature_names=feature_names
        )

        # Check train metrics
        train_m = outcome["train_metrics"]
        assert train_m["loss"] >= 0.0
        assert 0.0 <= train_m["auc"] <= 1.0
        assert 0.0 <= train_m["accuracy"] <= 1.0
        assert 0.0 <= train_m["precision"] <= 1.0
        assert 0.0 <= train_m["recall"] <= 1.0
        assert 0.0 <= train_m["f1_score"] <= 1.0

        # Check val metrics
        val_m = outcome["val_metrics"]
        assert val_m["loss"] >= 0.0
        assert 0.0 <= val_m["auc"] <= 1.0
        assert 0.0 <= val_m["accuracy"] <= 1.0
        assert 0.0 <= val_m["precision"] <= 1.0
        assert 0.0 <= val_m["recall"] <= 1.0
        assert 0.0 <= val_m["f1_score"] <= 1.0

        # Check test metrics
        test_m = outcome["test_metrics"]
        assert test_m["loss"] >= 0.0
        assert 0.0 <= test_m["auc"] <= 1.0
        assert 0.0 <= test_m["accuracy"] <= 1.0
        assert 0.0 <= test_m["precision"] <= 1.0
        assert 0.0 <= test_m["recall"] <= 1.0
        assert 0.0 <= test_m["f1_score"] <= 1.0


def test_train_model_with_validation_wrong_feature_names_length() -> None:
    """train_model_with_validation raises ValueError with wrong feature_names length."""
    from covenant_ml.trainer import extract_feature_importances

    x_features, y_labels = _make_larger_data(100)
    config = make_train_config(
        n_estimators=5,
        early_stopping_rounds=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    # Train a model first
    model = train_model(x_features, y_labels, config)

    # Try to extract importances with wrong length feature names
    wrong_names = ["feat_0", "feat_1"]  # Only 2 names, model has 8 features
    with pytest.raises(ValueError, match=r"feature_names length.*must match model features"):
        extract_feature_importances(model, wrong_names)
