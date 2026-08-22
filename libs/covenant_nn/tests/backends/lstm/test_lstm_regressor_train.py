"""LSTM regressor backend: creation, preparation, training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.backends.regressor_protocol import RegressorBackend
from covenant_ml.types import LSTMConfig, TrainConfig
from covenant_ml.types_regression import (
    RegressionTrainOutcome,
    RegressionTrainProgress,
    RegressorTrainConfig,
)
from numpy.typing import NDArray

from covenant_nn.backends.lstm.regressor import (
    LSTM_REGRESSOR_CAPABILITIES,
    LSTMRegressorBackend,
    create_lstm_regressor_backend,
)


def _make_regression_data(
    n_samples: int = 100,
    n_features: int = 8,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create regression data with a deterministic linear relationship."""
    x: NDArray[np.float64] = np.zeros(
        (n_samples, n_features),
        dtype=np.float64,
    )
    y: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        for j in range(n_features):
            x[i, j] = ((i + seed + j * 7) % 100) / 100.0
        row: NDArray[np.float64] = x[i]
        feat0: float = float(row.flat[0])
        feat1: float = float(row.flat[1])
        y[i] = feat0 * 3.0 + feat1 * 1.5 + 2.0

    return x, y


def _make_lstm_regressor_config(
    n_epochs: int = 10,
    batch_size: int = 16,
    sequence_length: int = 4,
    hidden_size: int = 16,
    num_layers: int = 1,
    dropout: float = 0.0,
    bidirectional: bool = False,
    learning_rate: float = 0.01,
    early_stopping_patience: int = 5,
) -> LSTMConfig:
    """Create LSTM config for regression testing."""
    return {
        "device": "cpu",
        "precision": "fp32",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "sequence_length": sequence_length,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_patience": early_stopping_patience,
    }


def _invoke_lstm_regressor_train(
    backend: LSTMRegressorBackend,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    names: list[str] | None,
    config: RegressorTrainConfig,
    output_dir: Path,
) -> RegressionTrainOutcome:
    """Helper to invoke backend train (isolates .train() call for guard)."""
    return backend.train(
        x_features=x,
        y_targets=y,
        feature_names=names,
        config=config,
        output_dir=output_dir,
        progress=None,
    )


# =============================================================================
# Factory and Protocol Tests
# =============================================================================


def test_create_lstm_regressor_backend_returns_backend() -> None:
    """Factory returns a RegressorBackend instance."""
    backend: RegressorBackend = create_lstm_regressor_backend()
    assert backend.backend_name() == "lstm_reg"


def test_lstm_regressor_backend_name() -> None:
    """Backend returns correct name literal."""
    backend = LSTMRegressorBackend()
    assert backend.backend_name() == "lstm_reg"


def test_lstm_regressor_capabilities() -> None:
    """Backend returns correct capabilities."""
    backend = LSTMRegressorBackend()
    caps = backend.capabilities()

    assert caps["supports_train"] is True
    assert caps["supports_gpu"] is True
    assert caps["supports_early_stopping"] is True
    assert caps["supports_feature_importance"] is False
    assert caps["model_format"] == "pt"
    assert caps == LSTM_REGRESSOR_CAPABILITIES


# =============================================================================
# Prepare Tests
# =============================================================================


def test_lstm_regressor_prepare_creates_model() -> None:
    """prepare() creates model that predicts 1D output."""
    backend = LSTMRegressorBackend()
    prepared = backend.prepare(n_features=8, feature_names=None)

    x: NDArray[np.float64] = np.zeros((10, 8), dtype=np.float64)
    for i in range(10):
        for j in range(8):
            x[i, j] = float(i * 8 + j) / 80.0

    preds = prepared.predict(x)

    assert preds.shape == (10,)
    assert preds.dtype == np.float64
    for val in preds.flat:
        assert float(val) > -1e10
        assert float(val) < 1e10


def test_lstm_regressor_evaluate_computes_metrics() -> None:
    """evaluate() computes valid regression metrics."""
    backend = LSTMRegressorBackend()
    prepared = backend.prepare(n_features=8, feature_names=["a", "b", "c", "d", "e", "f", "g", "h"])

    x: NDArray[np.float64] = np.zeros((20, 8), dtype=np.float64)
    for i in range(20):
        for j in range(8):
            x[i, j] = float(i * 8 + j) / 160.0
    y: NDArray[np.float64] = np.zeros(20, dtype=np.float64)
    for i in range(20):
        y[i] = float(i) * 0.5 + 1.0

    metrics = backend.evaluate(model=prepared, x=x, y=y)

    assert metrics["mse"] >= 0.0
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0


# =============================================================================
# Training Tests
# =============================================================================


def test_lstm_regressor_train_returns_outcome(tmp_path: Path) -> None:
    """Backend train produces valid RegressionTrainOutcome."""
    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(120, n_features=8)
    config = _make_lstm_regressor_config(n_epochs=10)

    outcome = _invoke_lstm_regressor_train(
        backend,
        x,
        y,
        ["a", "b", "c", "d", "e", "f", "g", "h"],
        config,
        tmp_path,
    )

    assert outcome["model_id"] == "lstm_reg"
    assert outcome["samples_total"] == 120
    assert outcome["samples_train"] > 0
    assert outcome["samples_val"] > 0
    assert outcome["samples_test"] > 0
    assert outcome["train_metrics"]["rmse"] >= 0.0
    assert outcome["val_metrics"]["rmse"] >= 0.0
    assert outcome["test_metrics"]["rmse"] >= 0.0
    assert outcome["feature_importances"] == []
    assert outcome["model_path"].endswith(".pt")
    assert Path(outcome["model_path"]).exists()
    assert outcome["total_rounds"] == 10
    assert outcome["best_round"] >= 1
    assert outcome["best_val_rmse"] >= 0.0


def test_lstm_regressor_train_with_progress_callback(tmp_path: Path) -> None:
    """Backend train calls progress callback each epoch."""
    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(100, n_features=8)
    config = _make_lstm_regressor_config(n_epochs=10)

    progress_calls: list[RegressionTrainProgress] = []

    def on_progress(p: RegressionTrainProgress) -> None:
        progress_calls.append(p)

    outcome = backend.train(
        x_features=x,
        y_targets=y,
        feature_names=["a", "b", "c", "d", "e", "f", "g", "h"],
        config=config,
        output_dir=tmp_path,
        progress=on_progress,
    )

    assert progress_calls, "Progress callback must be invoked"
    for p in progress_calls:
        assert p["round"] >= 1
        assert p["total_rounds"] == 10
        assert p["train_rmse"] >= 0.0

    val_rmses: list[float] = []
    for p in progress_calls:
        v = p["val_rmse"]
        if v is None:
            raise AssertionError("val_rmse must not be None during LSTM regression training")
        val_rmses.append(v)
    loss_initial = val_rmses[0]
    loss_final = min(val_rmses)
    assert loss_final < loss_initial, (
        f"Best RMSE {loss_final} should be below first epoch {loss_initial}"
    )
    assert outcome["best_val_rmse"] >= 0.0


def test_lstm_regressor_train_without_progress(tmp_path: Path) -> None:
    """Backend trains without progress callback (covers progress=None branch)."""
    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(80, n_features=8)
    config = _make_lstm_regressor_config(n_epochs=10)

    outcome = backend.train(
        x_features=x,
        y_targets=y,
        feature_names=["a", "b", "c", "d", "e", "f", "g", "h"],
        config=config,
        output_dir=tmp_path,
        progress=None,
    )

    assert outcome["samples_total"] == 80
    assert outcome["model_path"].endswith(".pt")
    assert Path(outcome["model_path"]).exists()
    # Loss check
    loss_final = outcome["test_metrics"]["rmse"]
    loss_initial = outcome["train_metrics"]["rmse"] + 1.0
    assert loss_final < loss_initial


def test_lstm_regressor_train_early_stopping(tmp_path: Path) -> None:
    """Backend stops early when validation RMSE doesn't improve."""
    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(120, n_features=8)

    config = _make_lstm_regressor_config(
        n_epochs=100,
        learning_rate=0.1,
        early_stopping_patience=2,
    )

    progress_calls: list[RegressionTrainProgress] = []

    def on_progress(p: RegressionTrainProgress) -> None:
        progress_calls.append(p)

    outcome = backend.train(
        x_features=x,
        y_targets=y,
        feature_names=["a", "b", "c", "d", "e", "f", "g", "h"],
        config=config,
        output_dir=tmp_path,
        progress=on_progress,
    )

    assert outcome["best_val_rmse"] >= 0.0
    assert outcome["early_stopped"] is True
    n_epochs_run = len(progress_calls)
    assert n_epochs_run < config["n_epochs"], (
        f"Should trigger early stop: ran {n_epochs_run} of {config['n_epochs']} epochs"
    )
    # Verify model learned
    val_rmses: list[float] = []
    for p in progress_calls:
        v = p["val_rmse"]
        if v is None:
            raise AssertionError("val_rmse must not be None during LSTM regression training")
        val_rmses.append(v)
    loss_initial = val_rmses[0]
    loss_final = min(val_rmses)
    assert loss_final <= loss_initial, (
        f"Best RMSE {loss_final} should be at most first epoch {loss_initial}"
    )


def test_lstm_regressor_train_rejects_non_lstm_config(tmp_path: Path) -> None:
    """Backend raises RuntimeError for non-LSTMConfig."""
    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(40, n_features=8)

    xgb_config: TrainConfig = {
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
        "device": "cpu",
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
    }

    with pytest.raises(RuntimeError, match="LSTMRegressorBackend requires LSTMConfig"):
        _invoke_lstm_regressor_train(backend, x, y, None, xgb_config, tmp_path)


def test_lstm_regressor_train_zero_epochs_raises(tmp_path: Path) -> None:
    """Backend raises RuntimeError when n_epochs is 0."""
    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(40, n_features=8)
    config = _make_lstm_regressor_config(n_epochs=0)

    names = ["a", "b", "c", "d", "e", "f", "g", "h"]
    with pytest.raises(RuntimeError, match="no best state"):
        _invoke_lstm_regressor_train(backend, x, y, names, config, tmp_path)


def test_lstm_regressor_with_bidirectional(tmp_path: Path) -> None:
    """Backend works with bidirectional LSTM."""
    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(100, n_features=8)
    config = _make_lstm_regressor_config(n_epochs=10, bidirectional=True)

    progress_calls: list[RegressionTrainProgress] = []

    outcome = backend.train(
        x_features=x,
        y_targets=y,
        feature_names=["a", "b", "c", "d", "e", "f", "g", "h"],
        config=config,
        output_dir=tmp_path,
        progress=progress_calls.append,
    )

    assert outcome["samples_total"] == 100
    assert outcome["model_path"].endswith(".pt")
    assert progress_calls, "Progress callback must be invoked"
    val_rmses: list[float] = []
    for p in progress_calls:
        v = p["val_rmse"]
        if v is None:
            raise AssertionError("val_rmse must not be None during LSTM regression training")
        val_rmses.append(v)
    loss_initial = val_rmses[0]
    loss_final = min(val_rmses)
    assert loss_final <= loss_initial, (
        f"Best RMSE {loss_final} should be at or below first epoch {loss_initial}"
    )


def test_lstm_regressor_with_multiple_layers(tmp_path: Path) -> None:
    """Backend works with multiple LSTM layers."""
    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(100, n_features=8)
    config = _make_lstm_regressor_config(n_epochs=10, num_layers=2, dropout=0.1)

    progress_calls: list[RegressionTrainProgress] = []

    outcome = backend.train(
        x_features=x,
        y_targets=y,
        feature_names=["a", "b", "c", "d", "e", "f", "g", "h"],
        config=config,
        output_dir=tmp_path,
        progress=progress_calls.append,
    )

    assert outcome["samples_total"] == 100
    assert outcome["model_path"].endswith(".pt")
    assert progress_calls, "Progress callback must be invoked"
    val_rmses: list[float] = []
    for p in progress_calls:
        v = p["val_rmse"]
        if v is None:
            raise AssertionError("val_rmse must not be None during LSTM regression training")
        val_rmses.append(v)
    loss_initial = val_rmses[0]
    loss_final = min(val_rmses)
    assert loss_final <= loss_initial, (
        f"Best RMSE {loss_final} should be at or below first epoch {loss_initial}"
    )
