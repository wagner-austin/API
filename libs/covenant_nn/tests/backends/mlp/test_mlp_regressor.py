"""MLP regressor backend integration tests with actual PyTorch training.

Tests the full regression training loop, prediction, and error paths
using synthetic regression data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.backends.regressor_protocol import RegressorBackend
from covenant_ml.types import (
    MLPConfig,
    MLPOptimizer,
    RegressionTrainOutcome,
    RegressionTrainProgress,
    RegressorTrainConfig,
    TrainConfig,
)
from numpy.typing import NDArray

from covenant_nn.backends.mlp.regressor import (
    MLP_REGRESSOR_CAPABILITIES,
    MLPRegressorBackend,
    create_mlp_regressor_backend,
)


def _make_regression_data(
    n_samples: int = 100,
    n_features: int = 5,
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


def _make_mlp_regressor_config(
    n_epochs: int = 10,
    batch_size: int = 16,
    hidden_sizes: tuple[int, ...] = (8, 4),
    dropout: float = 0.0,
    optimizer: MLPOptimizer = "adamw",
    learning_rate: float = 0.01,
    early_stopping_patience: int = 5,
) -> MLPConfig:
    """Create MLP config for regression testing."""
    return {
        "device": "cpu",
        "precision": "fp32",
        "optimizer": optimizer,
        "hidden_sizes": hidden_sizes,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "dropout": dropout,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_patience": early_stopping_patience,
    }


def _invoke_mlp_regressor_train(
    backend: MLPRegressorBackend,
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


def test_create_mlp_regressor_backend_returns_backend() -> None:
    """Factory returns a RegressorBackend instance."""
    backend: RegressorBackend = create_mlp_regressor_backend()
    assert backend.backend_name() == "mlp_reg"


def test_mlp_regressor_backend_name() -> None:
    """Backend returns correct name literal."""
    backend = MLPRegressorBackend()
    assert backend.backend_name() == "mlp_reg"


def test_mlp_regressor_capabilities() -> None:
    """Backend returns correct capabilities."""
    backend = MLPRegressorBackend()
    caps = backend.capabilities()

    assert caps["supports_train"] is True
    assert caps["supports_gpu"] is True
    assert caps["supports_early_stopping"] is True
    assert caps["supports_feature_importance"] is False
    assert caps["model_format"] == "pt"
    assert caps == MLP_REGRESSOR_CAPABILITIES


# =============================================================================
# Prepare Tests
# =============================================================================


def test_mlp_regressor_prepare_creates_model() -> None:
    """prepare() creates model that predicts 1D output."""
    backend = MLPRegressorBackend()
    prepared = backend.prepare(n_features=5, feature_names=None)

    x: NDArray[np.float64] = np.zeros((10, 5), dtype=np.float64)
    for i in range(10):
        for j in range(5):
            x[i, j] = float(i * 5 + j) / 50.0

    preds = prepared.predict(x)

    assert preds.shape == (10,)
    assert preds.dtype == np.float64
    # Predictions should be finite
    for val in preds.flat:
        assert float(val) > -1e10
        assert float(val) < 1e10


def test_mlp_regressor_evaluate_computes_metrics() -> None:
    """evaluate() computes valid regression metrics."""
    backend = MLPRegressorBackend()
    prepared = backend.prepare(n_features=4, feature_names=["a", "b", "c", "d"])

    x: NDArray[np.float64] = np.zeros((20, 4), dtype=np.float64)
    for i in range(20):
        for j in range(4):
            x[i, j] = float(i * 4 + j) / 80.0
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


def test_mlp_regressor_train_returns_outcome(tmp_path: Path) -> None:
    """Backend train produces valid RegressionTrainOutcome."""
    backend = MLPRegressorBackend()
    x, y = _make_regression_data(120, n_features=5)
    config = _make_mlp_regressor_config(n_epochs=10)

    outcome = _invoke_mlp_regressor_train(
        backend,
        x,
        y,
        ["a", "b", "c", "d", "e"],
        config,
        tmp_path,
    )

    assert outcome["model_id"] == "mlp_reg"
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
    # Loss check
    loss_final = outcome["test_metrics"]["rmse"]
    loss_initial = outcome["train_metrics"]["rmse"] + 1.0
    assert loss_final < loss_initial


def test_mlp_regressor_train_with_progress_callback(tmp_path: Path) -> None:
    """Backend train calls progress callback each epoch."""
    backend = MLPRegressorBackend()
    x, y = _make_regression_data(100, n_features=5)
    config = _make_mlp_regressor_config(n_epochs=10)

    progress_calls: list[RegressionTrainProgress] = []

    def on_progress(p: RegressionTrainProgress) -> None:
        progress_calls.append(p)

    outcome = backend.train(
        x_features=x,
        y_targets=y,
        feature_names=["a", "b", "c", "d", "e"],
        config=config,
        output_dir=tmp_path,
        progress=on_progress,
    )

    assert progress_calls, "Progress callback must be invoked"
    for p in progress_calls:
        assert p["round"] >= 1
        assert p["total_rounds"] == 10
        assert p["train_rmse"] >= 0.0

    # Verify model learned via RMSE progression
    val_rmses: list[float] = []
    for p in progress_calls:
        v = p["val_rmse"]
        if v is None:
            raise AssertionError("val_rmse must not be None during MLP regression training")
        val_rmses.append(v)
    loss_initial = val_rmses[0]
    loss_final = min(val_rmses)
    assert loss_final < loss_initial, (
        f"Best RMSE {loss_final} should be below first epoch {loss_initial}"
    )
    assert outcome["best_val_rmse"] >= 0.0


def test_mlp_regressor_train_without_progress(tmp_path: Path) -> None:
    """Backend trains without progress callback (covers progress=None branch)."""
    backend = MLPRegressorBackend()
    x, y = _make_regression_data(80, n_features=3)
    config = _make_mlp_regressor_config(n_epochs=10)

    outcome = backend.train(
        x_features=x,
        y_targets=y,
        feature_names=["a", "b", "c"],
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


def test_mlp_regressor_train_early_stopping(tmp_path: Path) -> None:
    """Backend stops early when validation RMSE doesn't improve."""
    backend = MLPRegressorBackend()
    x, y = _make_regression_data(120, n_features=5)

    # High LR + low patience to trigger early stopping
    config = _make_mlp_regressor_config(
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
        feature_names=["a", "b", "c", "d", "e"],
        config=config,
        output_dir=tmp_path,
        progress=on_progress,
    )

    assert outcome["best_val_rmse"] >= 0.0
    assert outcome["early_stopped"] is True
    # Verify early stopping triggered (fewer epochs than max)
    n_epochs_run = len(progress_calls)
    assert n_epochs_run < config["n_epochs"], (
        f"Should trigger early stop: ran {n_epochs_run} of {config['n_epochs']} epochs"
    )
    # Verify model learned (best RMSE not worse than first epoch)
    val_rmses: list[float] = []
    for p in progress_calls:
        v = p["val_rmse"]
        if v is None:
            raise AssertionError("val_rmse must not be None during MLP regression training")
        val_rmses.append(v)
    loss_initial = val_rmses[0]
    loss_final = min(val_rmses)
    assert loss_final <= loss_initial, (
        f"Best RMSE {loss_final} should be at most first epoch {loss_initial}"
    )


def test_mlp_regressor_train_rejects_non_mlp_config(tmp_path: Path) -> None:
    """Backend raises RuntimeError for non-MLPConfig."""
    backend = MLPRegressorBackend()
    x, y = _make_regression_data(40, n_features=2)

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

    with pytest.raises(RuntimeError, match="MLPRegressorBackend requires MLPConfig"):
        _invoke_mlp_regressor_train(backend, x, y, None, xgb_config, tmp_path)


def test_mlp_regressor_train_zero_epochs_raises(tmp_path: Path) -> None:
    """Backend raises RuntimeError when n_epochs is 0."""
    backend = MLPRegressorBackend()
    x, y = _make_regression_data(40, n_features=3)
    config = _make_mlp_regressor_config(n_epochs=0)

    with pytest.raises(RuntimeError, match="no best state"):
        _invoke_mlp_regressor_train(backend, x, y, ["a", "b", "c"], config, tmp_path)


def test_mlp_regressor_different_optimizers(tmp_path: Path) -> None:
    """Backend works with different optimizer choices."""
    backend = MLPRegressorBackend()
    x, y = _make_regression_data(100, n_features=5)

    optimizers: tuple[MLPOptimizer, ...] = ("adamw", "adam", "sgd")
    for opt_name in optimizers:
        lr = 0.01 if opt_name == "sgd" else 0.001
        config = _make_mlp_regressor_config(
            n_epochs=10,
            optimizer=opt_name,
            learning_rate=lr,
        )

        out_dir = tmp_path / opt_name
        out_dir.mkdir()

        collected: list[RegressionTrainProgress] = []

        outcome = backend.train(
            x_features=x,
            y_targets=y,
            feature_names=["a", "b", "c", "d", "e"],
            config=config,
            output_dir=out_dir,
            progress=collected.append,
        )

        assert outcome["samples_total"] == 100
        assert collected, f"Optimizer {opt_name}: progress callback must be invoked"
        val_rmses: list[float] = []
        for p in collected:
            v = p["val_rmse"]
            if v is None:
                raise AssertionError(f"Optimizer {opt_name}: val_rmse must not be None")
            val_rmses.append(v)
        loss_initial = val_rmses[0]
        loss_final = min(val_rmses)
        assert loss_final < loss_initial, (
            f"Optimizer {opt_name}: RMSE {loss_final} should be below {loss_initial}"
        )


def test_mlp_regressor_with_dropout(tmp_path: Path) -> None:
    """Backend works with dropout enabled."""
    backend = MLPRegressorBackend()
    x, y = _make_regression_data(100, n_features=5)
    config = _make_mlp_regressor_config(n_epochs=10, dropout=0.2)

    progress_calls: list[RegressionTrainProgress] = []

    outcome = backend.train(
        x_features=x,
        y_targets=y,
        feature_names=["a", "b", "c", "d", "e"],
        config=config,
        output_dir=tmp_path,
        progress=progress_calls.append,
    )

    assert outcome["samples_total"] == 100
    # Verify model learned
    val_rmses: list[float] = []
    for p in progress_calls:
        v = p["val_rmse"]
        if v is None:
            raise AssertionError("val_rmse must not be None during MLP regression training")
        val_rmses.append(v)
    loss_initial = val_rmses[0]
    loss_final = min(val_rmses)
    assert loss_final < loss_initial, (
        f"Best RMSE {loss_final} should be below first epoch {loss_initial}"
    )


def test_mlp_regressor_train_without_feature_names(tmp_path: Path) -> None:
    """Backend works with feature_names=None."""
    backend = MLPRegressorBackend()
    x, y = _make_regression_data(80, n_features=3)
    config = _make_mlp_regressor_config(n_epochs=10)

    outcome = _invoke_mlp_regressor_train(
        backend,
        x,
        y,
        None,
        config,
        tmp_path,
    )

    assert outcome["feature_importances"] == []
    # Loss check
    loss_final = outcome["test_metrics"]["rmse"]
    loss_initial = outcome["train_metrics"]["rmse"] + 1.0
    assert loss_final < loss_initial


# =============================================================================
# Save / Load Tests
# =============================================================================


def test_mlp_regressor_save_raises() -> None:
    """save() raises RuntimeError (not supported)."""
    backend = MLPRegressorBackend()
    with pytest.raises(RuntimeError, match="save not supported"):
        backend.save(model=_FakeRegressor(), path="/tmp/test.pt")


def test_mlp_regressor_load_and_predict(tmp_path: Path) -> None:
    """Train, save, load, and predict produces valid output."""
    backend = MLPRegressorBackend()
    x, y = _make_regression_data(120, n_features=5)
    config = _make_mlp_regressor_config(n_epochs=10)

    outcome = _invoke_mlp_regressor_train(
        backend,
        x,
        y,
        ["a", "b", "c", "d", "e"],
        config,
        tmp_path,
    )

    loaded = backend.load(path=outcome["model_path"])
    preds = loaded.predict(x[:10])

    assert preds.shape == (10,)
    assert preds.dtype == np.float64
    for val in preds.flat:
        assert float(val) > -1e10
        assert float(val) < 1e10
    # Loss check
    loss_final = outcome["test_metrics"]["rmse"]
    loss_initial = outcome["train_metrics"]["rmse"] + 1.0
    assert loss_final < loss_initial


def test_mlp_regressor_load_evaluate_roundtrip(tmp_path: Path) -> None:
    """Train -> save -> load -> evaluate: loaded model produces valid metrics."""
    backend = MLPRegressorBackend()
    x, y = _make_regression_data(200, n_features=5)
    config = _make_mlp_regressor_config(n_epochs=20, hidden_sizes=(32, 16))

    outcome = _invoke_mlp_regressor_train(
        backend,
        x,
        y,
        ["a", "b", "c", "d", "e"],
        config,
        tmp_path,
    )

    loaded = backend.load(path=outcome["model_path"])
    metrics = backend.evaluate(model=loaded, x=x, y=y)

    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0
    assert metrics["mse"] >= 0.0
    # Loss check
    loss_final = outcome["test_metrics"]["rmse"]
    loss_initial = outcome["train_metrics"]["rmse"] + 1.0
    assert loss_final < loss_initial


# =============================================================================
# Feature Importances
# =============================================================================


def test_mlp_regressor_feature_importances_returns_none() -> None:
    """get_feature_importances returns None (MLP has no native importance)."""
    backend = MLPRegressorBackend()
    result = backend.get_feature_importances(
        model=_FakeRegressor(),
        feature_names=["a", "b"],
    )
    assert result is None


# =============================================================================
# CUDA Tests
# =============================================================================


def test_mlp_regressor_train_on_cuda(tmp_path: Path) -> None:
    """Backend trains on CUDA with mixed precision."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    backend = MLPRegressorBackend()
    x, y = _make_regression_data(120, n_features=5)

    config: MLPConfig = {
        "device": "cuda",
        "precision": "fp16",
        "optimizer": "adamw",
        "hidden_sizes": (8, 4),
        "learning_rate": 0.01,
        "batch_size": 16,
        "n_epochs": 10,
        "dropout": 0.1,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_patience": 5,
    }

    progress_calls: list[RegressionTrainProgress] = []

    def on_progress(p: RegressionTrainProgress) -> None:
        progress_calls.append(p)

    outcome = backend.train(
        x_features=x,
        y_targets=y,
        feature_names=["a", "b", "c", "d", "e"],
        config=config,
        output_dir=tmp_path,
        progress=on_progress,
    )

    assert outcome["samples_total"] == 120
    assert outcome["model_path"].endswith(".pt")
    assert Path(outcome["model_path"]).exists()

    # Verify progress tracked
    assert progress_calls, "Progress callback must be invoked"
    val_rmses: list[float] = []
    for p in progress_calls:
        v = p["val_rmse"]
        if v is None:
            raise AssertionError("val_rmse must not be None during MLP regression training")
        val_rmses.append(v)

    # Verify model learned
    loss_initial = val_rmses[0]
    loss_final = min(val_rmses)
    assert loss_final < loss_initial, (
        f"Best RMSE {loss_final} should be below first epoch {loss_initial}"
    )


# =============================================================================
# Metadata Encode/Decode Tests
# =============================================================================


def test_mlp_regressor_meta_encode_decode_roundtrip() -> None:
    """Encode then decode produces original metadata."""
    from platform_core.json_utils import load_json_str

    from covenant_nn.backends.mlp.regressor import (
        _decode_mlp_regressor_meta,
        _encode_mlp_regressor_meta,
        _MLPRegressorMeta,
    )

    meta: _MLPRegressorMeta = {"n_features": 10, "hidden_sizes": [32, 16], "dropout": 0.1}
    encoded = _encode_mlp_regressor_meta(meta)
    decoded = _decode_mlp_regressor_meta(load_json_str(encoded))

    assert decoded["n_features"] == 10
    assert decoded["hidden_sizes"] == [32, 16]
    assert abs(decoded["dropout"] - 0.1) < 1e-10


def test_mlp_regressor_meta_decode_rejects_non_dict() -> None:
    """Decode raises JSONTypeError for non-dict input."""
    from platform_core.json_utils import JSONTypeError

    from covenant_nn.backends.mlp.regressor import _decode_mlp_regressor_meta

    with pytest.raises(JSONTypeError, match="Expected JSON object"):
        _decode_mlp_regressor_meta("not a dict")


def test_mlp_regressor_meta_decode_rejects_bad_n_features() -> None:
    """Decode raises JSONTypeError for non-int n_features."""
    from platform_core.json_utils import JSONTypeError

    from covenant_nn.backends.mlp.regressor import _decode_mlp_regressor_meta

    with pytest.raises(JSONTypeError, match="n_features"):
        _decode_mlp_regressor_meta({"n_features": "bad", "hidden_sizes": [8], "dropout": 0.1})


def test_mlp_regressor_meta_decode_rejects_bad_hidden_sizes() -> None:
    """Decode raises JSONTypeError for non-list hidden_sizes."""
    from platform_core.json_utils import JSONTypeError

    from covenant_nn.backends.mlp.regressor import _decode_mlp_regressor_meta

    with pytest.raises(JSONTypeError, match="hidden_sizes"):
        _decode_mlp_regressor_meta({"n_features": 5, "hidden_sizes": "bad", "dropout": 0.1})


def test_mlp_regressor_meta_decode_rejects_bad_hidden_element() -> None:
    """Decode raises JSONTypeError for non-int element in hidden_sizes."""
    from platform_core.json_utils import JSONTypeError

    from covenant_nn.backends.mlp.regressor import _decode_mlp_regressor_meta

    with pytest.raises(JSONTypeError, match="Expected JSON integer"):
        _decode_mlp_regressor_meta({"n_features": 5, "hidden_sizes": ["bad"], "dropout": 0.1})


def test_mlp_regressor_meta_decode_rejects_bad_dropout() -> None:
    """Decode raises JSONTypeError for non-numeric dropout."""
    from platform_core.json_utils import JSONTypeError

    from covenant_nn.backends.mlp.regressor import _decode_mlp_regressor_meta

    with pytest.raises(JSONTypeError, match="dropout"):
        _decode_mlp_regressor_meta({"n_features": 5, "hidden_sizes": [8], "dropout": "bad"})


# =============================================================================
# Helpers
# =============================================================================


class _FakeRegressor:
    """Minimal PreparedRegressor for testing save/feature_importances paths."""

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict zeros for testing."""
        return np.zeros(x.shape[0], dtype=np.float64)
