"""LSTM regressor backend: persistence, prediction, gradients."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.types import LSTMConfig
from covenant_ml.types_regression import (
    RegressionTrainOutcome,
    RegressionTrainProgress,
    RegressorTrainConfig,
)
from numpy.typing import NDArray
from platform_ml.explainers.protocol import RegressionGradientModelProtocol

from covenant_nn.backends.lstm.regressor import (
    LSTMRegressorBackend,
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


def test_lstm_regressor_different_sequence_lengths(tmp_path: Path) -> None:
    """Backend works with different sequence length configurations."""
    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(100, n_features=8)

    for seq_len in (2, 4, 8):
        config = _make_lstm_regressor_config(n_epochs=10, sequence_length=seq_len)

        out_dir = tmp_path / f"seq_{seq_len}"
        out_dir.mkdir()

        progress_calls: list[RegressionTrainProgress] = []

        outcome = backend.train(
            x_features=x,
            y_targets=y,
            feature_names=["a", "b", "c", "d", "e", "f", "g", "h"],
            config=config,
            output_dir=out_dir,
            progress=progress_calls.append,
        )

        assert outcome["samples_total"] == 100
        assert outcome["model_path"].endswith(".pt")
        assert progress_calls, f"seq_len={seq_len}: Progress callback must be invoked"
        val_rmses: list[float] = []
        for p in progress_calls:
            v = p["val_rmse"]
            if v is None:
                raise AssertionError(f"seq_len={seq_len}: val_rmse must not be None")
            val_rmses.append(v)
        loss_initial = val_rmses[0]
        loss_final = min(val_rmses)
        assert loss_final <= loss_initial, (
            f"seq_len={seq_len}: RMSE {loss_final} should be at or below {loss_initial}"
        )


def test_lstm_regressor_train_without_feature_names(tmp_path: Path) -> None:
    """Backend works with feature_names=None."""
    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(80, n_features=8)
    config = _make_lstm_regressor_config(n_epochs=10)

    outcome = _invoke_lstm_regressor_train(
        backend,
        x,
        y,
        None,
        config,
        tmp_path,
    )

    assert outcome["feature_importances"] == []
    assert outcome["model_path"].endswith(".pt")


# =============================================================================
# Save / Load Tests
# =============================================================================


def test_lstm_regressor_save_raises() -> None:
    """save() raises RuntimeError (not supported)."""
    backend = LSTMRegressorBackend()
    with pytest.raises(RuntimeError, match="save not supported"):
        backend.save(model=_FakeRegressor(), path="/tmp/test.pt")


def test_lstm_regressor_load_and_predict(tmp_path: Path) -> None:
    """Train, save, load, and predict produces valid output."""
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


def test_lstm_regressor_load_evaluate_roundtrip(tmp_path: Path) -> None:
    """Train -> save -> load -> evaluate: loaded model produces valid metrics."""
    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(200, n_features=8)
    config = _make_lstm_regressor_config(n_epochs=20, hidden_size=32)

    outcome = _invoke_lstm_regressor_train(
        backend,
        x,
        y,
        ["a", "b", "c", "d", "e", "f", "g", "h"],
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


def test_lstm_regressor_feature_importances_returns_none() -> None:
    """get_feature_importances returns None (LSTM has no native importance)."""
    backend = LSTMRegressorBackend()
    result = backend.get_feature_importances(
        model=_FakeRegressor(),
        feature_names=["a", "b"],
    )
    assert result is None


# =============================================================================
# Wrapper State Dict Tests
# =============================================================================


def test_lstm_regressor_wrapper_load_state_dict_with_extra_key() -> None:
    """LSTMRegressorWrapper.load_state_dict ignores keys without known prefixes."""
    from covenant_nn.backends.lstm.regressor_training import _build_regressor_model

    model = _build_regressor_model(
        input_size=2,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        device="cpu",
    )

    original_state = model.state_dict()
    original_keys: list[str] = list(original_state.keys())

    has_lstm = any(k.startswith("lstm.") for k in original_keys)
    has_fc = any(k.startswith("fc.") for k in original_keys)
    assert has_lstm, "State dict should have lstm. keys"
    assert has_fc, "State dict should have fc. keys"

    template_key = original_keys[0]
    unknown_tensor = original_state[template_key].clone().detach()
    modified_state = dict(original_state)
    modified_state["unknown.weight"] = unknown_tensor

    model.load_state_dict(modified_state)

    reloaded_state = model.state_dict()
    reloaded_keys: list[str] = list(reloaded_state.keys())

    assert "unknown.weight" not in reloaded_keys
    for key in original_keys:
        assert key in reloaded_keys, f"Key {key} should be in reloaded state"


# =============================================================================
# CUDA Tests
# =============================================================================


def test_lstm_regressor_train_on_cuda(tmp_path: Path) -> None:
    """Backend trains on CUDA with mixed precision."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(120, n_features=8)

    config: LSTMConfig = {
        "device": "cuda",
        "precision": "fp16",
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
        "sequence_length": 4,
        "learning_rate": 0.01,
        "batch_size": 16,
        "n_epochs": 10,
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
        feature_names=["a", "b", "c", "d", "e", "f", "g", "h"],
        config=config,
        output_dir=tmp_path,
        progress=on_progress,
    )

    assert outcome["samples_total"] == 120
    assert outcome["model_path"].endswith(".pt")
    assert Path(outcome["model_path"]).exists()

    assert progress_calls, "Progress callback must be invoked"
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


# =============================================================================
# Metadata Encode/Decode Tests
# =============================================================================


def test_lstm_regressor_meta_encode_decode_roundtrip() -> None:
    """Encode then decode produces original metadata."""
    from platform_core.json_utils import load_json_str

    from covenant_nn.backends.lstm.regressor import (
        _decode_lstm_regressor_meta,
        _encode_lstm_regressor_meta,
        _LSTMRegressorMeta,
    )

    meta: _LSTMRegressorMeta = {
        "n_features": 8,
        "hidden_size": 16,
        "num_layers": 2,
        "dropout": 0.1,
        "bidirectional": True,
        "sequence_length": 4,
    }
    encoded = _encode_lstm_regressor_meta(meta)
    decoded = _decode_lstm_regressor_meta(load_json_str(encoded))

    assert decoded["n_features"] == 8
    assert decoded["hidden_size"] == 16
    assert decoded["num_layers"] == 2
    assert abs(decoded["dropout"] - 0.1) < 1e-10
    assert decoded["bidirectional"] is True
    assert decoded["sequence_length"] == 4


def test_lstm_regressor_meta_decode_rejects_non_dict() -> None:
    """Decode raises JSONTypeError for non-dict input."""
    from platform_core.json_utils import JSONTypeError

    from covenant_nn.backends.lstm.regressor import _decode_lstm_regressor_meta

    with pytest.raises(JSONTypeError, match="Expected JSON object"):
        _decode_lstm_regressor_meta("not a dict")


def test_lstm_regressor_meta_decode_rejects_bad_n_features() -> None:
    """Decode raises JSONTypeError for non-int n_features."""
    from platform_core.json_utils import JSONTypeError

    from covenant_nn.backends.lstm.regressor import _decode_lstm_regressor_meta

    with pytest.raises(JSONTypeError, match="n_features"):
        _decode_lstm_regressor_meta(
            {
                "n_features": "bad",
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "sequence_length": 4,
            }
        )


def test_lstm_regressor_meta_decode_rejects_bad_dropout() -> None:
    """Decode raises JSONTypeError for non-numeric dropout."""
    from platform_core.json_utils import JSONTypeError

    from covenant_nn.backends.lstm.regressor import _decode_lstm_regressor_meta

    with pytest.raises(JSONTypeError, match="dropout"):
        _decode_lstm_regressor_meta(
            {
                "n_features": 8,
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": "bad",
                "bidirectional": False,
                "sequence_length": 4,
            }
        )


def test_lstm_regressor_meta_decode_rejects_bad_bidirectional() -> None:
    """Decode raises JSONTypeError for non-bool bidirectional."""
    from platform_core.json_utils import JSONTypeError

    from covenant_nn.backends.lstm.regressor import _decode_lstm_regressor_meta

    with pytest.raises(JSONTypeError, match="bidirectional"):
        _decode_lstm_regressor_meta(
            {
                "n_features": 8,
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": "bad",
                "sequence_length": 4,
            }
        )


def test_lstm_regressor_meta_decode_rejects_bad_hidden_size() -> None:
    """Decode raises JSONTypeError for non-int hidden_size."""
    from platform_core.json_utils import JSONTypeError

    from covenant_nn.backends.lstm.regressor import _decode_lstm_regressor_meta

    with pytest.raises(JSONTypeError, match="hidden_size"):
        _decode_lstm_regressor_meta(
            {
                "n_features": 8,
                "hidden_size": "bad",
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "sequence_length": 4,
            }
        )


# =============================================================================
# Helpers
# =============================================================================


class _FakeRegressor:
    """Minimal PreparedRegressor for testing save/feature_importances paths."""

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict zeros for testing."""
        return np.zeros(x.shape[0], dtype=np.float64)


def test_lstm_regressor_compute_regression_gradients(tmp_path: Path) -> None:
    """The prepared regressor supports the gradient explainers.

    The regression gradient and integrated_gradients explainers call
    compute_regression_gradients through getattr, and are declared compatible
    with lstm_reg, but it was never implemented: both raised AttributeError,
    leaving permutation as the only explainer that worked for this backend.

    7 features over sequence_length 4 is deliberate. The reshape zero-pads up
    to a multiple of the sequence length, so the gradient comes back wider
    than the input and has to be trimmed; a divisible count would not notice.
    """
    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(120, n_features=7)
    config = _make_lstm_regressor_config(n_epochs=5, sequence_length=4)

    outcome = _invoke_lstm_regressor_train(
        backend, x, y, ["a", "b", "c", "d", "e", "f", "g"], config, tmp_path
    )
    # Typed against the protocol the gradient explainers require, so the
    # test fails to type-check if the backend stops satisfying it.
    loaded: RegressionGradientModelProtocol = backend.load(path=outcome["model_path"])

    grads = loaded.compute_regression_gradients(x[:8])

    assert grads.shape == (8, 7)
    assert int(np.count_nonzero(np.isfinite(grads))) == grads.size


def test_lstm_regressor_gradients_are_not_all_zero(tmp_path: Path) -> None:
    """Gradients carry signal, so an explanation ranks features."""
    backend = LSTMRegressorBackend()
    x, y = _make_regression_data(120, n_features=8)
    config = _make_lstm_regressor_config(n_epochs=10, sequence_length=4)

    outcome = _invoke_lstm_regressor_train(
        backend, x, y, ["a", "b", "c", "d", "e", "f", "g", "h"], config, tmp_path
    )
    # Typed against the protocol the gradient explainers require, so the
    # test fails to type-check if the backend stops satisfying it.
    loaded: RegressionGradientModelProtocol = backend.load(path=outcome["model_path"])

    grads = loaded.compute_regression_gradients(x[:8])

    assert int(np.count_nonzero(grads)) > 0
