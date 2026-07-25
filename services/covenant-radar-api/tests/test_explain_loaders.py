"""Tests for model loaders for feature importance explanation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
from covenant_nn.backends.lstm.backend import FC_STATE_PREFIX, LSTM_STATE_PREFIX
from covenant_nn.backends.lstm.sequences import compute_features_per_step
from numpy.typing import NDArray

from covenant_radar_api.worker._explain_loaders import (
    LSTMModelConfig,
    MLPModelConfig,
    load_gradient_model,
    load_model_for_backend,
)

# ---------------------------------------------------------------------------
# Torch Protocol Types for Tests
# ---------------------------------------------------------------------------


class _TensorProtocol(Protocol):
    """Protocol for torch.Tensor."""

    @property
    def shape(self) -> tuple[int, ...]: ...


class _ModuleProtocol(Protocol):
    """Protocol for nn.Module."""

    def state_dict(self) -> dict[str, _TensorProtocol]: ...


class _SequentialProtocol(Protocol):
    """Protocol for nn.Sequential."""

    def state_dict(self) -> dict[str, _TensorProtocol]: ...


class _LSTMProtocol(Protocol):
    """Protocol for nn.LSTM."""

    def state_dict(self) -> dict[str, _TensorProtocol]: ...


class _LinearProtocol(Protocol):
    """Protocol for nn.Linear."""

    def state_dict(self) -> dict[str, _TensorProtocol]: ...


class _TorchSaveFn(Protocol):
    """Protocol for torch.save function."""

    def __call__(self, obj: dict[str, _TensorProtocol], f: str) -> None: ...


# ---------------------------------------------------------------------------
# Factory Protocols for Torch Classes
# ---------------------------------------------------------------------------


class _LinearFactory(Protocol):
    """Factory protocol for nn.Linear constructor."""

    def __call__(self, in_features: int, out_features: int) -> _LinearProtocol: ...


class _BatchNorm1dFactory(Protocol):
    """Factory protocol for nn.BatchNorm1d constructor."""

    def __call__(self, num_features: int) -> _ModuleProtocol: ...


class _ReLUFactory(Protocol):
    """Factory protocol for nn.ReLU constructor."""

    def __call__(self) -> _ModuleProtocol: ...


class _DropoutFactory(Protocol):
    """Factory protocol for nn.Dropout constructor."""

    def __call__(self, p: float) -> _ModuleProtocol: ...


class _SequentialFactory(Protocol):
    """Factory protocol for nn.Sequential constructor."""

    def __call__(self, *modules: _ModuleProtocol) -> _SequentialProtocol: ...


class _LSTMFactory(Protocol):
    """Factory protocol for nn.LSTM constructor."""

    def __call__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        *,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> _LSTMProtocol: ...


def _create_xgboost_model(model_path: Path, n_features: int = 10) -> None:
    """Create a simple XGBoost model for testing."""
    import xgboost as xgb

    rng = np.random.default_rng(42)
    x: NDArray[np.float64] = rng.random((100, n_features))
    y: NDArray[np.int64] = rng.integers(0, 2, size=100).astype(np.int64)

    model = xgb.XGBClassifier(
        n_estimators=5,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
    )
    model.fit(x, y)
    model.save_model(str(model_path))


def _create_lightgbm_model(model_path: Path, n_features: int = 10) -> None:
    """Create a simple LightGBM model for testing."""
    import lightgbm as lgb

    rng = np.random.default_rng(42)
    x: NDArray[np.float64] = rng.random((100, n_features))
    y: NDArray[np.int64] = rng.integers(0, 2, size=100).astype(np.int64)

    train_data = lgb.Dataset(x, label=y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 8,
        "learning_rate": 0.1,
        "verbose": -1,
    }
    booster = lgb.train(params, train_data, num_boost_round=5)
    booster.save_model(str(model_path))


def _create_mlp_model(model_path: Path, n_features: int = 10) -> MLPModelConfig:
    """Create a simple MLP model for testing and return its config."""
    torch_mod = __import__("torch")
    nn_mod = __import__("torch.nn", fromlist=["Module"])

    # Extract nn classes with factory protocols for proper typing
    linear_cls: _LinearFactory = nn_mod.Linear
    batchnorm_cls: _BatchNorm1dFactory = nn_mod.BatchNorm1d
    relu_cls: _ReLUFactory = nn_mod.ReLU
    dropout_cls: _DropoutFactory = nn_mod.Dropout
    sequential_cls: _SequentialFactory = nn_mod.Sequential

    hidden_sizes = (32, 16)
    dropout_rate = 0.1

    # Build model using typed references
    layers: list[_ModuleProtocol] = []
    in_f = n_features
    for width in hidden_sizes:
        layers.append(linear_cls(in_f, width))
        layers.append(batchnorm_cls(width))
        layers.append(relu_cls())
        if dropout_rate > 0.0:
            layers.append(dropout_cls(dropout_rate))
        in_f = width
    layers.append(linear_cls(in_f, 2))
    model: _SequentialProtocol = sequential_cls(*layers)

    # Save model with typed function
    save_fn: _TorchSaveFn = torch_mod.save
    state_dict: dict[str, _TensorProtocol] = model.state_dict()
    save_fn(state_dict, str(model_path))

    return MLPModelConfig(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout_rate,
    )


def _create_lstm_model(model_path: Path, n_features: int = 12) -> LSTMModelConfig:
    """Create a simple LSTM model for testing and return its config."""
    torch_mod = __import__("torch")
    nn_mod = __import__("torch.nn", fromlist=["LSTM", "Linear"])

    # Extract nn classes with factory protocols for proper typing
    lstm_cls: _LSTMFactory = nn_mod.LSTM
    linear_cls: _LinearFactory = nn_mod.Linear

    sequence_length = 4
    hidden_size = 16
    num_layers = 1
    dropout_rate = 0.0
    bidirectional = False

    input_size = compute_features_per_step(n_features, sequence_length)

    lstm: _LSTMProtocol = lstm_cls(
        input_size,
        hidden_size,
        num_layers,
        batch_first=True,
        dropout=dropout_rate,
        bidirectional=bidirectional,
    )

    num_directions = 2 if bidirectional else 1
    lstm_out_size = hidden_size * num_directions
    fc: _LinearProtocol = linear_cls(lstm_out_size, 2)

    # Save combined state dict with prefixes
    state_dict: dict[str, _TensorProtocol] = {}
    lstm_sd: dict[str, _TensorProtocol] = lstm.state_dict()
    fc_sd: dict[str, _TensorProtocol] = fc.state_dict()
    for k in lstm_sd:
        state_dict[f"{LSTM_STATE_PREFIX}{k}"] = lstm_sd[k]
    for k in fc_sd:
        state_dict[f"{FC_STATE_PREFIX}{k}"] = fc_sd[k]

    save_fn: _TorchSaveFn = torch_mod.save
    save_fn(state_dict, str(model_path))

    return LSTMModelConfig(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout_rate,
        bidirectional=bidirectional,
        sequence_length=sequence_length,
    )


class TestLoadModelForBackendXGBoost:
    """Tests for load_model_for_backend with XGBoost."""

    def test_loads_xgboost_model(self, tmp_path: Path) -> None:
        """Loads XGBoost model successfully."""
        model_path = tmp_path / "model.ubj"
        n_features = 10
        _create_xgboost_model(model_path, n_features)

        model = load_model_for_backend("xgboost", str(model_path))

        # Verify model can predict
        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((5, n_features))
        proba: NDArray[np.float64] = model.predict_proba(x)

        assert proba.shape == (5, 2)
        row_sums: NDArray[np.float64] = np.sum(proba, axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing XGBoost model."""
        model_path = tmp_path / "nonexistent.ubj"

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_model_for_backend("xgboost", str(model_path))


class TestLoadModelForBackendUnsupported:
    """Backends with no explain loader are named explicitly."""

    def test_raises_naming_the_backend(self, tmp_path: Path) -> None:
        """cleargbm, logreg and random_forest report the real problem.

        These are valid BackendName values that this module has no loader for.
        They previously fell through to the LSTM branch and surfaced as
        "lstm_config is required for LSTM backend", which named the wrong
        problem and was undiagnosable for the caller.

        Args:
            tmp_path: Pytest temporary directory unique to this test.
        """
        model_path = tmp_path / "model.bin"
        model_path.write_bytes(b"stub")

        for backend in ("cleargbm", "logreg", "random_forest"):
            with pytest.raises(ValueError, match=f"No explain loader for backend: {backend}"):
                load_model_for_backend(backend, str(model_path))


class TestLoadModelForBackendLightGBM:
    """Tests for load_model_for_backend with LightGBM."""

    def test_loads_lightgbm_model(self, tmp_path: Path) -> None:
        """Loads LightGBM model successfully."""
        model_path = tmp_path / "model.txt"
        n_features = 10
        _create_lightgbm_model(model_path, n_features)

        model = load_model_for_backend("lightgbm", str(model_path))

        # Verify model can predict
        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((5, n_features))
        proba: NDArray[np.float64] = model.predict_proba(x)

        assert proba.shape == (5, 2)
        row_sums: NDArray[np.float64] = np.sum(proba, axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_lightgbm_wrapper_predict_proba(self, tmp_path: Path) -> None:
        """LightGBM wrapper produces valid probability arrays."""
        model_path = tmp_path / "model.txt"
        n_features = 10
        _create_lightgbm_model(model_path, n_features)

        model = load_model_for_backend("lightgbm", str(model_path))

        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((10, n_features))
        proba: NDArray[np.float64] = model.predict_proba(x)

        # Verify probabilities sum to 1
        for i in range(10):
            row: NDArray[np.float64] = proba[i]
            p0: np.float64 = row[0]
            p1: np.float64 = row[1]
            assert float(p0) + float(p1) == pytest.approx(1.0)
            assert 0.0 <= float(p0) <= 1.0
            assert 0.0 <= float(p1) <= 1.0


class TestLoadModelForBackendMLP:
    """Tests for load_model_for_backend with MLP."""

    def test_loads_mlp_model(self, tmp_path: Path) -> None:
        """Loads MLP model successfully."""
        model_path = tmp_path / "model.pt"
        mlp_config = _create_mlp_model(model_path)

        model = load_model_for_backend("mlp", str(model_path), mlp_config=mlp_config)

        # Verify model can predict
        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((5, mlp_config["n_features"]))
        proba: NDArray[np.float64] = model.predict_proba(x)

        assert proba.shape == (5, 2)
        row_sums: NDArray[np.float64] = np.sum(proba, axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_mlp_compute_gradients(self, tmp_path: Path) -> None:
        """MLP model computes gradients successfully."""
        model_path = tmp_path / "model.pt"
        mlp_config = _create_mlp_model(model_path)

        model = load_gradient_model("mlp", str(model_path), mlp_config=mlp_config)

        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((5, mlp_config["n_features"]))

        gradients: NDArray[np.float64] = model.compute_gradients(x, target_class=1)

        assert gradients.shape == (5, mlp_config["n_features"])

    def test_raises_without_mlp_config(self, tmp_path: Path) -> None:
        """Raises ValueError when mlp_config is missing."""
        model_path = tmp_path / "model.pt"
        _create_mlp_model(model_path)

        with pytest.raises(ValueError, match="mlp_config is required for MLP backend"):
            load_model_for_backend("mlp", str(model_path))

    def test_mlp_model_no_dropout(self, tmp_path: Path) -> None:
        """MLP model works with dropout=0."""
        torch_mod = __import__("torch")
        nn_mod = __import__("torch.nn", fromlist=["Module"])

        # Extract nn classes with factory protocols for proper typing
        linear_cls: _LinearFactory = nn_mod.Linear
        batchnorm_cls: _BatchNorm1dFactory = nn_mod.BatchNorm1d
        relu_cls: _ReLUFactory = nn_mod.ReLU
        sequential_cls: _SequentialFactory = nn_mod.Sequential

        n_features = 10
        hidden_sizes = (16,)
        dropout_rate = 0.0  # No dropout

        # Build model without dropout layers
        layers: list[_ModuleProtocol] = []
        in_f = n_features
        for width in hidden_sizes:
            layers.append(linear_cls(in_f, width))
            layers.append(batchnorm_cls(width))
            layers.append(relu_cls())
            in_f = width
        layers.append(linear_cls(in_f, 2))
        model: _SequentialProtocol = sequential_cls(*layers)

        model_path = tmp_path / "model_no_dropout.pt"
        save_fn: _TorchSaveFn = torch_mod.save
        state_dict: dict[str, _TensorProtocol] = model.state_dict()
        save_fn(state_dict, str(model_path))

        config = MLPModelConfig(
            n_features=n_features,
            hidden_sizes=hidden_sizes,
            dropout=dropout_rate,
        )

        loaded = load_model_for_backend("mlp", str(model_path), mlp_config=config)

        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((3, n_features))
        proba: NDArray[np.float64] = loaded.predict_proba(x)

        assert proba.shape == (3, 2)


class TestLoadModelForBackendLSTM:
    """Tests for load_model_for_backend with LSTM."""

    def test_loads_lstm_model(self, tmp_path: Path) -> None:
        """Loads LSTM model successfully."""
        model_path = tmp_path / "model.pt"
        lstm_config = _create_lstm_model(model_path)

        model = load_model_for_backend("lstm", str(model_path), lstm_config=lstm_config)

        # Verify model can predict
        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((5, lstm_config["n_features"]))
        proba: NDArray[np.float64] = model.predict_proba(x)

        assert proba.shape == (5, 2)
        row_sums: NDArray[np.float64] = np.sum(proba, axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_lstm_compute_gradients(self, tmp_path: Path) -> None:
        """LSTM model computes gradients successfully."""
        model_path = tmp_path / "model.pt"
        lstm_config = _create_lstm_model(model_path)

        model = load_gradient_model("lstm", str(model_path), lstm_config=lstm_config)

        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((5, lstm_config["n_features"]))

        gradients: NDArray[np.float64] = model.compute_gradients(x, target_class=1)

        assert gradients.shape == (5, lstm_config["n_features"])

    def test_lstm_with_features_not_divisible_by_sequence_length(self, tmp_path: Path) -> None:
        """A feature count that does not divide evenly still loads and predicts.

        Regression guard for two defects that were invisible while every
        fixture used 12 features over 4 timesteps, where floor and ceil agree:

        - input_size was floored here but ceiled during training, so the
          explain path built a differently-shaped LSTM than the checkpoint.
        - the reshape was a bare `.reshape` with that floored width, silently
          dropping trailing features instead of zero-padding them.

        Args:
            tmp_path: Pytest temporary directory unique to this test.
        """
        model_path = tmp_path / "model.pt"
        # 13 features over 4 timesteps: ceil gives 4 per step and pads to 16,
        # floor would give 3 and drop a feature entirely.
        lstm_config = _create_lstm_model(model_path, n_features=13)

        model = load_gradient_model("lstm", str(model_path), lstm_config=lstm_config)

        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((5, 13))

        proba: NDArray[np.float64] = model.predict_proba(x)
        assert proba.shape == (5, 2)
        row_sums: NDArray[np.float64] = np.sum(proba, axis=1)
        assert np.allclose(row_sums, 1.0)

        # Gradients come back trimmed to the real feature count, not the
        # padded width.
        gradients: NDArray[np.float64] = model.compute_gradients(x, target_class=1)
        assert gradients.shape == (5, 13)

    def test_raises_without_lstm_config(self, tmp_path: Path) -> None:
        """Raises ValueError when lstm_config is missing."""
        model_path = tmp_path / "model.pt"
        _create_lstm_model(model_path)

        with pytest.raises(ValueError, match="lstm_config is required for LSTM backend"):
            load_model_for_backend("lstm", str(model_path))

    def test_lstm_ignores_extra_state_dict_keys(self, tmp_path: Path) -> None:
        """LSTM model ignores keys that don't start with 'lstm.' or 'fc.'."""
        torch_mod = __import__("torch")
        nn_mod = __import__("torch.nn", fromlist=["LSTM", "Linear"])

        # Extract nn classes with factory protocols for proper typing
        lstm_cls: _LSTMFactory = nn_mod.LSTM
        linear_cls: _LinearFactory = nn_mod.Linear

        n_features = 12
        sequence_length = 4
        hidden_size = 16
        num_layers = 1
        dropout_rate = 0.0
        bidirectional = False

        input_size = compute_features_per_step(n_features, sequence_length)

        lstm: _LSTMProtocol = lstm_cls(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=bidirectional,
        )

        fc: _LinearProtocol = linear_cls(hidden_size, 2)

        # Build state dict with extra keys that should be ignored
        state_dict: dict[str, _TensorProtocol] = {}
        lstm_sd: dict[str, _TensorProtocol] = lstm.state_dict()
        fc_sd: dict[str, _TensorProtocol] = fc.state_dict()
        for k in lstm_sd:
            state_dict[f"{LSTM_STATE_PREFIX}{k}"] = lstm_sd[k]
        for k in fc_sd:
            state_dict[f"{FC_STATE_PREFIX}{k}"] = fc_sd[k]
        # Add extra key that doesn't match either prefix
        state_dict["extra_key"] = lstm_sd[next(iter(lstm_sd))]

        model_path = tmp_path / "model_extra_keys.pt"
        save_fn: _TorchSaveFn = torch_mod.save
        save_fn(state_dict, str(model_path))

        config = LSTMModelConfig(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            bidirectional=bidirectional,
            sequence_length=sequence_length,
        )

        # Should load successfully, ignoring the extra key
        loaded = load_model_for_backend("lstm", str(model_path), lstm_config=config)

        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((3, n_features))
        proba: NDArray[np.float64] = loaded.predict_proba(x)

        assert proba.shape == (3, 2)

    def test_lstm_bidirectional(self, tmp_path: Path) -> None:
        """LSTM model works with bidirectional=True."""
        torch_mod = __import__("torch")
        nn_mod = __import__("torch.nn", fromlist=["LSTM", "Linear"])

        # Extract nn classes with factory protocols for proper typing
        lstm_cls: _LSTMFactory = nn_mod.LSTM
        linear_cls: _LinearFactory = nn_mod.Linear

        n_features = 12
        sequence_length = 4
        hidden_size = 16
        num_layers = 1
        dropout_rate = 0.0
        bidirectional = True

        input_size = compute_features_per_step(n_features, sequence_length)

        lstm: _LSTMProtocol = lstm_cls(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=bidirectional,
        )

        num_directions = 2
        lstm_out_size = hidden_size * num_directions
        fc: _LinearProtocol = linear_cls(lstm_out_size, 2)

        state_dict: dict[str, _TensorProtocol] = {}
        lstm_sd: dict[str, _TensorProtocol] = lstm.state_dict()
        fc_sd: dict[str, _TensorProtocol] = fc.state_dict()
        for k in lstm_sd:
            state_dict[f"{LSTM_STATE_PREFIX}{k}"] = lstm_sd[k]
        for k in fc_sd:
            state_dict[f"{FC_STATE_PREFIX}{k}"] = fc_sd[k]

        model_path = tmp_path / "model_bidir.pt"
        save_fn: _TorchSaveFn = torch_mod.save
        save_fn(state_dict, str(model_path))

        config = LSTMModelConfig(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            bidirectional=bidirectional,
            sequence_length=sequence_length,
        )

        loaded = load_model_for_backend("lstm", str(model_path), lstm_config=config)

        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((3, n_features))
        proba: NDArray[np.float64] = loaded.predict_proba(x)

        assert proba.shape == (3, 2)

    def test_lstm_multi_layer_with_dropout(self, tmp_path: Path) -> None:
        """LSTM model works with num_layers > 1 and dropout."""
        torch_mod = __import__("torch")
        nn_mod = __import__("torch.nn", fromlist=["LSTM", "Linear"])

        # Extract nn classes with factory protocols for proper typing
        lstm_cls: _LSTMFactory = nn_mod.LSTM
        linear_cls: _LinearFactory = nn_mod.Linear

        n_features = 12
        sequence_length = 4
        hidden_size = 16
        num_layers = 2  # Multi-layer
        dropout_rate = 0.1  # Dropout between layers
        bidirectional = False

        input_size = compute_features_per_step(n_features, sequence_length)

        lstm: _LSTMProtocol = lstm_cls(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size
        fc: _LinearProtocol = linear_cls(lstm_out_size, 2)

        state_dict: dict[str, _TensorProtocol] = {}
        lstm_sd: dict[str, _TensorProtocol] = lstm.state_dict()
        fc_sd: dict[str, _TensorProtocol] = fc.state_dict()
        for k in lstm_sd:
            state_dict[f"{LSTM_STATE_PREFIX}{k}"] = lstm_sd[k]
        for k in fc_sd:
            state_dict[f"{FC_STATE_PREFIX}{k}"] = fc_sd[k]

        model_path = tmp_path / "model_multi.pt"
        save_fn: _TorchSaveFn = torch_mod.save
        save_fn(state_dict, str(model_path))

        config = LSTMModelConfig(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            bidirectional=bidirectional,
            sequence_length=sequence_length,
        )

        loaded = load_model_for_backend("lstm", str(model_path), lstm_config=config)

        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((3, n_features))
        proba: NDArray[np.float64] = loaded.predict_proba(x)

        assert proba.shape == (3, 2)


class TestLoadGradientModelErrors:
    """Tests for load_gradient_model error handling."""

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing model file."""
        model_path = tmp_path / "nonexistent.pt"

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_gradient_model(
                "mlp",
                str(model_path),
                mlp_config=MLPModelConfig(
                    n_features=10,
                    hidden_sizes=(32,),
                    dropout=0.0,
                ),
            )

    def test_raises_without_mlp_config(self, tmp_path: Path) -> None:
        """Raises ValueError when mlp_config is missing for MLP backend."""
        model_path = tmp_path / "model.pt"
        _create_mlp_model(model_path)

        with pytest.raises(ValueError, match="mlp_config is required for MLP backend"):
            load_gradient_model("mlp", str(model_path))

    def test_raises_without_lstm_config(self, tmp_path: Path) -> None:
        """Raises ValueError when lstm_config is missing for LSTM backend."""
        model_path = tmp_path / "model.pt"
        _create_lstm_model(model_path)

        with pytest.raises(ValueError, match="lstm_config is required for LSTM backend"):
            load_gradient_model("lstm", str(model_path))

    def test_raises_on_unsupported_backend(self, tmp_path: Path) -> None:
        """Raises ValueError for backends that don't support gradients."""
        model_path = tmp_path / "model.ubj"
        _create_xgboost_model(model_path)

        with pytest.raises(ValueError, match="does not support gradients"):
            load_gradient_model("xgboost", str(model_path))
