"""Shared fixtures and helpers for test_explain_loaders splits."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from covenant_nn.backends.lstm.backend import FC_STATE_PREFIX, LSTM_STATE_PREFIX
from covenant_nn.backends.lstm.sequences import compute_features_per_step
from numpy.typing import NDArray

from covenant_radar_api.worker._explain_loaders import (
    LSTMModelConfig,
    MLPModelConfig,
)


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
