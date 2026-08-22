"""Tests for model loaders for feature importance explanation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_nn.backends.lstm.backend import FC_STATE_PREFIX, LSTM_STATE_PREFIX
from covenant_nn.backends.lstm.sequences import compute_features_per_step
from numpy.typing import NDArray

from covenant_radar_api.worker._explain_loaders import (
    LSTMModelConfig,
    load_gradient_model,
    load_model_for_backend,
)
from tests._explain_loaders_fixtures import (
    _create_lstm_model,
    _LinearFactory,
    _LinearProtocol,
    _LSTMFactory,
    _LSTMProtocol,
    _TensorProtocol,
    _TorchSaveFn,
)


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
