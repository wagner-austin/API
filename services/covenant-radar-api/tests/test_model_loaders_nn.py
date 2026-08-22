"""Tests for model loading functions.

Tests cover metadata decoding, model architecture building, and inference
for MLP, LSTM, LightGBM, LogReg, and RandomForest models.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.json_utils import JSONTypeError, dump_json_str

from covenant_radar_api.worker._model_loaders import (
    load_lstm_model,
    load_mlp_model,
)


class TestLoadMlpModel:
    """Tests for load_mlp_model function."""

    def test_load_mlp_model_metadata_mismatch(self, tmp_path: Path) -> None:
        """Raise error when metadata says wrong backend."""
        # Create LSTM metadata instead of MLP
        meta_content = dump_json_str(
            {
                "backend": "lstm",
                "n_features": 50,
                "sequence_length": 5,
                "hidden_size": 32,
                "num_layers": 1,
                "bidirectional": False,
                "dropout": 0.0,
            }
        )
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(meta_content, encoding="utf-8")

        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        with pytest.raises(JSONTypeError) as exc_info:
            load_mlp_model(model_path, meta_path)

        assert "Expected MLP metadata" in str(exc_info.value)


class TestLoadLstmModel:
    """Tests for load_lstm_model function."""

    def test_load_lstm_model_metadata_mismatch(self, tmp_path: Path) -> None:
        """Raise error when metadata says wrong backend."""
        # Create MLP metadata instead of LSTM
        meta_content = dump_json_str(
            {
                "backend": "mlp",
                "n_features": 25,
                "hidden_sizes": [64],
                "dropout": 0.1,
            }
        )
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(meta_content, encoding="utf-8")

        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        with pytest.raises(JSONTypeError) as exc_info:
            load_lstm_model(model_path, meta_path)

        assert "Expected LSTM metadata" in str(exc_info.value)


class TestBuildMlpModel:
    """Tests for _build_mlp_model function."""

    def test_build_mlp_model_creates_sequential(self) -> None:
        """Build MLP model and verify structure via predictor interface."""
        from covenant_radar_api.worker._model_loaders import (
            _build_mlp_model,
            _MLPPreparedForInference,
        )

        model = _build_mlp_model(
            n_features=10,
            hidden_sizes=[32, 16],
            dropout=0.1,
            device="cpu",
        )
        model.eval()

        # Test via predictor interface
        predictor = _MLPPreparedForInference(model)
        x: NDArray[np.float64] = np.random.randn(2, 10).astype(np.float64)
        proba = predictor.predict_proba(x)

        assert proba.shape == (2, 2)  # 2 classes output
        # Probabilities should sum to ~1 (implicitly validates finite values)
        sums: NDArray[np.float64] = proba.sum(axis=1)
        assert bool(np.allclose(sums, 1.0))

    def test_build_mlp_model_no_dropout(self) -> None:
        """Build MLP model without dropout."""
        from covenant_radar_api.worker._model_loaders import (
            _build_mlp_model,
            _MLPPreparedForInference,
        )

        model = _build_mlp_model(
            n_features=10,
            hidden_sizes=[32],
            dropout=0.0,  # No dropout
            device="cpu",
        )

        # Test via predictor interface
        predictor = _MLPPreparedForInference(model)
        x: NDArray[np.float64] = np.random.randn(2, 10).astype(np.float64)
        proba = predictor.predict_proba(x)

        assert proba.shape == (2, 2)
        # Probabilities should sum to ~1 (implicitly validates finite values)
        sums: NDArray[np.float64] = proba.sum(axis=1)
        assert bool(np.allclose(sums, 1.0))


class TestMlpPreparedForInference:
    """Tests for _MLPPreparedForInference class."""

    def test_mlp_predict_proba(self) -> None:
        """Test MLP predict_proba returns correct shape."""
        from covenant_radar_api.worker._model_loaders import (
            _build_mlp_model,
            _MLPPreparedForInference,
        )

        # Build and prepare model
        model = _build_mlp_model(
            n_features=10,
            hidden_sizes=[32, 16],
            dropout=0.0,
            device="cpu",
        )
        model.eval()

        predictor = _MLPPreparedForInference(model)

        # Create sample input
        x: NDArray[np.float64] = np.random.randn(5, 10).astype(np.float64)
        proba = predictor.predict_proba(x)

        # Check output shape (n_samples, n_classes)
        assert proba.shape == (5, 2)
        # Probabilities should sum to ~1
        sums: NDArray[np.float64] = proba.sum(axis=1)
        assert bool(np.allclose(sums, 1.0))


class TestLoadMlpModelFull:
    """Tests for full MLP model loading path."""

    def test_load_mlp_model_full_path(self, tmp_path: Path) -> None:
        """Load MLP model from saved state dict."""
        import torch

        from covenant_radar_api.worker._model_loaders import (
            _build_mlp_model,
            load_mlp_model,
        )

        # Build and save model
        model = _build_mlp_model(
            n_features=10,
            hidden_sizes=[32, 16],
            dropout=0.1,
            device="cpu",
        )

        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), str(model_path))

        # Create metadata
        meta_content = dump_json_str(
            {
                "backend": "mlp",
                "n_features": 10,
                "hidden_sizes": [32, 16],
                "dropout": 0.1,
            }
        )
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(meta_content, encoding="utf-8")

        # Load and verify by testing prediction
        loaded = load_mlp_model(model_path, meta_path)

        # Test prediction
        x: NDArray[np.float64] = np.random.randn(3, 10).astype(np.float64)
        proba = loaded.predict_proba(x)
        assert proba.shape == (3, 2)
        # Probabilities should sum to ~1
        sums: NDArray[np.float64] = proba.sum(axis=1)
        assert bool(np.allclose(sums, 1.0))


class TestLstmClassifierWrapper:
    """Tests for _LSTMClassifierWrapper class."""

    def test_lstm_wrapper_forward(self) -> None:
        """Test LSTM wrapper forward pass via predictor interface."""
        from covenant_radar_api.worker._model_loaders import (
            _build_lstm_model,
            _LSTMPreparedForInference,
        )

        # Build model using production function
        model, _ = _build_lstm_model(
            n_features=20,
            sequence_length=5,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            device="cpu",
        )
        model.eval()

        # Test via predictor interface
        predictor = _LSTMPreparedForInference(model, sequence_length=5)
        x: NDArray[np.float64] = np.random.randn(2, 20).astype(np.float64)
        proba = predictor.predict_proba(x)

        assert proba.shape == (2, 2)  # batch=2, classes=2
        sums: NDArray[np.float64] = proba.sum(axis=1)
        assert bool(np.allclose(sums, 1.0))

    def test_lstm_wrapper_eval_mode(self) -> None:
        """Test LSTM wrapper eval mode."""
        from covenant_radar_api.worker._model_loaders import _build_lstm_model

        model, _ = _build_lstm_model(
            n_features=20,
            sequence_length=5,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            device="cpu",
        )
        result = model.eval()

        assert result is model  # Returns self

    def test_lstm_wrapper_load_state_dict(self) -> None:
        """Test LSTM wrapper load_state_dict."""
        from covenant_radar_api.worker._model_loaders import _build_lstm_model

        model1, _ = _build_lstm_model(
            n_features=20,
            sequence_length=5,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            device="cpu",
        )

        # Get state dict from model1
        state_dict = model1.state_dict()

        # Create new model and load
        model2, _ = _build_lstm_model(
            n_features=20,
            sequence_length=5,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            device="cpu",
        )

        model2.load_state_dict(state_dict)
        # Should not raise

    def test_lstm_wrapper_load_state_dict_ignores_unknown_keys(self) -> None:
        """Test LSTM wrapper load_state_dict ignores keys without known prefixes."""
        from covenant_radar_api.worker._model_loaders import _build_lstm_model

        model1, _ = _build_lstm_model(
            n_features=20,
            sequence_length=5,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            device="cpu",
        )

        # Get state dict and add unknown key (reuse existing tensor for type safety)
        state_dict = model1.state_dict()
        # Get first tensor value from state dict and add it with unknown key
        first_key = next(iter(state_dict.keys()))
        state_dict["unknown_key"] = state_dict[first_key]

        # Create new model and load (should not raise, unknown key is ignored)
        model2, _ = _build_lstm_model(
            n_features=20,
            sequence_length=5,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            device="cpu",
        )

        model2.load_state_dict(state_dict)
        # Should complete without error - unknown key is silently ignored

    def test_lstm_wrapper_to_device(self) -> None:
        """Test LSTM wrapper device movement."""
        from covenant_radar_api.worker._model_loaders import _build_lstm_model

        model, _ = _build_lstm_model(
            n_features=20,
            sequence_length=5,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            device="cpu",
        )
        result = model.to("cpu")

        assert result is model  # Returns self


class TestBuildLstmModel:
    """Tests for _build_lstm_model function."""

    def test_build_lstm_model_creates_wrapper(self) -> None:
        """Build LSTM model and verify structure."""
        from covenant_radar_api.worker._model_loaders import (
            _build_lstm_model,
            _LSTMPreparedForInference,
        )

        model, input_size = _build_lstm_model(
            n_features=40,
            sequence_length=10,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            device="cpu",
        )

        assert input_size == 4  # 40 / 10

        # Test via predictor interface to verify model works
        model.eval()
        predictor = _LSTMPreparedForInference(model, sequence_length=10)
        x: NDArray[np.float64] = np.random.randn(2, 40).astype(np.float64)
        proba = predictor.predict_proba(x)

        assert proba.shape == (2, 2)  # batch=2, classes=2
        # Probabilities should sum to ~1 (implicitly validates finite values)
        sums: NDArray[np.float64] = proba.sum(axis=1)
        assert bool(np.allclose(sums, 1.0))

    def test_build_lstm_model_bidirectional(self) -> None:
        """Build bidirectional LSTM model."""
        from covenant_radar_api.worker._model_loaders import (
            _build_lstm_model,
            _LSTMPreparedForInference,
        )

        model, input_size = _build_lstm_model(
            n_features=40,
            sequence_length=10,
            hidden_size=32,
            num_layers=2,  # Multi-layer for dropout
            dropout=0.2,
            bidirectional=True,
            device="cpu",
        )

        assert input_size == 4  # 40 / 10

        model.eval()  # Set to eval mode to disable dropout

        # Test via predictor interface
        predictor = _LSTMPreparedForInference(model, sequence_length=10)
        x: NDArray[np.float64] = np.random.randn(2, 40).astype(np.float64)
        proba = predictor.predict_proba(x)

        assert proba.shape == (2, 2)  # batch=2, classes=2
        # Probabilities should sum to ~1 (implicitly validates finite values)
        sums: NDArray[np.float64] = proba.sum(axis=1)
        assert bool(np.allclose(sums, 1.0))


class TestLstmPreparedForInference:
    """Tests for _LSTMPreparedForInference class."""

    def test_lstm_predict_proba(self) -> None:
        """Test LSTM predict_proba returns correct shape."""
        from covenant_radar_api.worker._model_loaders import (
            _build_lstm_model,
            _LSTMPreparedForInference,
        )

        model, _ = _build_lstm_model(
            n_features=40,
            sequence_length=10,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            device="cpu",
        )
        model.eval()

        predictor = _LSTMPreparedForInference(model, sequence_length=10)

        # Create sample input (flat features)
        x: NDArray[np.float64] = np.random.randn(5, 40).astype(np.float64)
        proba = predictor.predict_proba(x)

        # Check output shape (n_samples, n_classes)
        assert proba.shape == (5, 2)
        # Probabilities should sum to ~1
        sums: NDArray[np.float64] = proba.sum(axis=1)
        assert bool(np.allclose(sums, 1.0))


class TestLoadLstmModelFull:
    """Tests for full LSTM model loading path."""

    def test_load_lstm_model_full_path(self, tmp_path: Path) -> None:
        """Load LSTM model from saved state dict."""
        import torch

        from covenant_radar_api.worker._model_loaders import (
            _build_lstm_model,
            load_lstm_model,
        )

        # Build model
        model, _ = _build_lstm_model(
            n_features=40,
            sequence_length=10,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            device="cpu",
        )

        # Save state dict using wrapper method
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), str(model_path))

        # Create metadata
        meta_content = dump_json_str(
            {
                "backend": "lstm",
                "n_features": 40,
                "sequence_length": 10,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
            }
        )
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(meta_content, encoding="utf-8")

        # Load and verify by testing prediction
        loaded = load_lstm_model(model_path, meta_path)

        # Test prediction
        x: NDArray[np.float64] = np.random.randn(3, 40).astype(np.float64)
        proba = loaded.predict_proba(x)
        assert proba.shape == (3, 2)
        # Probabilities should sum to ~1
        sums: NDArray[np.float64] = proba.sum(axis=1)
        assert bool(np.allclose(sums, 1.0))
