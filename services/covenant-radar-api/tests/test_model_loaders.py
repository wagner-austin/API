"""Tests for model loading functions.

Tests cover metadata decoding, model architecture building, and inference
for MLP, LSTM, and LightGBM models.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.json_utils import JSONObject, JSONTypeError, dump_json_str

from covenant_radar_api.worker._model_loaders import (
    _decode_lightgbm_meta,
    _decode_lstm_meta,
    _decode_mlp_meta,
    _load_model_metadata,
    _reshape_flat_to_sequences,
    load_lightgbm_model,
    load_lstm_model,
    load_mlp_model,
)


class TestDecodeMlpMeta:
    """Tests for _decode_mlp_meta function."""

    def test_decode_mlp_meta_valid(self) -> None:
        """Decode valid MLP metadata successfully."""
        raw: JSONObject = {
            "backend": "mlp",
            "n_features": 50,
            "hidden_sizes": [128, 64],
            "dropout": 0.2,
        }
        result = _decode_mlp_meta(raw)

        assert result["backend"] == "mlp"
        assert result["n_features"] == 50
        assert result["hidden_sizes"] == [128, 64]
        assert result["dropout"] == 0.2

    def test_decode_mlp_meta_wrong_backend(self) -> None:
        """Raise error when backend is not 'mlp'."""
        raw: JSONObject = {
            "backend": "xgboost",
            "n_features": 50,
            "hidden_sizes": [128],
            "dropout": 0.1,
        }
        with pytest.raises(JSONTypeError) as exc_info:
            _decode_mlp_meta(raw)

        assert "Expected backend 'mlp'" in str(exc_info.value)

    def test_decode_mlp_meta_invalid_hidden_sizes_type(self) -> None:
        """Raise error when hidden_sizes contains non-integers."""
        raw: JSONObject = {
            "backend": "mlp",
            "n_features": 50,
            "hidden_sizes": [128, "bad"],
            "dropout": 0.1,
        }
        with pytest.raises(JSONTypeError) as exc_info:
            _decode_mlp_meta(raw)

        assert "hidden_sizes[1] must be an integer" in str(exc_info.value)

    def test_decode_mlp_meta_boolean_in_hidden_sizes(self) -> None:
        """Raise error when hidden_sizes contains boolean (which is technically int)."""
        raw: JSONObject = {
            "backend": "mlp",
            "n_features": 50,
            "hidden_sizes": [128, True],
            "dropout": 0.1,
        }
        with pytest.raises(JSONTypeError) as exc_info:
            _decode_mlp_meta(raw)

        assert "hidden_sizes[1] must be an integer" in str(exc_info.value)


class TestDecodeLstmMeta:
    """Tests for _decode_lstm_meta function."""

    def test_decode_lstm_meta_valid(self) -> None:
        """Decode valid LSTM metadata successfully."""
        raw: JSONObject = {
            "backend": "lstm",
            "n_features": 100,
            "sequence_length": 10,
            "hidden_size": 64,
            "num_layers": 2,
            "bidirectional": True,
            "dropout": 0.3,
        }
        result = _decode_lstm_meta(raw)

        assert result["backend"] == "lstm"
        assert result["n_features"] == 100
        assert result["sequence_length"] == 10
        assert result["hidden_size"] == 64
        assert result["num_layers"] == 2
        assert result["bidirectional"] is True
        assert result["dropout"] == 0.3

    def test_decode_lstm_meta_wrong_backend(self) -> None:
        """Raise error when backend is not 'lstm'."""
        raw: JSONObject = {
            "backend": "mlp",
            "n_features": 100,
            "sequence_length": 10,
            "hidden_size": 64,
            "num_layers": 2,
            "bidirectional": False,
            "dropout": 0.1,
        }
        with pytest.raises(JSONTypeError) as exc_info:
            _decode_lstm_meta(raw)

        assert "Expected backend 'lstm'" in str(exc_info.value)


class TestDecodeLightgbmMeta:
    """Tests for _decode_lightgbm_meta function."""

    def test_decode_lightgbm_meta_valid(self) -> None:
        """Decode valid LightGBM metadata successfully."""
        raw: JSONObject = {"backend": "lightgbm"}
        result = _decode_lightgbm_meta(raw)

        assert result["backend"] == "lightgbm"

    def test_decode_lightgbm_meta_wrong_backend(self) -> None:
        """Raise error when backend is not 'lightgbm'."""
        raw: JSONObject = {"backend": "xgboost"}
        with pytest.raises(JSONTypeError) as exc_info:
            _decode_lightgbm_meta(raw)

        assert "Expected backend 'lightgbm'" in str(exc_info.value)


class TestLoadModelMetadata:
    """Tests for _load_model_metadata function."""

    def test_load_mlp_metadata(self, tmp_path: Path) -> None:
        """Load MLP metadata from file."""
        meta_content = dump_json_str(
            {
                "backend": "mlp",
                "n_features": 25,
                "hidden_sizes": [64, 32],
                "dropout": 0.15,
            }
        )
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(meta_content, encoding="utf-8")

        result = _load_model_metadata(meta_path)

        assert result["backend"] == "mlp"

    def test_load_lstm_metadata(self, tmp_path: Path) -> None:
        """Load LSTM metadata from file."""
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

        result = _load_model_metadata(meta_path)

        assert result["backend"] == "lstm"

    def test_load_lightgbm_metadata(self, tmp_path: Path) -> None:
        """Load LightGBM metadata from file."""
        meta_content = dump_json_str({"backend": "lightgbm"})
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(meta_content, encoding="utf-8")

        result = _load_model_metadata(meta_path)

        assert result["backend"] == "lightgbm"

    def test_load_unknown_backend_raises(self, tmp_path: Path) -> None:
        """Raise error for unknown backend in metadata."""
        meta_content = dump_json_str({"backend": "unknown"})
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(meta_content, encoding="utf-8")

        with pytest.raises(JSONTypeError) as exc_info:
            _load_model_metadata(meta_path)

        assert "Unknown backend 'unknown'" in str(exc_info.value)


class TestReshapeFlatToSequences:
    """Tests for _reshape_flat_to_sequences function."""

    def test_reshape_exact_fit(self) -> None:
        """Reshape when features divide evenly into sequences."""
        x: NDArray[np.float64] = np.arange(20.0, dtype=np.float64).reshape(2, 10)
        result = _reshape_flat_to_sequences(x, sequence_length=5)

        assert result.shape == (2, 5, 2)

    def test_reshape_with_padding(self) -> None:
        """Reshape with padding when features don't divide evenly."""
        x: NDArray[np.float64] = np.arange(18.0, dtype=np.float64).reshape(2, 9)
        result = _reshape_flat_to_sequences(x, sequence_length=5)

        # 9 features with seq_len=5 means 2 features per step, padding to 10
        assert result.shape == (2, 5, 2)


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


class TestLoadLightgbmModel:
    """Tests for load_lightgbm_model function."""

    def test_load_lightgbm_model_file_not_found(self, tmp_path: Path) -> None:
        """Raise error when model file doesn't exist."""
        model_path = tmp_path / "nonexistent.txt"

        # LightGBM raises its own error type
        from lightgbm.basic import LightGBMError

        with pytest.raises(LightGBMError):
            load_lightgbm_model(model_path)


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
