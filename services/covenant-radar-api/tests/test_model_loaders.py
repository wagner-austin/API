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
from platform_core.json_utils import JSONObject, JSONTypeError, dump_json_str

from covenant_radar_api.worker._model_loaders import (
    _decode_lightgbm_meta,
    _decode_logreg_meta,
    _decode_lstm_meta,
    _decode_mlp_meta,
    _decode_random_forest_meta,
    _load_model_metadata,
    load_lightgbm_model,
    load_logreg_model,
    load_random_forest_model,
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


class TestDecodeLogregMeta:
    """Tests for _decode_logreg_meta function."""

    def test_decode_logreg_meta_valid(self) -> None:
        """Decode valid LogReg metadata successfully."""
        raw: JSONObject = {
            "backend": "logreg",
            "n_features": 25,
            "penalty": "l2",
            "solver": "lbfgs",
        }
        result = _decode_logreg_meta(raw)

        assert result["backend"] == "logreg"
        assert result["n_features"] == 25
        assert result["penalty"] == "l2"
        assert result["solver"] == "lbfgs"

    def test_decode_logreg_meta_all_penalties(self) -> None:
        """Decode LogReg metadata with all valid penalties."""
        for penalty in ("l1", "l2", "elasticnet", "none"):
            raw: JSONObject = {
                "backend": "logreg",
                "n_features": 10,
                "penalty": penalty,
                "solver": "saga",
            }
            result = _decode_logreg_meta(raw)
            assert result["penalty"] == penalty

    def test_decode_logreg_meta_all_solvers(self) -> None:
        """Decode LogReg metadata with all valid solvers."""
        for solver in ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"):
            raw: JSONObject = {
                "backend": "logreg",
                "n_features": 10,
                "penalty": "l2",
                "solver": solver,
            }
            result = _decode_logreg_meta(raw)
            assert result["solver"] == solver

    def test_decode_logreg_meta_wrong_backend(self) -> None:
        """Raise error when backend is not 'logreg'."""
        raw: JSONObject = {
            "backend": "mlp",
            "n_features": 25,
            "penalty": "l2",
            "solver": "lbfgs",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            _decode_logreg_meta(raw)

        assert "Expected backend 'logreg'" in str(exc_info.value)

    def test_decode_logreg_meta_invalid_penalty(self) -> None:
        """Raise error when penalty is invalid."""
        raw: JSONObject = {
            "backend": "logreg",
            "n_features": 25,
            "penalty": "invalid",
            "solver": "lbfgs",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            _decode_logreg_meta(raw)

        assert "Invalid penalty" in str(exc_info.value)

    def test_decode_logreg_meta_invalid_solver(self) -> None:
        """Raise error when solver is invalid."""
        raw: JSONObject = {
            "backend": "logreg",
            "n_features": 25,
            "penalty": "l2",
            "solver": "invalid",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            _decode_logreg_meta(raw)

        assert "Invalid solver" in str(exc_info.value)


class TestDecodeRandomForestMeta:
    """Tests for _decode_random_forest_meta function."""

    def test_decode_random_forest_meta_valid(self) -> None:
        """Decode valid Random Forest metadata successfully."""
        raw: JSONObject = {
            "backend": "random_forest",
            "n_features": 50,
            "n_estimators": 100,
            "max_depth": 10,
        }
        result = _decode_random_forest_meta(raw)

        assert result["backend"] == "random_forest"
        assert result["n_features"] == 50
        assert result["n_estimators"] == 100
        assert result["max_depth"] == 10

    def test_decode_random_forest_meta_null_max_depth(self) -> None:
        """Decode Random Forest metadata with null max_depth."""
        raw: JSONObject = {
            "backend": "random_forest",
            "n_features": 50,
            "n_estimators": 100,
            "max_depth": None,
        }
        result = _decode_random_forest_meta(raw)

        assert result["backend"] == "random_forest"
        assert result["max_depth"] is None

    def test_decode_random_forest_meta_wrong_backend(self) -> None:
        """Raise error when backend is not 'random_forest'."""
        raw: JSONObject = {
            "backend": "xgboost",
            "n_features": 50,
            "n_estimators": 100,
            "max_depth": 10,
        }
        with pytest.raises(JSONTypeError) as exc_info:
            _decode_random_forest_meta(raw)

        assert "Expected backend 'random_forest'" in str(exc_info.value)

    def test_decode_random_forest_meta_invalid_max_depth_type(self) -> None:
        """Raise error when max_depth is not int or null."""
        raw: JSONObject = {
            "backend": "random_forest",
            "n_features": 50,
            "n_estimators": 100,
            "max_depth": "invalid",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            _decode_random_forest_meta(raw)

        assert "max_depth must be an integer or null" in str(exc_info.value)

    def test_decode_random_forest_meta_boolean_max_depth(self) -> None:
        """Raise error when max_depth is boolean (which is technically int)."""
        raw: JSONObject = {
            "backend": "random_forest",
            "n_features": 50,
            "n_estimators": 100,
            "max_depth": True,
        }
        with pytest.raises(JSONTypeError) as exc_info:
            _decode_random_forest_meta(raw)

        assert "max_depth must be an integer or null" in str(exc_info.value)


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

    def test_load_logreg_metadata(self, tmp_path: Path) -> None:
        """Load LogReg metadata from file."""
        meta_content = dump_json_str(
            {
                "backend": "logreg",
                "n_features": 25,
                "penalty": "l2",
                "solver": "lbfgs",
            }
        )
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(meta_content, encoding="utf-8")

        result = _load_model_metadata(meta_path)

        assert result["backend"] == "logreg"

    def test_load_random_forest_metadata(self, tmp_path: Path) -> None:
        """Load Random Forest metadata from file."""
        meta_content = dump_json_str(
            {
                "backend": "random_forest",
                "n_features": 50,
                "n_estimators": 100,
                "max_depth": 10,
            }
        )
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(meta_content, encoding="utf-8")

        result = _load_model_metadata(meta_path)

        assert result["backend"] == "random_forest"

    def test_load_unknown_backend_raises(self, tmp_path: Path) -> None:
        """Raise error for unknown backend in metadata."""
        meta_content = dump_json_str({"backend": "unknown"})
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(meta_content, encoding="utf-8")

        with pytest.raises(JSONTypeError) as exc_info:
            _load_model_metadata(meta_path)

        assert "Unknown backend 'unknown'" in str(exc_info.value)


class TestLoadLightgbmModel:
    """Tests for load_lightgbm_model function."""

    def test_load_lightgbm_model_file_not_found(self, tmp_path: Path) -> None:
        """Raise error when model file doesn't exist."""
        model_path = tmp_path / "nonexistent.txt"

        # LightGBM raises its own error type
        from lightgbm.basic import LightGBMError

        with pytest.raises(LightGBMError):
            load_lightgbm_model(model_path)


class TestLoadLogregModel:
    """Tests for load_logreg_model function."""

    def test_load_logreg_model_file_not_found(self, tmp_path: Path) -> None:
        """Raise error when model file doesn't exist."""
        model_path = tmp_path / "nonexistent.joblib"

        with pytest.raises(FileNotFoundError):
            load_logreg_model(model_path)


class TestLoadLogregModelFull:
    """Tests for full LogReg model loading path."""

    def test_load_logreg_model_full_path(self, tmp_path: Path) -> None:
        """Load LogReg model from saved joblib file."""
        from covenant_ml.backends.logreg.backend import (
            _create_logreg_model,
            _get_joblib_imports,
        )

        # Create and fit a model directly (not via backend.train)
        x_data: NDArray[np.float64] = np.random.randn(100, 10).astype(np.float64)
        y_data: NDArray[np.int64] = np.random.randint(0, 2, size=100).astype(np.int64)

        model = _create_logreg_model(
            penalty="l2",
            inverse_reg_strength=1.0,
            solver="lbfgs",
            max_iter=200,
            tol=1e-4,
            random_state=42,
            class_weight=None,
            l1_ratio=None,
            n_jobs=-1,
        )
        model.fit(x_data, y_data)

        # Save model
        model_path = tmp_path / "logreg_model.joblib"
        dump_fn, _ = _get_joblib_imports()
        dump_fn(model, str(model_path))

        # Load and verify by testing prediction
        loaded = load_logreg_model(model_path)

        # Test prediction
        x: NDArray[np.float64] = np.random.randn(3, 10).astype(np.float64)
        proba = loaded.predict_proba(x)
        assert proba.shape == (3, 2)
        # Probabilities should sum to ~1
        sums: NDArray[np.float64] = proba.sum(axis=1)
        assert bool(np.allclose(sums, 1.0))


class TestLoadRandomForestModel:
    """Tests for load_random_forest_model function."""

    def test_load_random_forest_model_file_not_found(self, tmp_path: Path) -> None:
        """Raise error when model file doesn't exist."""
        model_path = tmp_path / "nonexistent.joblib"

        with pytest.raises(FileNotFoundError):
            load_random_forest_model(model_path)


class TestLoadRandomForestModelFull:
    """Tests for full Random Forest model loading path."""

    def test_load_random_forest_model_full_path(self, tmp_path: Path) -> None:
        """Load Random Forest model from saved joblib file."""
        from covenant_ml.backends.random_forest.backend import _get_sklearn_imports

        # Get sklearn imports via typed accessor
        rf_ctor, dump_fn, _ = _get_sklearn_imports()

        # Create and fit a model directly (not via backend.train)
        x_data: NDArray[np.float64] = np.random.randn(100, 10).astype(np.float64)
        y_data: NDArray[np.int64] = np.random.randint(0, 2, size=100).astype(np.int64)

        model = rf_ctor(
            n_estimators=10,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight=None,
            n_jobs=-1,
            random_state=42,
            oob_score=False,
        )
        model.fit(x_data, y_data)

        # Save model
        model_path = tmp_path / "rf_model.joblib"
        dump_fn(model, str(model_path))

        # Load and verify by testing prediction
        loaded = load_random_forest_model(model_path)

        # Test prediction
        x: NDArray[np.float64] = np.random.randn(3, 10).astype(np.float64)
        proba = loaded.predict_proba(x)
        assert proba.shape == (3, 2)
        # Probabilities should sum to ~1
        sums: NDArray[np.float64] = proba.sum(axis=1)
        assert bool(np.allclose(sums, 1.0))
