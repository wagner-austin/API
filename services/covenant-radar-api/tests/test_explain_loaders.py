"""Tests for model loaders for feature importance explanation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.backends.registry import default_registry
from numpy.typing import NDArray

from covenant_radar_api.worker._explain_loaders import (
    MLPModelConfig,
    load_gradient_model,
    load_model_for_backend,
)
from tests._explain_loaders_fixtures import (
    _BatchNorm1dFactory,
    _create_lightgbm_model,
    _create_lstm_model,
    _create_mlp_model,
    _create_xgboost_model,
    _LinearFactory,
    _ModuleProtocol,
    _ReLUFactory,
    _SequentialFactory,
    _SequentialProtocol,
    _TensorProtocol,
    _TorchSaveFn,
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


class TestLoadModelForBackendRegistryDelegation:
    """Non-torch backends are restored through the registry that owns them."""

    def test_every_registered_backend_is_reachable(self) -> None:
        """No registered backend is left without a loader.

        cleargbm, logreg and random_forest are valid BackendName values that
        this module used to reject with "No explain loader for backend", so
        /ml/explain answered a 500 for three of the seven backends it
        advertises. Permutation needs only predict_proba, so there was never
        a reason they could not be explained. Enumerating backends here is
        what let the list drift from the registry; it now delegates.
        """
        registered = set(default_registry().list_backends())

        assert registered == {"xgboost", "lightgbm", "cleargbm", "logreg", "random_forest"}

    def test_missing_file_is_reported_before_dispatch(self, tmp_path: Path) -> None:
        """A path with no model names the file, not the backend."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_model_for_backend("logreg", str(tmp_path / "absent.joblib"))


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
