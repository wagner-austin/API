"""Model loaders for feature importance explanation.

Provides model loading functions for all supported backends:
- XGBoost: Load from .ubj files
- LightGBM: Load from .txt files
- MLP: Load from .pt files (requires architecture config)
- LSTM: Load from .pt files (requires architecture config)

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
from covenant_ml.types import BackendName
from covenant_nn.backends.lstm.backend import load_lstm_for_inference
from covenant_nn.backends.mlp.backend import load_mlp_for_inference
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Common Predictor Protocol
# ---------------------------------------------------------------------------


class PredictorProtocol(Protocol):
    """Protocol for model with predict_proba method."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Class probabilities with shape (n_samples, n_classes).
        """
        ...


class GradientPredictorProtocol(Protocol):
    """Protocol for model with predict_proba and compute_gradients methods."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Class probabilities with shape (n_samples, n_classes).
        """
        ...

    def compute_gradients(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Compute gradients of output w.r.t. input features.

        Args:
            x: Input features with shape (n_samples, n_features).
            target_class: Class index for which to compute gradients.

        Returns:
            Gradients with shape (n_samples, n_features).
        """
        ...


# ---------------------------------------------------------------------------
# XGBoost Loading
# ---------------------------------------------------------------------------


class _XGBBoosterProtocol(Protocol):
    """Protocol for XGBoost Booster."""

    def save_model(self, fname: str) -> None: ...


class _XGBModelProtocol(Protocol):
    """Protocol for XGBoost classifier with load_model."""

    def load_model(self, fname: str) -> None: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def get_booster(self) -> _XGBBoosterProtocol: ...


class _XGBClassifierCtor(Protocol):
    """Protocol for XGBClassifier constructor."""

    def __call__(self) -> _XGBModelProtocol: ...


def _load_xgboost_model(model_path: str) -> PredictorProtocol:
    """Load XGBoost model from file.

    Args:
        model_path: Path to saved model file (.ubj format).

    Returns:
        Model implementing PredictorProtocol.
    """
    xgb_module = __import__("xgboost")
    classifier_ctor: _XGBClassifierCtor = xgb_module.XGBClassifier
    model = classifier_ctor()
    model.load_model(model_path)
    return model


# ---------------------------------------------------------------------------
# LightGBM Loading
# ---------------------------------------------------------------------------


class _LGBBoosterProtocol(Protocol):
    """Protocol for LightGBM Booster."""

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]: ...


class _LGBBoosterCtor(Protocol):
    """Protocol for LightGBM Booster constructor."""

    def __call__(self, *, model_file: str) -> _LGBBoosterProtocol: ...


class _LGBMBoosterWrapper:
    """Wrapper for LightGBM Booster providing predict_proba interface."""

    def __init__(self, booster: _LGBBoosterProtocol) -> None:
        """Initialize wrapper.

        Args:
            booster: LightGBM Booster instance.
        """
        self._booster = booster

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with [P(class=0), P(class=1)].
        """
        proba_positive: NDArray[np.float64] = self._booster.predict(x)
        n_samples = int(proba_positive.shape[0])

        result: NDArray[np.float64] = np.zeros((n_samples, 2), dtype=np.float64)
        for i in range(n_samples):
            p: float = float(proba_positive.flat[i])
            result[i, 0] = 1.0 - p
            result[i, 1] = p
        return result


def _load_lightgbm_model(model_path: str) -> PredictorProtocol:
    """Load LightGBM model from file.

    Args:
        model_path: Path to saved model file (.txt format).

    Returns:
        Model implementing PredictorProtocol.
    """
    lgb_module = __import__("lightgbm", fromlist=["Booster"])
    booster_ctor: _LGBBoosterCtor = lgb_module.Booster
    booster = booster_ctor(model_file=model_path)
    return _LGBMBoosterWrapper(booster)


# ---------------------------------------------------------------------------
# MLP and LSTM Model Loading
#
# The architectures, their checkpoint key layout and their prepared predictors
# all live in covenant_nn beside the code that trains and saves them. They are
# imported, never restated here: this module previously carried a second copy
# of both model stacks, and the copies drifted -- the LSTM one derived its
# state-dict prefixes from the wrapper's attribute names and so loaded no
# weights at all, which every unit test passed straight over.
# ---------------------------------------------------------------------------


class MLPModelConfig(TypedDict, total=True):
    """Configuration required to reconstruct MLP model architecture.

    Args:
        n_features: Number of input features.
        hidden_sizes: Tuple of hidden layer sizes.
        dropout: Dropout rate.
    """

    n_features: int
    hidden_sizes: tuple[int, ...]
    dropout: float


class LSTMModelConfig(TypedDict, total=True):
    """Configuration required to reconstruct LSTM model architecture.

    Args:
        n_features: Number of input features.
        hidden_size: LSTM hidden state size.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between layers.
        bidirectional: Whether the LSTM is bidirectional.
        sequence_length: Number of timesteps per sequence.
    """

    n_features: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool
    sequence_length: int


def _load_mlp_model(model_path: str, config: MLPModelConfig) -> GradientPredictorProtocol:
    """Load an MLP model from file.

    Args:
        model_path: Path to saved model file (.pt format).
        config: Model architecture configuration.

    Returns:
        Model implementing GradientPredictorProtocol.
    """
    return load_mlp_for_inference(
        path=model_path,
        n_features=config["n_features"],
        hidden_sizes=config["hidden_sizes"],
        dropout=config["dropout"],
    )


def _load_lstm_model(model_path: str, config: LSTMModelConfig) -> GradientPredictorProtocol:
    """Load an LSTM model from file.

    Args:
        model_path: Path to saved model file (.pt format).
        config: Model architecture configuration.

    Returns:
        Model implementing GradientPredictorProtocol.
    """
    return load_lstm_for_inference(
        path=model_path,
        n_features=config["n_features"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
        sequence_length=config["sequence_length"],
    )


# ---------------------------------------------------------------------------
# Unified Model Loading Entry Point
# ---------------------------------------------------------------------------


def load_model_for_backend(
    backend: BackendName,
    model_path: str,
    mlp_config: MLPModelConfig | None = None,
    lstm_config: LSTMModelConfig | None = None,
) -> PredictorProtocol:
    """Load model based on backend type.

    Args:
        backend: Backend name (xgboost, lightgbm, mlp, lstm).
        model_path: Path to saved model file.
        mlp_config: MLP architecture config (required if backend is 'mlp').
        lstm_config: LSTM architecture config (required if backend is 'lstm').

    Returns:
        Model implementing PredictorProtocol.

    Raises:
        ValueError: If required config is missing for MLP/LSTM backend, or if
            the backend has no loader here.
        FileNotFoundError: If model file doesn't exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if backend == "xgboost":
        return _load_xgboost_model(model_path)
    if backend == "lightgbm":
        return _load_lightgbm_model(model_path)
    if backend == "mlp":
        if mlp_config is None:
            raise ValueError("mlp_config is required for MLP backend")
        return _load_mlp_model(model_path, mlp_config)
    if backend == "lstm":
        if lstm_config is None:
            raise ValueError("lstm_config is required for LSTM backend")
        return _load_lstm_model(model_path, lstm_config)
    # BackendName also covers cleargbm, logreg and random_forest, which have no
    # loader here. Falling through to the LSTM branch reported them as a
    # missing lstm_config, which named the wrong problem entirely.
    raise ValueError(f"No explain loader for backend: {backend}")


def load_gradient_model(
    backend: str,
    model_path: str,
    mlp_config: MLPModelConfig | None = None,
    lstm_config: LSTMModelConfig | None = None,
) -> GradientPredictorProtocol:
    """Load MLP or LSTM model with gradient support.

    This function is for backends that support compute_gradients() (MLP, LSTM).
    Use load_model_for_backend for XGBoost/LightGBM.

    Args:
        backend: Backend name ('mlp' or 'lstm').
        model_path: Path to saved model file.
        mlp_config: MLP architecture config (required if backend is 'mlp').
        lstm_config: LSTM architecture config (required if backend is 'lstm').

    Returns:
        Model implementing GradientPredictorProtocol.

    Raises:
        ValueError: If backend is not 'mlp' or 'lstm', or required config missing.
        FileNotFoundError: If model file doesn't exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if backend == "mlp":
        if mlp_config is None:
            raise ValueError("mlp_config is required for MLP backend")
        return _load_mlp_model(model_path, mlp_config)
    if backend == "lstm":
        if lstm_config is None:
            raise ValueError("lstm_config is required for LSTM backend")
        return _load_lstm_model(model_path, lstm_config)
    raise ValueError(f"Backend '{backend}' does not support gradients. Use 'mlp' or 'lstm'.")


__all__ = [
    "GradientPredictorProtocol",
    "LSTMModelConfig",
    "MLPModelConfig",
    "PredictorProtocol",
    "load_gradient_model",
    "load_model_for_backend",
]
