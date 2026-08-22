"""Model loading functions for MLP, LSTM, LightGBM, LogReg, and RandomForest inference.

This module provides functions to load trained models from disk for inference.
Models are loaded using saved architecture metadata (for neural networks) or
directly from self-describing formats (for tree-based and sklearn models).

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from covenant_ml.types import (
    LSTMModelMeta,
    MLPModelMeta,
    PredictorProtocol,
)
from covenant_nn.backends.lstm.sequences import (
    compute_features_per_step,
    reshape_flat_to_pseudo_sequences,
)
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONTypeError,
)
from platform_ml.torch_types import (
    TensorProtocol,
    TrainableModel,
    _import_torch,
)

# =============================================================================
# Metadata Decoding Functions
# =============================================================================
from covenant_radar_api.worker._model_meta import (
    _decode_lightgbm_meta,
    _decode_logreg_meta,
    _decode_lstm_meta,
    _decode_mlp_meta,
    _decode_random_forest_meta,
    _load_model_metadata,
)


class _NNLinearCtor(Protocol):
    """Protocol for nn.Linear constructor."""

    def __call__(self, in_features: int, out_features: int) -> TrainableModel: ...


class _NNBatchNorm1dCtor(Protocol):
    """Protocol for nn.BatchNorm1d constructor."""

    def __call__(self, num_features: int) -> TrainableModel: ...


class _NNReLUCtor(Protocol):
    """Protocol for nn.ReLU constructor."""

    def __call__(self) -> TrainableModel: ...


class _NNDropoutCtor(Protocol):
    """Protocol for nn.Dropout constructor."""

    def __call__(self, p: float) -> TrainableModel: ...


class _NNSequentialCtor(Protocol):
    """Protocol for nn.Sequential constructor."""

    def __call__(self, *args: TrainableModel) -> TrainableModel: ...


class _SoftmaxCtor(Protocol):
    """Protocol for nn.Softmax constructor."""

    def __call__(self, dim: int) -> TrainableModel: ...


def _build_mlp_model(
    n_features: int,
    hidden_sizes: list[int],
    dropout: float,
    device: str,
) -> TrainableModel:
    """Build MLP model architecture for loading state dict.

    Mirrors the _build_model function in covenant_nn.backends.mlp.backend.

    Args:
        n_features: Number of input features.
        hidden_sizes: List of hidden layer sizes.
        dropout: Dropout rate for regularization.
        device: Device to place model on ("cpu" or "cuda").

    Returns:
        Constructed MLP model ready for state dict loading.
    """
    nn_mod = __import__(
        "torch.nn",
        fromlist=["Linear", "BatchNorm1d", "ReLU", "Dropout", "Sequential"],
    )

    # Get constructors with explicit Protocol annotations to avoid Any
    linear: _NNLinearCtor = nn_mod.Linear
    bn: _NNBatchNorm1dCtor = nn_mod.BatchNorm1d
    relu: _NNReLUCtor = nn_mod.ReLU
    drop: _NNDropoutCtor = nn_mod.Dropout
    sequential: _NNSequentialCtor = nn_mod.Sequential

    parts: list[TrainableModel] = []
    in_f = n_features
    for width in hidden_sizes:
        parts.append(linear(in_f, width))
        parts.append(bn(width))
        parts.append(relu())
        if dropout > 0.0:
            parts.append(drop(dropout))
        in_f = width
    parts.append(linear(in_f, 2))

    model: TrainableModel = sequential(*parts)
    return model.to(device)


class _MLPPreparedForInference:
    """Prepared MLP model wrapper for inference.

    Wraps a PyTorch Sequential model and provides predict_proba method
    implementing PredictorProtocol.
    """

    def __init__(self, model: TrainableModel) -> None:
        """Initialize with trained model.

        Args:
            model: PyTorch model with loaded weights in eval mode.
        """
        self._model = model

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return class probabilities for input samples.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Class probabilities with shape (n_samples, n_classes).
        """
        torch_mod = _import_torch()
        nn_mod = __import__("torch.nn", fromlist=["Softmax"])
        softmax_ctor: _SoftmaxCtor = nn_mod.Softmax

        with torch_mod.no_grad():
            xt: TensorProtocol = torch_mod.tensor(x, dtype=torch_mod.float32)
            logits: TensorProtocol = self._model(xt)
            sm: TrainableModel = softmax_ctor(dim=1)
            proba: TensorProtocol = sm(logits)
            proba_cpu: TensorProtocol = proba.cpu()
            result: NDArray[np.float64] = proba_cpu.numpy().astype(np.float64)
            return result


def load_mlp_model(model_path: Path, meta_path: Path) -> PredictorProtocol:
    """Load MLP model from state dict using metadata.

    Args:
        model_path: Path to .pt state dict file.
        meta_path: Path to metadata JSON file.

    Returns:
        Prepared MLP model implementing PredictorProtocol.

    Raises:
        FileNotFoundError: If model or metadata file missing.
        JSONTypeError: If metadata is invalid.
    """
    # Load and validate metadata
    meta = _load_model_metadata(meta_path)
    if meta["backend"] != "mlp":
        raise JSONTypeError(f"Expected MLP metadata, got {meta['backend']}")
    mlp_meta: MLPModelMeta = meta  # type narrowing via backend check

    # Build model architecture
    model = _build_mlp_model(
        n_features=mlp_meta["n_features"],
        hidden_sizes=mlp_meta["hidden_sizes"],
        dropout=mlp_meta["dropout"],
        device="cpu",  # Load on CPU, move to GPU later if needed
    )

    # Load state dict. weights_only=True keeps torch on the safe unpickling
    # path: model_path originates from a request body.
    torch_mod = _import_torch()
    state_dict: dict[str, TensorProtocol] = torch_mod.load(str(model_path), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return _MLPPreparedForInference(model)


# =============================================================================
# LSTM Model Loading
# =============================================================================


class _LSTMLayerProto(Protocol):
    """Protocol for nn.LSTM layer."""

    def __call__(
        self, x: TensorProtocol
    ) -> tuple[TensorProtocol, tuple[TensorProtocol, TensorProtocol]]: ...
    def eval(self) -> _LSTMLayerProto: ...
    def state_dict(self) -> dict[str, TensorProtocol]: ...
    def load_state_dict(self, state_dict: dict[str, TensorProtocol]) -> None: ...
    def to(self, device: str) -> _LSTMLayerProto: ...


class _LinearLayerProto(Protocol):
    """Protocol for nn.Linear layer."""

    def __call__(self, x: TensorProtocol) -> TensorProtocol: ...
    def eval(self) -> _LinearLayerProto: ...
    def state_dict(self) -> dict[str, TensorProtocol]: ...
    def load_state_dict(self, state_dict: dict[str, TensorProtocol]) -> None: ...
    def to(self, device: str) -> _LinearLayerProto: ...


class _LSTMClassifierWrapper:
    """LSTM classifier combining LSTM and linear layers.

    Mirrors the _LSTMClassifierWrapper in covenant_nn.backends.lstm.backend.
    """

    def __init__(
        self,
        lstm: _LSTMLayerProto,
        fc: _LinearLayerProto,
    ) -> None:
        """Initialize with LSTM and fully-connected layers.

        Args:
            lstm: LSTM layer for sequence processing.
            fc: Linear layer for classification.
        """
        self._lstm = lstm
        self._fc = fc

    def __call__(self, x: TensorProtocol) -> TensorProtocol:
        """Forward pass through LSTM and classifier."""
        lstm_out, _ = self._lstm(x)
        # Take last timestep: lstm_out shape is (batch, seq_len, hidden)
        # Select last timestep along dim 1
        seq_len = int(lstm_out.shape[1])
        last_hidden: TensorProtocol = lstm_out.select(1, seq_len - 1)
        logits: TensorProtocol = self._fc(last_hidden)
        return logits

    def eval(self) -> _LSTMClassifierWrapper:
        """Set model to evaluation mode."""
        self._lstm.eval()
        self._fc.eval()
        return self

    def state_dict(self) -> dict[str, TensorProtocol]:
        """Get combined state dict with prefixed keys."""
        result: dict[str, TensorProtocol] = {}
        for key, val in self._lstm.state_dict().items():
            result[f"_lstm.{key}"] = val
        for key, val in self._fc.state_dict().items():
            result[f"_fc.{key}"] = val
        return result

    def load_state_dict(self, state_dict: dict[str, TensorProtocol]) -> None:
        """Load state dict into LSTM and FC layers."""
        # Split state dict by prefix
        lstm_state: dict[str, TensorProtocol] = {}
        fc_state: dict[str, TensorProtocol] = {}

        for key, val in state_dict.items():
            if key.startswith("_lstm."):
                new_key = key[6:]  # Remove "_lstm." prefix
                lstm_state[new_key] = val
            elif key.startswith("_fc."):
                new_key = key[4:]  # Remove "_fc." prefix
                fc_state[new_key] = val

        self._lstm.load_state_dict(lstm_state)
        self._fc.load_state_dict(fc_state)

    def to(self, device: str) -> _LSTMClassifierWrapper:
        """Move model to device."""
        self._lstm = self._lstm.to(device)
        self._fc = self._fc.to(device)
        return self


def _build_lstm_model(
    n_features: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    device: str,
) -> tuple[_LSTMClassifierWrapper, int]:
    """Build LSTM model architecture for loading state dict.

    Mirrors the _build_model function in covenant_nn.backends.lstm.backend.

    Args:
        n_features: Total number of input features.
        sequence_length: Number of time steps in sequence.
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between LSTM layers.
        bidirectional: Whether to use bidirectional LSTM.
        device: Device to place model on.

    Returns:
        Tuple of (model, input_size) where input_size is features per timestep.
    """
    nn_mod = __import__("torch.nn", fromlist=["LSTM", "Linear"])

    # Calculate input size (features per timestep)
    input_size = compute_features_per_step(n_features, sequence_length)

    # Create LSTM layer
    lstm: _LSTMLayerProto = nn_mod.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
        bidirectional=bidirectional,
    )

    # Create classifier head
    num_directions = 2 if bidirectional else 1
    lstm_out_size = hidden_size * num_directions
    fc: _LinearLayerProto = nn_mod.Linear(lstm_out_size, 2)

    # Build wrapper
    model = _LSTMClassifierWrapper(lstm, fc)

    return model.to(device), input_size


class _LSTMPreparedForInference:
    """Prepared LSTM model wrapper for inference.

    Wraps an LSTM classifier and provides predict_proba method
    implementing PredictorProtocol.
    """

    def __init__(self, model: _LSTMClassifierWrapper, sequence_length: int) -> None:
        """Initialize with trained model.

        Args:
            model: LSTM classifier with loaded weights in eval mode.
            sequence_length: Number of timesteps for reshaping input.
        """
        self._model = model
        self._sequence_length = sequence_length

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return class probabilities for input samples.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Class probabilities with shape (n_samples, n_classes).
        """
        torch_mod = _import_torch()
        nn_mod = __import__("torch.nn", fromlist=["Softmax"])
        softmax_ctor: _SoftmaxCtor = nn_mod.Softmax

        # Reshape to sequences
        x_seq = reshape_flat_to_pseudo_sequences(x, self._sequence_length)

        with torch_mod.no_grad():
            xt: TensorProtocol = torch_mod.tensor(x_seq, dtype=torch_mod.float32)
            logits: TensorProtocol = self._model(xt)
            sm: TrainableModel = softmax_ctor(dim=1)
            proba: TensorProtocol = sm(logits)
            proba_cpu: TensorProtocol = proba.cpu()
            result: NDArray[np.float64] = proba_cpu.numpy().astype(np.float64)
            return result


def load_lstm_model(model_path: Path, meta_path: Path) -> PredictorProtocol:
    """Load LSTM model from state dict using metadata.

    Args:
        model_path: Path to .pt state dict file.
        meta_path: Path to metadata JSON file.

    Returns:
        Prepared LSTM model implementing PredictorProtocol.

    Raises:
        FileNotFoundError: If model or metadata file missing.
        JSONTypeError: If metadata is invalid.
    """
    # Load and validate metadata
    meta = _load_model_metadata(meta_path)
    if meta["backend"] != "lstm":
        raise JSONTypeError(f"Expected LSTM metadata, got {meta['backend']}")
    lstm_meta: LSTMModelMeta = meta  # type narrowing via backend check

    # Build model architecture
    model, _ = _build_lstm_model(
        n_features=lstm_meta["n_features"],
        sequence_length=lstm_meta["sequence_length"],
        hidden_size=lstm_meta["hidden_size"],
        num_layers=lstm_meta["num_layers"],
        dropout=lstm_meta["dropout"],
        bidirectional=lstm_meta["bidirectional"],
        device="cpu",
    )

    # Load state dict. weights_only=True keeps torch on the safe unpickling
    # path: model_path originates from a request body.
    torch_mod = _import_torch()
    state_dict: dict[str, TensorProtocol] = torch_mod.load(str(model_path), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return _LSTMPreparedForInference(model, lstm_meta["sequence_length"])


# =============================================================================
# LightGBM Model Loading
# =============================================================================


def load_lightgbm_model(model_path: Path) -> PredictorProtocol:
    """Load LightGBM model from .txt file.

    LightGBM's text format is self-describing, so no metadata is needed.

    Args:
        model_path: Path to the saved model file (.txt format).

    Returns:
        Prepared LightGBM model implementing PredictorProtocol.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    from covenant_ml.backends.lightgbm import create_lightgbm_backend

    backend = create_lightgbm_backend()
    return backend.load(path=str(model_path))


# =============================================================================
# LogReg Model Loading
# =============================================================================


def load_logreg_model(model_path: Path) -> PredictorProtocol:
    """Load Logistic Regression model from .joblib file.

    LogReg models are saved using joblib serialization. The model file
    contains the full sklearn LogisticRegression estimator.

    Args:
        model_path: Path to the saved model file (.joblib format).

    Returns:
        Prepared LogReg model implementing PredictorProtocol.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    from covenant_ml.backends.logreg import create_logreg_backend

    backend = create_logreg_backend()
    return backend.load(path=str(model_path))


# =============================================================================
# Random Forest Model Loading
# =============================================================================


def load_random_forest_model(model_path: Path) -> PredictorProtocol:
    """Load Random Forest model from .joblib file.

    Random Forest models are saved using joblib serialization. The model file
    contains the full sklearn RandomForestClassifier estimator.

    Args:
        model_path: Path to the saved model file (.joblib format).

    Returns:
        Prepared Random Forest model implementing PredictorProtocol.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    from covenant_ml.backends.random_forest import create_random_forest_backend

    backend = create_random_forest_backend()
    return backend.load(path=str(model_path))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "_decode_lightgbm_meta",
    "_decode_logreg_meta",
    "_decode_lstm_meta",
    "_decode_mlp_meta",
    "_decode_random_forest_meta",
    "_load_model_metadata",
    "load_lightgbm_model",
    "load_logreg_model",
    "load_lstm_model",
    "load_mlp_model",
    "load_random_forest_model",
]
