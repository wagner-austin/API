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
    LightGBMModelMeta,
    LogRegModelMeta,
    LogRegPenalty,
    LogRegSolver,
    LSTMModelMeta,
    MLPModelMeta,
    ModelMeta,
    PredictorProtocol,
    RandomForestModelMeta,
)
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_float,
    require_int,
    require_list,
    require_str,
)
from platform_ml.torch_types import (
    TensorProtocol,
    TrainableModel,
    _import_torch,
)

# =============================================================================
# Metadata Decoding Functions
# =============================================================================


def _decode_mlp_meta(raw: JSONObject) -> MLPModelMeta:
    """Decode and validate MLP model metadata from JSON object.

    Args:
        raw: Parsed JSON object with MLP metadata fields.

    Returns:
        Validated MLPModelMeta TypedDict.

    Raises:
        JSONTypeError: If any field is missing or has wrong type.
    """
    backend = require_str(raw, "backend")
    if backend != "mlp":
        raise JSONTypeError(f"Expected backend 'mlp', got '{backend}'")

    n_features = require_int(raw, "n_features")
    dropout = require_float(raw, "dropout")

    # Parse hidden_sizes as list of ints
    hidden_sizes_raw = require_list(raw, "hidden_sizes")
    hidden_sizes: list[int] = []
    for i, val in enumerate(hidden_sizes_raw):
        if isinstance(val, bool) or not isinstance(val, int):
            raise JSONTypeError(f"hidden_sizes[{i}] must be an integer, got {type(val).__name__}")
        hidden_sizes.append(val)

    return {
        "backend": "mlp",
        "n_features": n_features,
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
    }


def _decode_lstm_meta(raw: JSONObject) -> LSTMModelMeta:
    """Decode and validate LSTM model metadata from JSON object.

    Args:
        raw: Parsed JSON object with LSTM metadata fields.

    Returns:
        Validated LSTMModelMeta TypedDict.

    Raises:
        JSONTypeError: If any field is missing or has wrong type.
    """
    backend = require_str(raw, "backend")
    if backend != "lstm":
        raise JSONTypeError(f"Expected backend 'lstm', got '{backend}'")

    return {
        "backend": "lstm",
        "n_features": require_int(raw, "n_features"),
        "sequence_length": require_int(raw, "sequence_length"),
        "hidden_size": require_int(raw, "hidden_size"),
        "num_layers": require_int(raw, "num_layers"),
        "bidirectional": require_bool(raw, "bidirectional"),
        "dropout": require_float(raw, "dropout"),
    }


def _decode_lightgbm_meta(raw: JSONObject) -> LightGBMModelMeta:
    """Decode and validate LightGBM model metadata from JSON object.

    Args:
        raw: Parsed JSON object with LightGBM metadata fields.

    Returns:
        Validated LightGBMModelMeta TypedDict.

    Raises:
        JSONTypeError: If backend field is missing or wrong.
    """
    backend = require_str(raw, "backend")
    if backend != "lightgbm":
        raise JSONTypeError(f"Expected backend 'lightgbm', got '{backend}'")

    return {"backend": "lightgbm"}


_LOGREG_PENALTIES: dict[str, LogRegPenalty] = {
    "l1": "l1",
    "l2": "l2",
    "elasticnet": "elasticnet",
    "none": "none",
}


def _parse_logreg_penalty(raw: str) -> LogRegPenalty:
    """Parse and validate logistic regression penalty type.

    Args:
        raw: Penalty string from metadata.

    Returns:
        Validated LogRegPenalty literal.

    Raises:
        JSONTypeError: If penalty is not a valid option.
    """
    penalty = _LOGREG_PENALTIES.get(raw)
    if penalty is not None:
        return penalty
    raise JSONTypeError(f"Invalid penalty '{raw}', expected one of: l1, l2, elasticnet, none")


_LOGREG_SOLVERS: dict[str, LogRegSolver] = {
    "lbfgs": "lbfgs",
    "liblinear": "liblinear",
    "newton-cg": "newton-cg",
    "newton-cholesky": "newton-cholesky",
    "sag": "sag",
    "saga": "saga",
}


def _parse_logreg_solver(raw: str) -> LogRegSolver:
    """Parse and validate logistic regression solver type.

    Args:
        raw: Solver string from metadata.

    Returns:
        Validated LogRegSolver literal.

    Raises:
        JSONTypeError: If solver is not a valid option.
    """
    solver = _LOGREG_SOLVERS.get(raw)
    if solver is not None:
        return solver
    raise JSONTypeError(
        f"Invalid solver '{raw}', expected one of: lbfgs, liblinear, newton-cg, "
        "newton-cholesky, sag, saga"
    )


def _decode_logreg_meta(raw: JSONObject) -> LogRegModelMeta:
    """Decode and validate Logistic Regression model metadata from JSON object.

    Args:
        raw: Parsed JSON object with LogReg metadata fields.

    Returns:
        Validated LogRegModelMeta TypedDict.

    Raises:
        JSONTypeError: If any field is missing or has wrong type.
    """
    backend = require_str(raw, "backend")
    if backend != "logreg":
        raise JSONTypeError(f"Expected backend 'logreg', got '{backend}'")

    n_features = require_int(raw, "n_features")
    penalty_raw = require_str(raw, "penalty")
    solver_raw = require_str(raw, "solver")

    return {
        "backend": "logreg",
        "n_features": n_features,
        "penalty": _parse_logreg_penalty(penalty_raw),
        "solver": _parse_logreg_solver(solver_raw),
    }


def _decode_random_forest_meta(raw: JSONObject) -> RandomForestModelMeta:
    """Decode and validate Random Forest model metadata from JSON object.

    Args:
        raw: Parsed JSON object with Random Forest metadata fields.

    Returns:
        Validated RandomForestModelMeta TypedDict.

    Raises:
        JSONTypeError: If any field is missing or has wrong type.
    """
    backend = require_str(raw, "backend")
    if backend != "random_forest":
        raise JSONTypeError(f"Expected backend 'random_forest', got '{backend}'")

    n_features = require_int(raw, "n_features")
    n_estimators = require_int(raw, "n_estimators")

    # max_depth can be None or int
    max_depth_raw = raw.get("max_depth")
    max_depth: int | None = None
    if max_depth_raw is not None:
        if isinstance(max_depth_raw, bool) or not isinstance(max_depth_raw, int):
            raise JSONTypeError("max_depth must be an integer or null")
        max_depth = max_depth_raw

    return {
        "backend": "random_forest",
        "n_features": n_features,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }


def _load_model_metadata(meta_path: Path) -> ModelMeta:
    """Load and decode model metadata from JSON file.

    Args:
        meta_path: Path to the metadata JSON file.

    Returns:
        Decoded ModelMeta (one of MLPModelMeta, LSTMModelMeta, LightGBMModelMeta,
        LogRegModelMeta, RandomForestModelMeta).

    Raises:
        FileNotFoundError: If metadata file doesn't exist.
        JSONTypeError: If metadata is invalid.
    """
    content = meta_path.read_text(encoding="utf-8")
    parsed = load_json_str(content)
    raw = narrow_json_to_dict(parsed)

    backend = require_str(raw, "backend")

    if backend == "mlp":
        return _decode_mlp_meta(raw)
    if backend == "lstm":
        return _decode_lstm_meta(raw)
    if backend == "lightgbm":
        return _decode_lightgbm_meta(raw)
    if backend == "logreg":
        return _decode_logreg_meta(raw)
    if backend == "random_forest":
        return _decode_random_forest_meta(raw)
    raise JSONTypeError(f"Unknown backend '{backend}' in metadata")


# =============================================================================
# MLP Model Loading - Constructor Protocols
# =============================================================================


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

    Mirrors the _build_model function in covenant_ml.backends.mlp.backend.

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

    # Load state dict
    torch_mod = _import_torch()
    state_dict: dict[str, TensorProtocol] = torch_mod.load(str(model_path))
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

    Mirrors the _LSTMClassifierWrapper in covenant_ml.backends.lstm.backend.
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

    Mirrors the _build_model function in covenant_ml.backends.lstm.backend.

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
    input_size = (n_features + sequence_length - 1) // sequence_length

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


def _reshape_flat_to_sequences(x: NDArray[np.float64], sequence_length: int) -> NDArray[np.float64]:
    """Reshape flat features to sequence format for LSTM.

    Args:
        x: Flat input array with shape (n_samples, n_features).
        sequence_length: Number of timesteps to create.

    Returns:
        Reshaped array with shape (n_samples, sequence_length, features_per_step).
    """
    n_samples = int(x.shape[0])
    n_features = int(x.shape[1])
    features_per_step = (n_features + sequence_length - 1) // sequence_length

    # Pad if necessary
    target_size = sequence_length * features_per_step
    if n_features < target_size:
        padding = target_size - n_features
        x = np.pad(x, ((0, 0), (0, padding)), mode="constant", constant_values=0.0)

    # Reshape to (n_samples, sequence_length, features_per_step)
    result: NDArray[np.float64] = x[:, : sequence_length * features_per_step].reshape(
        n_samples, sequence_length, features_per_step
    )
    return result


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
        x_seq = _reshape_flat_to_sequences(x, self._sequence_length)

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

    # Load state dict
    torch_mod = _import_torch()
    state_dict: dict[str, TensorProtocol] = torch_mod.load(str(model_path))
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
