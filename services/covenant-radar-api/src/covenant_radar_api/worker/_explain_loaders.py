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
from types import TracebackType
from typing import Protocol, TypedDict

import numpy as np
from covenant_ml.types import BackendName
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
# PyTorch Protocol Definitions
# ---------------------------------------------------------------------------


class _DTypeProtocol(Protocol):
    """Protocol for PyTorch dtype (torch.float32, etc)."""

    @property
    def is_floating_point(self) -> bool: ...


class _HiddenStateProtocol(Protocol):
    """Protocol for LSTM hidden state tuple (h_n, c_n)."""

    def __len__(self) -> int: ...


class _TensorProtocol(Protocol):
    """Protocol for PyTorch tensor."""

    @property
    def grad(self) -> _TensorProtocol | None: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def cpu(self) -> _TensorProtocol: ...
    def numpy(self) -> NDArray[np.float64]: ...
    def select(self, dim: int, index: int) -> _TensorProtocol: ...
    def sum(self) -> _TensorProtocol: ...
    def backward(self) -> None: ...


class _TrainableModel(Protocol):
    """Protocol for PyTorch trainable model."""

    def __call__(self, x: _TensorProtocol) -> _TensorProtocol: ...
    def eval(self) -> _TrainableModel: ...
    def state_dict(self) -> dict[str, _TensorProtocol]: ...
    def load_state_dict(self, state_dict: dict[str, _TensorProtocol]) -> None: ...
    def to(self, device: str) -> _TrainableModel: ...


class _NoGradContext(Protocol):
    """Protocol for torch.no_grad context manager."""

    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...


class _NoGradFactory(Protocol):
    """Protocol for torch.no_grad factory."""

    def __call__(self) -> _NoGradContext: ...


class _EnableGradContext(Protocol):
    """Protocol for torch.enable_grad context manager."""

    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...


class _EnableGradFactory(Protocol):
    """Protocol for torch.enable_grad factory."""

    def __call__(self) -> _EnableGradContext: ...


class _SoftmaxFn(Protocol):
    """Protocol for softmax function (already configured with dim)."""

    def __call__(self, x: _TensorProtocol) -> _TensorProtocol: ...


class _SoftmaxCtor(Protocol):
    """Protocol for nn.Softmax constructor."""

    def __call__(self, dim: int) -> _SoftmaxFn: ...


class _TensorCtor(Protocol):
    """Protocol for torch.tensor constructor."""

    def __call__(
        self,
        data: NDArray[np.float32],
        dtype: _DTypeProtocol,
        requires_grad: bool = ...,
    ) -> _TensorProtocol: ...


class _TorchLoadFn(Protocol):
    """Protocol for torch.load function."""

    def __call__(self, f: str, weights_only: bool = ...) -> dict[str, _TensorProtocol]: ...


class _NNLinearCtor(Protocol):
    """Protocol for nn.Linear constructor."""

    def __call__(self, in_features: int, out_features: int) -> _TrainableModel: ...


class _NNBatchNorm1dCtor(Protocol):
    """Protocol for nn.BatchNorm1d constructor."""

    def __call__(self, num_features: int) -> _TrainableModel: ...


class _NNReLUCtor(Protocol):
    """Protocol for nn.ReLU constructor."""

    def __call__(self) -> _TrainableModel: ...


class _NNDropoutCtor(Protocol):
    """Protocol for nn.Dropout constructor."""

    def __call__(self, p: float) -> _TrainableModel: ...


class _NNSequentialCtor(Protocol):
    """Protocol for nn.Sequential constructor."""

    def __call__(self, *modules: _TrainableModel) -> _TrainableModel: ...


# ---------------------------------------------------------------------------
# MLP Model Config and Loading
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


def _build_mlp_model(
    n_in: int,
    hidden: tuple[int, ...],
    dropout: float,
) -> _TrainableModel:
    """Build MLP model architecture (CPU only for inference).

    Args:
        n_in: Number of input features.
        hidden: Hidden layer sizes.
        dropout: Dropout rate.

    Returns:
        MLP model (on CPU).
    """
    nn_mod = __import__(
        "torch.nn",
        fromlist=["Linear", "BatchNorm1d", "ReLU", "Dropout", "Sequential"],
    )
    linear: _NNLinearCtor = nn_mod.Linear
    bn: _NNBatchNorm1dCtor = nn_mod.BatchNorm1d
    relu: _NNReLUCtor = nn_mod.ReLU
    drop: _NNDropoutCtor = nn_mod.Dropout
    sequential: _NNSequentialCtor = nn_mod.Sequential

    parts: list[_TrainableModel] = []
    in_f = int(n_in)
    for width in hidden:
        parts.append(linear(in_f, int(width)))
        parts.append(bn(int(width)))
        parts.append(relu())
        if dropout > 0.0:
            parts.append(drop(dropout))
        in_f = int(width)
    parts.append(linear(in_f, 2))
    return sequential(*parts)


class _MLPPrepared:
    """Prepared MLP model implementing PredictorProtocol with gradient support."""

    def __init__(self, model: _TrainableModel) -> None:
        """Initialize with PyTorch model.

        Args:
            model: PyTorch Sequential model.
        """
        self._model = model

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return class probabilities for input samples.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Class probabilities with shape (n_samples, n_classes).
        """
        torch_mod = __import__("torch")
        nn_mod = __import__("torch.nn", fromlist=["Softmax"])
        no_grad: _NoGradFactory = torch_mod.no_grad
        tensor: _TensorCtor = torch_mod.tensor
        fp32_dtype: _DTypeProtocol = torch_mod.float32
        softmax_ctor: _SoftmaxCtor = nn_mod.Softmax

        softmax_fn: _SoftmaxFn = softmax_ctor(dim=1)
        x_fp32: NDArray[np.float32] = x.astype(np.float32)
        x_tensor: _TensorProtocol = tensor(x_fp32, dtype=fp32_dtype)

        with no_grad():
            logits: _TensorProtocol = self._model(x_tensor)
            proba: _TensorProtocol = softmax_fn(logits)

        proba_np: NDArray[np.float64] = proba.cpu().numpy().astype(np.float64)
        return proba_np

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
        torch_mod = __import__("torch")
        nn_mod = __import__("torch.nn", fromlist=["Softmax"])
        enable_grad: _EnableGradFactory = torch_mod.enable_grad
        tensor: _TensorCtor = torch_mod.tensor
        fp32_dtype: _DTypeProtocol = torch_mod.float32
        softmax_ctor: _SoftmaxCtor = nn_mod.Softmax

        softmax_fn: _SoftmaxFn = softmax_ctor(dim=1)

        x_fp32: NDArray[np.float32] = x.astype(np.float32)
        x_tensor: _TensorProtocol = tensor(
            x_fp32,
            dtype=fp32_dtype,
            requires_grad=True,
        )

        with enable_grad():
            logits: _TensorProtocol = self._model(x_tensor)
            proba: _TensorProtocol = softmax_fn(logits)
            target_proba: _TensorProtocol = proba.select(1, target_class)
            scalar_output: _TensorProtocol = target_proba.sum()
            scalar_output.backward()

        grad_tensor = x_tensor.grad
        assert grad_tensor is not None, "Gradient should not be None after backward()"
        grad_cpu: _TensorProtocol = grad_tensor.cpu()
        grad_numpy = grad_cpu.numpy()
        gradients: NDArray[np.float64] = grad_numpy.astype(np.float64)
        return gradients


def _load_mlp_model(model_path: str, config: MLPModelConfig) -> GradientPredictorProtocol:
    """Load MLP model from file.

    Args:
        model_path: Path to saved model file (.pt format).
        config: Model architecture configuration.

    Returns:
        Model implementing GradientPredictorProtocol.
    """
    torch_mod = __import__("torch")
    load_fn: _TorchLoadFn = torch_mod.load

    model = _build_mlp_model(
        n_in=config["n_features"],
        hidden=config["hidden_sizes"],
        dropout=config["dropout"],
    )

    state_dict: dict[str, _TensorProtocol] = load_fn(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    _ = model.eval()

    return _MLPPrepared(model)


# ---------------------------------------------------------------------------
# LSTM Model Config and Loading
# ---------------------------------------------------------------------------


class LSTMModelConfig(TypedDict, total=True):
    """Configuration required to reconstruct LSTM model architecture.

    Args:
        n_features: Number of input features (flat, before reshaping to sequences).
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between LSTM layers.
        bidirectional: Whether to use bidirectional LSTM.
        sequence_length: Number of time steps in each sequence.
    """

    n_features: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool
    sequence_length: int


class _LSTMLayerProto(Protocol):
    """Protocol for LSTM layer."""

    def __call__(self, x: _TensorProtocol) -> tuple[_TensorProtocol, _HiddenStateProtocol]: ...
    def eval(self) -> _LSTMLayerProto: ...
    def state_dict(self) -> dict[str, _TensorProtocol]: ...
    def load_state_dict(self, state_dict: dict[str, _TensorProtocol]) -> None: ...


class _LinearLayerProto(Protocol):
    """Protocol for Linear layer."""

    def __call__(self, x: _TensorProtocol) -> _TensorProtocol: ...
    def eval(self) -> _LinearLayerProto: ...
    def state_dict(self) -> dict[str, _TensorProtocol]: ...
    def load_state_dict(self, state_dict: dict[str, _TensorProtocol]) -> None: ...


class _NNLSTMCtor(Protocol):
    """Protocol for nn.LSTM constructor."""

    def __call__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool,
        dropout: float,
        bidirectional: bool,
    ) -> _LSTMLayerProto: ...


class _NNLinearLayerCtor(Protocol):
    """Protocol for nn.Linear constructor for LSTM classifier head."""

    def __call__(self, in_features: int, out_features: int) -> _LinearLayerProto: ...


class _LSTMClassifierWrapper:
    """LSTM classifier wrapper for loading saved models."""

    def __init__(self, lstm: _LSTMLayerProto, fc: _LinearLayerProto) -> None:
        """Initialize wrapper.

        Args:
            lstm: LSTM layer.
            fc: Fully connected classification head.
        """
        self._lstm = lstm
        self._fc = fc

    def eval(self) -> _LSTMClassifierWrapper:
        """Set evaluation mode.

        Returns:
            Self for chaining.
        """
        _ = self._lstm.eval()
        _ = self._fc.eval()
        return self

    def __call__(self, x: _TensorProtocol) -> _TensorProtocol:
        """Forward pass: (batch, seq_len, input_size) -> (batch, 2).

        Args:
            x: Input tensor with shape (batch, seq_len, features_per_step).

        Returns:
            Logits tensor with shape (batch, 2).
        """
        out_tuple = self._lstm(x)
        lstm_out: _TensorProtocol = out_tuple[0]
        last_out: _TensorProtocol = lstm_out.select(1, -1)
        return self._fc(last_out)

    def load_state_dict(self, state_dict: dict[str, _TensorProtocol]) -> None:
        """Load state dictionary.

        Args:
            state_dict: Dictionary with 'lstm.*' and 'fc.*' keys.
        """
        lstm_state: dict[str, _TensorProtocol] = {}
        fc_state: dict[str, _TensorProtocol] = {}
        for k, v in state_dict.items():
            if k.startswith("lstm."):
                lstm_state[k[5:]] = v
            elif k.startswith("fc."):
                fc_state[k[3:]] = v
        self._lstm.load_state_dict(lstm_state)
        self._fc.load_state_dict(fc_state)


def _build_lstm_model(config: LSTMModelConfig) -> _LSTMClassifierWrapper:
    """Build LSTM model architecture.

    Args:
        config: LSTM model configuration.

    Returns:
        LSTM classifier wrapper (on CPU).
    """
    nn_mod = __import__("torch.nn", fromlist=["LSTM", "Linear"])
    lstm_ctor: _NNLSTMCtor = nn_mod.LSTM
    linear_ctor: _NNLinearLayerCtor = nn_mod.Linear

    input_size = config["n_features"] // config["sequence_length"]

    lstm: _LSTMLayerProto = lstm_ctor(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        batch_first=True,
        dropout=config["dropout"] if config["num_layers"] > 1 else 0.0,
        bidirectional=config["bidirectional"],
    )

    num_directions = 2 if config["bidirectional"] else 1
    lstm_out_size = config["hidden_size"] * num_directions
    fc: _LinearLayerProto = linear_ctor(lstm_out_size, 2)

    return _LSTMClassifierWrapper(lstm, fc)


class _LSTMPrepared:
    """Prepared LSTM model implementing PredictorProtocol with gradient support."""

    def __init__(
        self,
        model: _LSTMClassifierWrapper,
        sequence_length: int,
        n_features: int,
    ) -> None:
        """Initialize with LSTM model.

        Args:
            model: LSTM classifier wrapper.
            sequence_length: Number of time steps.
            n_features: Number of flat input features.
        """
        self._model = model
        self._sequence_length = sequence_length
        self._n_features = n_features

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return class probabilities for input samples.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Class probabilities with shape (n_samples, n_classes).
        """
        torch_mod = __import__("torch")
        nn_mod = __import__("torch.nn", fromlist=["Softmax"])
        no_grad: _NoGradFactory = torch_mod.no_grad
        tensor: _TensorCtor = torch_mod.tensor
        fp32_dtype: _DTypeProtocol = torch_mod.float32
        softmax_ctor: _SoftmaxCtor = nn_mod.Softmax

        softmax_fn: _SoftmaxFn = softmax_ctor(dim=1)

        n_samples = int(x.shape[0])
        features_per_step = self._n_features // self._sequence_length
        x_seq: NDArray[np.float64] = x.reshape(n_samples, self._sequence_length, features_per_step)

        x_seq_fp32: NDArray[np.float32] = x_seq.astype(np.float32)
        x_tensor: _TensorProtocol = tensor(x_seq_fp32, dtype=fp32_dtype)

        with no_grad():
            logits: _TensorProtocol = self._model(x_tensor)
            proba: _TensorProtocol = softmax_fn(logits)

        proba_np: NDArray[np.float64] = proba.cpu().numpy().astype(np.float64)
        return proba_np

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
        torch_mod = __import__("torch")
        nn_mod = __import__("torch.nn", fromlist=["Softmax"])
        enable_grad: _EnableGradFactory = torch_mod.enable_grad
        tensor: _TensorCtor = torch_mod.tensor
        fp32_dtype: _DTypeProtocol = torch_mod.float32
        softmax_ctor: _SoftmaxCtor = nn_mod.Softmax

        softmax_fn: _SoftmaxFn = softmax_ctor(dim=1)

        n_samples = int(x.shape[0])
        features_per_step = self._n_features // self._sequence_length
        x_seq: NDArray[np.float64] = x.reshape(n_samples, self._sequence_length, features_per_step)

        x_seq_fp32: NDArray[np.float32] = x_seq.astype(np.float32)
        x_tensor: _TensorProtocol = tensor(
            x_seq_fp32,
            dtype=fp32_dtype,
            requires_grad=True,
        )

        with enable_grad():
            logits: _TensorProtocol = self._model(x_tensor)
            proba: _TensorProtocol = softmax_fn(logits)
            target_proba: _TensorProtocol = proba.select(1, target_class)
            scalar_output: _TensorProtocol = target_proba.sum()
            scalar_output.backward()

        grad_tensor = x_tensor.grad
        assert grad_tensor is not None, "Gradient should not be None after backward()"
        grad_cpu: _TensorProtocol = grad_tensor.cpu()
        grad_numpy = grad_cpu.numpy()
        grad_seq: NDArray[np.float64] = grad_numpy.astype(np.float64)

        seq_len = int(grad_seq.shape[1])
        features_per_step_actual = int(grad_seq.shape[2])
        flat_seq_features = seq_len * features_per_step_actual
        grad_flat: NDArray[np.float64] = grad_seq.reshape(n_samples, flat_seq_features)

        gradients: NDArray[np.float64] = grad_flat[:, : self._n_features]
        return gradients


def _load_lstm_model(model_path: str, config: LSTMModelConfig) -> GradientPredictorProtocol:
    """Load LSTM model from file.

    Args:
        model_path: Path to saved model file (.pt format).
        config: Model architecture configuration.

    Returns:
        Model implementing GradientPredictorProtocol.
    """
    torch_mod = __import__("torch")
    load_fn: _TorchLoadFn = torch_mod.load

    model = _build_lstm_model(config)

    state_dict: dict[str, _TensorProtocol] = load_fn(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    _ = model.eval()

    return _LSTMPrepared(
        model=model,
        sequence_length=config["sequence_length"],
        n_features=config["n_features"],
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
        ValueError: If required config is missing for MLP/LSTM backend.
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
    # backend == "lstm"
    if lstm_config is None:
        raise ValueError("lstm_config is required for LSTM backend")
    return _load_lstm_model(model_path, lstm_config)


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
