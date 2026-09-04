"""Training internals for the LSTM regressor backend."""

from __future__ import annotations

import numpy as np
from covenant_ml.metrics_regression import compute_all_regression_metrics
from covenant_ml.preprocessing import AutoPreprocessor
from covenant_ml.trainer import RegressionDataSplits
from covenant_ml.types import LSTMConfig
from covenant_ml.types_regression import (
    RegressionMetrics,
)
from numpy.typing import NDArray
from platform_core.logging import get_logger
from platform_ml.torch_types import (
    DeviceProtocol,
    DTypeProtocol,
    TensorIterable,
    TensorProtocol,
    TrainableModel,
    _import_torch,
    set_manual_seed,
)

from covenant_nn.backends import _amp
from covenant_nn.backends.lstm.regressor_protocols import (
    _LinearLayerProto,
    _LossCtor,
    _LossProto,
    _LSTMLayerProto,
    _NNLinearCtor,
    _NNLSTMCtor,
    _OptimizerCtor,
    _OptimizerProto,
    _SequenceTensorProto,
    _TensorCtor,
)
from covenant_nn.backends.lstm.regressor_training_loop import (
    _RegressorTrainComponents,
    _reshape_to_sequence,
)

from .sequences import reshape_flat_to_pseudo_sequences

_log = get_logger(__name__)


class _LSTMRegressorWrapper:
    """LSTM regressor using composition instead of inheritance.

    Implements TrainableModel protocol by composing LSTM and Linear layers.
    Output is (batch, 1) for regression via a single-neuron linear head.
    """

    def __init__(self, lstm: _LSTMLayerProto, fc: _LinearLayerProto) -> None:
        self._lstm = lstm
        self._fc = fc

    def train(self) -> TrainableModel:
        """Set training mode."""
        self._lstm.train(True)
        self._fc.train(True)
        return self

    def eval(self) -> TrainableModel:
        """Set evaluation mode."""
        self._lstm.eval()
        self._fc.eval()
        return self

    def __call__(self, x: TensorProtocol) -> TensorProtocol:
        """Forward pass: (batch, seq_len, input_size) -> (batch, 1)."""
        out_tuple = self._lstm(x)
        lstm_out: _SequenceTensorProto = out_tuple[0]
        # Take the last timestep output using select (dim=1 is seq_len, -1 is last)
        last_out: TensorProtocol = lstm_out.select(1, -1)
        return self._fc(last_out)

    def state_dict(self) -> dict[str, TensorProtocol]:
        """Return combined state dictionary."""
        combined: dict[str, TensorProtocol] = {}
        for k, v in self._lstm.state_dict().items():
            combined[f"lstm.{k}"] = v
        for k, v in self._fc.state_dict().items():
            combined[f"fc.{k}"] = v
        return combined

    def load_state_dict(self, state_dict: dict[str, TensorProtocol]) -> None:
        """Load state dictionary."""
        lstm_state: dict[str, TensorProtocol] = {}
        fc_state: dict[str, TensorProtocol] = {}
        for k, v in state_dict.items():
            if k.startswith("lstm."):
                lstm_state[k[5:]] = v
            elif k.startswith("fc."):
                fc_state[k[3:]] = v
        self._lstm.load_state_dict(lstm_state)
        self._fc.load_state_dict(fc_state)

    def parameters(self) -> TensorIterable:
        """Return all model parameters."""
        lstm_params = list(self._lstm.parameters())
        fc_params = list(self._fc.parameters())
        all_params: list[TensorProtocol] = lstm_params + fc_params
        return all_params

    def to(self, device: DeviceProtocol | str) -> TrainableModel:
        """Move model to device."""
        device_str: str = device if isinstance(device, str) else device.type
        self._lstm = self._lstm.to(device_str)
        self._fc = self._fc.to(device_str)
        return self


def _build_regressor_model(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    device: str,
) -> TrainableModel:
    """Build LSTM regressor model with single-neuron output.

    Args:
        input_size: Features per timestep.
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout between LSTM layers (only if num_layers > 1).
        bidirectional: Whether to use bidirectional LSTM.
        device: Target device ('cpu' or 'cuda').

    Returns:
        A TrainableModel ready for training.
    """
    nn_mod = __import__("torch.nn", fromlist=["LSTM", "Linear"])
    lstm_ctor: _NNLSTMCtor = nn_mod.LSTM
    linear_ctor: _NNLinearCtor = nn_mod.Linear

    lstm: _LSTMLayerProto = lstm_ctor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
        bidirectional=bidirectional,
    )

    num_directions = 2 if bidirectional else 1
    lstm_out_size = hidden_size * num_directions
    fc: _LinearLayerProto = linear_ctor(lstm_out_size, 1)  # Single output for regression

    model: TrainableModel = _LSTMRegressorWrapper(lstm, fc)

    return model.to(device)


# =============================================================================
# Preprocessing
# =============================================================================


def _preprocess_regression_splits(splits: RegressionDataSplits) -> RegressionDataSplits:
    """Preprocess features using AutoPreprocessor.

    Applies outlier capping, special code replacement, imputation, and
    z-score normalization. Statistics computed from training data only.

    Args:
        splits: Raw regression data splits.

    Returns:
        New RegressionDataSplits with preprocessed features, y unchanged.
    """
    preprocessor = AutoPreprocessor()
    dummy_y: NDArray[np.int64] = np.zeros(splits.n_train, dtype=np.int64)
    state = preprocessor.fit(splits.x_train, dummy_y)

    _log.info(
        "Preprocessing LSTM regression splits",
        extra={
            "n_features": state["n_features"],
            "n_train": splits.n_train,
            "n_val": splits.n_val,
            "n_test": splits.n_test,
        },
    )

    return RegressionDataSplits(
        x_train=preprocessor.transform(splits.x_train, state),
        y_train=splits.y_train,
        x_val=preprocessor.transform(splits.x_val, state),
        y_val=splits.y_val,
        x_test=preprocessor.transform(splits.x_test, state),
        y_test=splits.y_test,
    )


# =============================================================================
# Train components
# =============================================================================


def _prepare_regression_components(
    *,
    cfg: LSTMConfig,
    device: str,
    precision: str,
    features_per_step: int,
) -> _RegressorTrainComponents:
    """Build model, optimizer, MSELoss, and AMP helpers for regression.

    Args:
        cfg: LSTM training configuration.
        device: Resolved device string ('cpu' or 'cuda').
        precision: Resolved precision string ('fp32' or 'fp16').
        features_per_step: Number of features per LSTM timestep.

    Returns:
        _RegressorTrainComponents with all training objects.
    """
    _ = _import_torch()
    nn_mod = __import__("torch.nn", fromlist=["MSELoss"])
    loss_ctor: _LossCtor = nn_mod.MSELoss

    set_manual_seed(int(cfg["random_state"]))

    # Guarded on the device, and it has to be. Setting these looks like it
    # should be a no-op without a GPU -- they are module-level flags and no
    # CPU kernel consults them -- but torch resolves the cuDNN version on
    # assignment, and with a CUDA build and no visible device that raises
    # `ValueError: min() arg is an empty sequence` out of
    # torch/backends/cudnn/__init__.py. Measured, not assumed: making these
    # unconditional turned two passing tests red under
    # CUDA_VISIBLE_DEVICES="".
    _amp.configure_cudnn_determinism(device)
    scaler = _amp.make_grad_scaler(device, precision)

    model = _build_regressor_model(
        input_size=features_per_step,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional"],
        device=device,
    )

    optim_mod = __import__("torch.optim", fromlist=["AdamW"])
    opt_ctor: _OptimizerCtor = optim_mod.AdamW
    optimizer: _OptimizerProto = opt_ctor(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_fn: _LossProto = loss_ctor()

    return {
        "model": model,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "scaler": scaler,
    }


# =============================================================================
# Sequence reshaping helper
# =============================================================================


# =============================================================================
# Training loop
# =============================================================================


# =============================================================================
# Final metrics
# =============================================================================


def _finalize_regression_metrics(
    *,
    model: TrainableModel,
    device: str,
    splits: RegressionDataSplits,
    sequence_length: int,
) -> tuple[RegressionMetrics, RegressionMetrics, RegressionMetrics]:
    """Compute final regression metrics on train/val/test splits.

    Args:
        model: Trained LSTM model with best weights loaded.
        device: Device string.
        splits: Preprocessed regression data splits.
        sequence_length: LSTM sequence length for reshaping.

    Returns:
        Tuple of (train_metrics, val_metrics, test_metrics).
    """
    torch_mod = _import_torch()

    def _predict(x: NDArray[np.float64]) -> NDArray[np.float64]:
        x_seq: NDArray[np.float64] = _reshape_to_sequence(x, sequence_length)
        with torch_mod.no_grad():
            xb: TensorProtocol = torch_mod.tensor(x_seq, dtype=torch_mod.float32)
            xb = xb.to(device)
            logits: TensorProtocol = model(xb)
            preds: TensorProtocol = logits.select(1, 0)
            return preds.detach().cpu().numpy().astype(np.float64)

    train = compute_all_regression_metrics(splits.y_train, _predict(splits.x_train))
    val = compute_all_regression_metrics(splits.y_val, _predict(splits.x_val))
    test = compute_all_regression_metrics(splits.y_test, _predict(splits.x_test))
    return train, val, test


# =============================================================================
# Prepared model for inference
# =============================================================================


class _LSTMRegressorPrepared:
    """Prepared LSTM regressor for inference.

    Returns 1D continuous predictions via select(1, 0) on the single
    output neuron. No softmax, no class probabilities.
    """

    def __init__(self, model: TrainableModel, sequence_length: int) -> None:
        self._model = model
        self._sequence_length = sequence_length

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict continuous values for input samples.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Predicted values with shape (n_samples,).
        """
        m = self._model
        m.eval()

        x_seq: NDArray[np.float64] = reshape_flat_to_pseudo_sequences(x, self._sequence_length)
        torch_mod = _import_torch()

        with torch_mod.no_grad():
            xt: TensorProtocol = torch_mod.tensor(x_seq, dtype=torch_mod.float32)
            logits: TensorProtocol = m(xt)
            preds: TensorProtocol = logits.select(1, 0)
            return preds.cpu().numpy().astype(np.float64)

    def compute_regression_gradients(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute gradients of the prediction w.r.t. input features.

        Flat features are reshaped to pseudo-sequences for the forward pass,
        so the gradients come back shaped (n_samples, seq_len,
        features_per_step) and are flattened, then trimmed to the original
        feature count -- the reshape zero-pads up to a multiple of
        sequence_length, and those padding columns are not real features.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Gradients with shape (n_samples, n_features).
        """
        torch_mod = _import_torch()
        tensor: _TensorCtor = torch_mod.tensor
        float32: DTypeProtocol = torch_mod.float32

        m = self._model
        m.eval()

        n_samples = int(x.shape[0])
        n_features = int(x.shape[1])
        x_seq: NDArray[np.float64] = reshape_flat_to_pseudo_sequences(x, self._sequence_length)

        x_tensor: TensorProtocol = tensor(x_seq, dtype=float32)
        x_tensor = x_tensor.requires_grad_(True)

        with torch_mod.enable_grad():
            logits: TensorProtocol = m(x_tensor)
            preds: TensorProtocol = logits.select(1, 0)
            scalar_output: TensorProtocol = preds.sum()
            scalar_output.backward()

        grad_tensor = x_tensor.grad
        assert grad_tensor is not None, "Gradient tensor should not be None after backward()"
        grad_cpu: TensorProtocol = grad_tensor.cpu()
        grad_seq: NDArray[np.float64] = grad_cpu.numpy().astype(np.float64)

        seq_len = int(grad_seq.shape[1])
        features_per_step = int(grad_seq.shape[2])
        grad_flat: NDArray[np.float64] = grad_seq.reshape(n_samples, seq_len * features_per_step)
        gradients: NDArray[np.float64] = grad_flat[:, :n_features]
        return gradients


# =============================================================================
# Backend class
# =============================================================================
