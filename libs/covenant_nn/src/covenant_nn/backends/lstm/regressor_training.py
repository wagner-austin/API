"""Training internals for the LSTM regressor backend."""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import TypedDict

import numpy as np
from covenant_ml.backends.regressor_protocol import (
    RegressorProgressCallback,
)
from covenant_ml.metrics_regression import compute_all_regression_metrics
from covenant_ml.preprocessing import AutoPreprocessor
from covenant_ml.trainer import RegressionDataSplits
from covenant_ml.types import LSTMConfig
from covenant_ml.types_regression import (
    RegressionMetrics,
    RegressionTrainProgress,
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

from covenant_nn.backends.lstm.regressor_protocols import (
    _AutocastFactory,
    _CudnnConfigProto,
    _GradScalerProto,
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

    if device == "cuda":
        model = model.to("cuda")
    return model


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


class _RegressorTrainComponents(TypedDict):
    """Components assembled for LSTM regression training."""

    model: TrainableModel
    optimizer: _OptimizerProto
    loss_fn: _LossProto
    scaler: _GradScalerProto | None


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

    if device == "cuda":
        backends_mod = __import__("torch.backends", fromlist=["cudnn"])
        cudnn: _CudnnConfigProto = backends_mod.cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False

    scaler: _GradScalerProto | None = None
    if device == "cuda" and precision != "fp32":
        amp_mod = __import__("torch.amp", fromlist=["GradScaler"])
        scaler = amp_mod.GradScaler("cuda")

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


def _reshape_to_sequence(x: NDArray[np.float64], sequence_length: int) -> NDArray[np.float64]:
    """Reshape tabular data to sequence format."""
    return reshape_flat_to_pseudo_sequences(x, sequence_length)


# =============================================================================
# Training loop
# =============================================================================


def _train_one_epoch_regression(
    *,
    model: TrainableModel,
    optimizer: _OptimizerProto,
    loss_fn: _LossProto,
    scaler: _GradScalerProto | None,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    batch_size: int,
    device: str,
    train_scale: float,
    sequence_length: int,
) -> float:
    """Train model for one epoch and return average MSE loss.

    Args:
        model: The LSTM model to train.
        optimizer: Optimizer instance.
        loss_fn: MSELoss instance.
        scaler: Optional GradScaler for mixed precision.
        x_train: Training features.
        y_train: Training targets (float64).
        batch_size: Mini-batch size.
        device: Device string ('cpu' or 'cuda').
        train_scale: Loss scaling factor (warmup + LR decay).
        sequence_length: LSTM sequence length for reshaping.

    Returns:
        Average MSE loss over the epoch.
    """
    torch_mod = _import_torch()

    model.train()
    total_loss = 0.0
    total_count = 0
    n_train: int = int(x_train.shape[0])

    x_seq: NDArray[np.float64] = _reshape_to_sequence(x_train, sequence_length)

    for start in range(0, n_train, batch_size):
        end: int = min(n_train, start + batch_size)
        batch_len: int = end - start
        xb: TensorProtocol = torch_mod.tensor(x_seq[start:end], dtype=torch_mod.float32)
        yb: TensorProtocol = torch_mod.tensor(y_train[start:end], dtype=torch_mod.float32)
        if device == "cuda":
            xb = xb.cuda()
            yb = yb.cuda()

        optimizer.zero_grad()
        if scaler is not None:
            amp_mod = __import__("torch.amp", fromlist=["autocast"])
            autocast: _AutocastFactory = amp_mod.autocast
            with autocast("cuda", dtype=torch_mod.float16):
                logits: TensorProtocol = model(xb)
                preds: TensorProtocol = logits.select(1, 0)
                loss: TensorProtocol = loss_fn(preds, yb)
            scaled: TensorProtocol = scaler.scale(loss * float(train_scale))
            scaled.backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with nullcontext():
                logits = model(xb)
                preds = logits.select(1, 0)
                loss = loss_fn(preds, yb)
                (loss * float(train_scale)).backward()
                optimizer.step()

        total_loss += float(loss.item()) * batch_len
        total_count += batch_len

    return total_loss / max(1, total_count)


def _validate_regression_model(
    *,
    model: TrainableModel,
    loss_fn: _LossProto,
    x_val: NDArray[np.float64],
    y_val: NDArray[np.float64],
    batch_size: int,
    device: str,
    sequence_length: int,
) -> tuple[float, float]:
    """Validate model and return (val_loss, val_rmse).

    Args:
        model: The trained LSTM model.
        loss_fn: MSELoss instance.
        x_val: Validation features.
        y_val: Validation targets (float64).
        batch_size: Mini-batch size.
        device: Device string.
        sequence_length: LSTM sequence length for reshaping.

    Returns:
        Tuple of (average MSE loss, RMSE on validation set).
    """
    torch_mod = _import_torch()

    model.eval()
    v_preds: list[float] = []
    v_targets: list[float] = []
    v_loss_total = 0.0
    v_count = 0
    n_val: int = int(x_val.shape[0])

    x_seq: NDArray[np.float64] = _reshape_to_sequence(x_val, sequence_length)

    with torch_mod.no_grad():
        for start in range(0, n_val, batch_size):
            end: int = min(n_val, start + batch_size)
            batch_len: int = end - start
            xb: TensorProtocol = torch_mod.tensor(x_seq[start:end], dtype=torch_mod.float32)
            yb: TensorProtocol = torch_mod.tensor(y_val[start:end], dtype=torch_mod.float32)
            if device == "cuda":
                xb = xb.cuda()
                yb = yb.cuda()

            logits: TensorProtocol = model(xb)
            preds: TensorProtocol = logits.select(1, 0)
            v_loss_total += float(loss_fn(preds, yb).item()) * batch_len
            v_count += batch_len

            preds_np: NDArray[np.float64] = preds.detach().cpu().numpy().astype(np.float64)
            v_preds.extend([float(v) for v in preds_np.flat])
            target_slice: NDArray[np.float64] = y_val[start:end]
            v_targets.extend([float(v) for v in target_slice.flat])

    val_loss = v_loss_total / max(1, v_count)
    # Compute RMSE from Python lists to avoid np.mean Any typing
    sse = 0.0
    for k in range(v_count):
        d = v_preds[k] - v_targets[k]
        sse += d * d
    val_rmse = math.sqrt(sse / max(1, v_count))
    return val_loss, val_rmse


class _EarlyStopState(TypedDict):
    """State for early stopping tracking (regression, RMSE lower is better)."""

    best_val_rmse: float
    best_round: int
    best_state: dict[str, TensorProtocol] | None
    patience: int
    early_stopped: bool


def _run_regression_training_loop(
    *,
    components: _RegressorTrainComponents,
    splits: RegressionDataSplits,
    cfg: LSTMConfig,
    device: str,
    output_dir: Path,
    progress: RegressorProgressCallback | None,
    sequence_length: int,
) -> _EarlyStopState:
    """Run the regression training loop with early stopping on val RMSE.

    Args:
        components: Model, optimizer, loss, AMP helpers.
        splits: Preprocessed regression data splits.
        cfg: LSTM training configuration.
        device: Resolved device string.
        output_dir: Directory for checkpoints.
        progress: Optional callback for progress updates.
        sequence_length: LSTM sequence length for reshaping.

    Returns:
        _EarlyStopState with best model state and training metadata.
    """
    torch_mod = _import_torch()

    batch_size = int(cfg["batch_size"])
    max_patience = int(cfg["early_stopping_patience"])
    n_epochs = int(cfg["n_epochs"])

    state: _EarlyStopState = {
        "best_val_rmse": float("inf"),
        "best_round": 0,
        "best_state": None,
        "patience": 0,
        "early_stopped": False,
    }

    model = components["model"]
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    warmup_epochs: int = 3
    lr_scale: float = 1.0

    for epoch in range(1, n_epochs + 1):
        warmup_scale = 1.0 if epoch > warmup_epochs else float(epoch) / float(warmup_epochs)
        train_scale = float(warmup_scale) * float(lr_scale)

        train_loss = _train_one_epoch_regression(
            model=model,
            optimizer=components["optimizer"],
            loss_fn=components["loss_fn"],
            scaler=components["scaler"],
            x_train=splits.x_train,
            y_train=splits.y_train,
            batch_size=batch_size,
            device=device,
            train_scale=train_scale,
            sequence_length=sequence_length,
        )

        _val_loss, val_rmse = _validate_regression_model(
            model=model,
            loss_fn=components["loss_fn"],
            x_val=splits.x_val,
            y_val=splits.y_val,
            batch_size=batch_size,
            device=device,
            sequence_length=sequence_length,
        )

        # Annotated: np.sqrt is untyped here, and disallow_any_expr
        # rejects the bare expression.
        train_mse: float = max(0.0, train_loss)
        train_rmse = math.sqrt(train_mse)

        if progress is not None:
            prog: RegressionTrainProgress = {
                "round": epoch,
                "total_rounds": n_epochs,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
            }
            progress(prog)

        # Save last checkpoint
        torch_mod.save(model.state_dict(), str(ckpt_dir / "last.pt"))

        # Check for improvement (RMSE: lower is better)
        if val_rmse < state["best_val_rmse"]:
            state["best_val_rmse"] = val_rmse
            state["best_round"] = epoch
            state["best_state"] = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            torch_mod.save(model.state_dict(), str(ckpt_dir / "best.pt"))
            state["patience"] = 0
        else:
            state["patience"] += 1
            if state["patience"] in (2, 4):
                lr_scale *= 0.5
            if state["patience"] >= max_patience:
                state["early_stopped"] = True
                break

    return state


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
            if device == "cuda":
                xb = xb.cuda()
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
