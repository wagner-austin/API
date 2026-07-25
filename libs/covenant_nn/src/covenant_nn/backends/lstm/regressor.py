"""LSTM regressor backend for continuous target prediction.

Parallel to LSTM classifier backend (backend.py). Key differences:
- Output layer: nn.Linear(hidden_size * num_directions, 1) instead of 2
- Loss: nn.MSELoss() instead of nn.CrossEntropyLoss()
- No class weighting or scale_pos_weight
- No softmax — direct scalar output via select(1, 0)
- Early stopping on val RMSE (lower is better, not AUC higher is better)
- predict() returns 1D float64 array (not 2D class probabilities)
- No compute_gradients (regression has no target class)
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Protocol, TypedDict, TypeGuard

import numpy as np
from covenant_ml.backends.protocol import BackendCapabilities
from covenant_ml.backends.regressor_protocol import (
    PreparedRegressor,
    RegressorProgressCallback,
)
from covenant_ml.metrics import compute_all_regression_metrics
from covenant_ml.optimizer.search_spaces import (
    make_lstm_default_space,
    make_lstm_focused_space,
)
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SearchSpace,
)
from covenant_ml.preprocessing import AutoPreprocessor
from covenant_ml.trainer import RegressionDataSplits, regression_split
from covenant_ml.types import (
    FeatureImportance,
    LSTMConfig,
    RegressionMetrics,
    RegressionTrainOutcome,
    RegressionTrainProgress,
    RegressorBackendName,
    RegressorTrainConfig,
)
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_float,
    require_int,
)
from platform_core.logging import get_logger
from platform_ml.device_selector import resolve_device, resolve_precision
from platform_ml.torch_types import (
    DeviceProtocol,
    DTypeProtocol,
    TensorIterable,
    TensorProtocol,
    TrainableModel,
    _import_torch,
    set_manual_seed,
)

from .sequences import reshape_flat_to_pseudo_sequences

_log = get_logger(__name__)


# =============================================================================
# Protocols for PyTorch dynamic imports
# =============================================================================


class _SequenceTensorProto(Protocol):
    """Protocol for 3D sequence tensor that supports select indexing.

    Shape: (batch, seq_len, hidden_size)
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    def select(self, dim: int, index: int) -> TensorProtocol:
        """Select a slice along a dimension, removing that dimension."""
        ...

    def detach(self) -> _SequenceTensorProto: ...

    def cpu(self) -> _SequenceTensorProto: ...


class _LSTMLayerProto(Protocol):
    """Protocol for nn.LSTM layer with tuple output."""

    def __call__(
        self, x: TensorProtocol
    ) -> tuple[_SequenceTensorProto, tuple[TensorProtocol, TensorProtocol]]: ...
    def train(self, mode: bool = True) -> _LSTMLayerProto: ...
    def eval(self) -> _LSTMLayerProto: ...
    def state_dict(self) -> dict[str, TensorProtocol]: ...
    def load_state_dict(self, state_dict: dict[str, TensorProtocol]) -> None: ...
    def parameters(self) -> TensorIterable: ...
    def to(self, device: str) -> _LSTMLayerProto: ...
    def cuda(self) -> _LSTMLayerProto: ...


class _LinearLayerProto(Protocol):
    """Protocol for nn.Linear layer."""

    def __call__(self, x: TensorProtocol) -> TensorProtocol: ...
    def train(self, mode: bool = True) -> _LinearLayerProto: ...
    def eval(self) -> _LinearLayerProto: ...
    def state_dict(self) -> dict[str, TensorProtocol]: ...
    def load_state_dict(self, state_dict: dict[str, TensorProtocol]) -> None: ...
    def parameters(self) -> TensorIterable: ...
    def to(self, device: str) -> _LinearLayerProto: ...
    def cuda(self) -> _LinearLayerProto: ...


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


class _NNLinearCtor(Protocol):
    """Protocol for nn.Linear constructor."""

    def __call__(self, in_features: int, out_features: int) -> _LinearLayerProto: ...


class _OptimizerProto(Protocol):
    """Protocol for optimizer."""

    def zero_grad(self) -> None: ...
    def step(self) -> None: ...


class _OptimizerCtor(Protocol):
    """Protocol for optimizer constructor."""

    def __call__(self, params: TensorIterable, lr: float) -> _OptimizerProto: ...


class _LossProto(Protocol):
    """Protocol for loss function (MSELoss)."""

    def __call__(self, input: TensorProtocol, target: TensorProtocol) -> TensorProtocol: ...


class _LossCtor(Protocol):
    """Protocol for MSELoss constructor (no arguments)."""

    def __call__(self) -> _LossProto: ...


class _GradScalerProto(Protocol):
    """Protocol for gradient scaler."""

    def scale(self, loss: TensorProtocol) -> TensorProtocol: ...
    def step(self, optimizer: _OptimizerProto) -> None: ...
    def update(self) -> None: ...


class _AutocastFactory(Protocol):
    """Protocol for autocast context manager factory."""

    def __call__(
        self, device_type: str, *, dtype: DTypeProtocol
    ) -> AbstractContextManager[None]: ...


class _CudnnConfigProto(Protocol):
    """Protocol for torch.backends.cudnn config."""

    deterministic: bool
    benchmark: bool


class _TensorCtor(Protocol):
    """Protocol for torch.tensor constructor."""

    def __call__(self, data: NDArray[np.float64], dtype: DTypeProtocol) -> TensorProtocol: ...


class _NoGradFactory(Protocol):
    """Protocol for torch.no_grad context manager factory."""

    def __call__(self) -> AbstractContextManager[None]: ...


# =============================================================================
# Constants and type guard
# =============================================================================


LSTM_REGRESSOR_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": True,
    "supports_early_stopping": True,
    "supports_feature_importance": False,
    "model_format": "pt",
}


def _is_lstm_config(cfg: RegressorTrainConfig) -> TypeGuard[LSTMConfig]:
    """Check if config is LSTMConfig by looking for LSTM-specific keys.

    Args:
        cfg: Regressor training configuration to check.

    Returns:
        True if config contains hidden_size and bidirectional keys.
    """
    return (
        isinstance(cfg, dict)
        and "hidden_size" in cfg
        and "num_layers" in cfg
        and "bidirectional" in cfg
    )


# =============================================================================
# Model metadata for save/load
# =============================================================================


class _LSTMRegressorMeta(TypedDict, total=True):
    """Architecture metadata saved alongside .pt state dict.

    Contains the minimal parameters needed to reconstruct the LSTM
    architecture before loading the state dict.

    Args:
        n_features: Number of input features the model was trained on.
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between LSTM layers.
        bidirectional: Whether the LSTM is bidirectional.
        sequence_length: Sequence length used to reshape tabular data.
    """

    n_features: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool
    sequence_length: int


def _encode_lstm_regressor_meta(meta: _LSTMRegressorMeta) -> str:
    """Encode LSTM regressor metadata to JSON string.

    Args:
        meta: Metadata TypedDict to encode.

    Returns:
        JSON string representation.
    """
    return dump_json_str(
        {
            "n_features": meta["n_features"],
            "hidden_size": meta["hidden_size"],
            "num_layers": meta["num_layers"],
            "dropout": meta["dropout"],
            "bidirectional": meta["bidirectional"],
            "sequence_length": meta["sequence_length"],
        }
    )


def _decode_lstm_regressor_meta(raw: JSONValue) -> _LSTMRegressorMeta:
    """Decode and validate LSTM regressor metadata from parsed JSON.

    Args:
        raw: Parsed JSON value (from load_json_str).

    Returns:
        Validated _LSTMRegressorMeta TypedDict.

    Raises:
        JSONTypeError: If structure or types are invalid.
    """
    obj = narrow_json_to_dict(raw)
    n_features = require_int(obj, "n_features")
    hidden_size = require_int(obj, "hidden_size")
    num_layers = require_int(obj, "num_layers")
    dropout = require_float(obj, "dropout")
    bidirectional = require_bool(obj, "bidirectional")
    sequence_length = require_int(obj, "sequence_length")
    return {
        "n_features": n_features,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "sequence_length": sequence_length,
    }


# =============================================================================
# Model building
# =============================================================================


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


# =============================================================================
# Backend class
# =============================================================================


class LSTMRegressorBackend:
    """LSTM regressor backend for continuous target prediction.

    Implements the RegressorBackend protocol. Parallel to LSTMBackend
    (classifier). Uses MSELoss, no class weights, output dim=1,
    early stopping on validation RMSE.
    """

    def get_default_search_space(self) -> SearchSpace:
        """Return the backend's default hyperparameter search space.

        Returns:
            The lstm_reg default SearchSpace.
        """
        return make_lstm_default_space()

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Return a search space narrowed around prior best params.

        Args:
            best_int_params: Best integer params from prior optimization.
            best_float_params: Best float params from prior optimization.

        Returns:
            The lstm_reg focused SearchSpace.
        """
        return make_lstm_focused_space(
            best_hidden_size=best_int_params["hidden_size"],
            best_num_layers=best_int_params["num_layers"],
            best_learning_rate=best_float_params["learning_rate"],
        )

    def backend_name(self) -> RegressorBackendName:
        """Return the backend identifier.

        Returns:
            The backend name literal 'lstm_reg'.
        """
        return "lstm_reg"

    def capabilities(self) -> BackendCapabilities:
        """Return capability flags.

        Returns:
            BackendCapabilities for LSTM regressor.
        """
        return LSTM_REGRESSOR_CAPABILITIES

    def prepare(
        self,
        *,
        n_features: int,
        feature_names: list[str] | None,
    ) -> PreparedRegressor:
        """Prepare a minimal regressor for inference.

        Uses default sequence_length = n_features (each feature as one timestep).

        Args:
            n_features: Number of input features.
            feature_names: Optional feature names (unused).

        Returns:
            A PreparedRegressor with a simple LSTM model.
        """
        _ = feature_names
        default_sequence_length = n_features
        features_per_step = 1
        model = _build_regressor_model(
            input_size=features_per_step,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            device="cpu",
        )
        return _LSTMRegressorPrepared(model, default_sequence_length)

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],
        feature_names: list[str] | None,
        config: RegressorTrainConfig,
        output_dir: Path,
        progress: RegressorProgressCallback | None,
    ) -> RegressionTrainOutcome:
        """Train an LSTM regressor with early stopping on val RMSE.

        Splits data, preprocesses features (outlier capping, imputation,
        z-score normalization), reshapes to pseudo-sequences, trains with
        MSELoss, and returns complete regression metrics.

        Args:
            x_features: Feature matrix (n_samples, n_features).
            y_targets: Continuous target values (n_samples,).
            feature_names: Optional feature names (unused by LSTM).
            config: LSTMConfig with training hyperparameters.
            output_dir: Directory for model checkpoints and final model.
            progress: Optional callback for training progress.

        Returns:
            RegressionTrainOutcome with metrics from all splits.

        Raises:
            RuntimeError: If config is not LSTMConfig.
            RuntimeError: If training produces no best state (n_epochs=0).
        """
        if not _is_lstm_config(config):
            raise RuntimeError(
                "LSTMRegressorBackend requires LSTMConfig (found RegressorTrainConfig)"
            )
        cfg = config
        _ = feature_names
        device = resolve_device(cfg["device"])
        precision = resolve_precision(cfg["precision"], device)

        raw_splits = regression_split(
            x_features,
            y_targets,
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            test_ratio=cfg["test_ratio"],
            random_state=cfg["random_state"],
        )

        splits = _preprocess_regression_splits(raw_splits)
        n_features = int(splits.x_train.shape[1])
        sequence_length = int(cfg["sequence_length"])
        features_per_step = (n_features + sequence_length - 1) // sequence_length

        components = _prepare_regression_components(
            cfg=cfg,
            device=device,
            precision=precision,
            features_per_step=features_per_step,
        )

        state = _run_regression_training_loop(
            components=components,
            splits=splits,
            cfg=cfg,
            device=device,
            output_dir=output_dir,
            progress=progress,
            sequence_length=sequence_length,
        )

        model = components["model"]
        best_state = state["best_state"]
        if best_state is None:
            raise RuntimeError("Training completed with no best state; check n_epochs >= 1")
        model.load_state_dict(best_state)

        train_metrics, val_metrics, test_metrics = _finalize_regression_metrics(
            model=model, device=device, splits=splits, sequence_length=sequence_length
        )

        torch_mod = _import_torch()
        final_path = output_dir / "lstm_reg_final.pt"
        torch_mod.save(model.state_dict(), str(final_path))

        meta_path = output_dir / "lstm_reg_final.json"
        meta: _LSTMRegressorMeta = {
            "n_features": n_features,
            "hidden_size": int(cfg["hidden_size"]),
            "num_layers": int(cfg["num_layers"]),
            "dropout": float(cfg["dropout"]),
            "bidirectional": bool(cfg["bidirectional"]),
            "sequence_length": sequence_length,
        }
        meta_path.write_text(_encode_lstm_regressor_meta(meta), encoding="utf-8")

        return RegressionTrainOutcome(
            model_path=str(final_path),
            model_id="lstm_reg",
            samples_total=splits.n_total,
            samples_train=splits.n_train,
            samples_val=splits.n_val,
            samples_test=splits.n_test,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_val_rmse=state["best_val_rmse"],
            best_round=state["best_round"],
            total_rounds=int(cfg["n_epochs"]),
            early_stopped=state["early_stopped"],
            config=cfg,
            feature_importances=[],
        )

    def evaluate(
        self,
        *,
        model: PreparedRegressor,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> RegressionMetrics:
        """Evaluate a trained regressor on data.

        Args:
            model: A trained PreparedRegressor.
            x: Feature matrix (n_samples, n_features).
            y: True continuous target values (n_samples,).

        Returns:
            RegressionMetrics with mse, rmse, mae, r_squared, mape.
        """
        preds = model.predict(x)
        return compute_all_regression_metrics(y, preds)

    def save(self, *, model: PreparedRegressor, path: str) -> None:
        """Save not supported; use RegressionTrainOutcome.model_path.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            "LSTMRegressorBackend.save not supported; use RegressionTrainOutcome.model_path."
        )

    def load(self, *, path: str) -> PreparedRegressor:
        """Load a trained LSTM regressor from saved state dict and metadata.

        Expects a JSON metadata file alongside the .pt file at the same
        path with a .json extension. The metadata contains architecture
        parameters needed to reconstruct the model before loading weights.

        Args:
            path: Path to the saved .pt state dict file.

        Returns:
            A PreparedRegressor wrapping the loaded LSTM model.

        Raises:
            FileNotFoundError: If the .pt or .json file does not exist.
            ValueError: If metadata JSON is invalid.
        """
        torch_mod = _import_torch()
        pt_path = Path(path)
        meta_path = pt_path.with_suffix(".json")
        raw = load_json_str(meta_path.read_text(encoding="utf-8"))
        meta = _decode_lstm_regressor_meta(raw)

        sequence_length = meta["sequence_length"]
        features_per_step = (meta["n_features"] + sequence_length - 1) // sequence_length

        model = _build_regressor_model(
            input_size=features_per_step,
            hidden_size=meta["hidden_size"],
            num_layers=meta["num_layers"],
            dropout=meta["dropout"],
            bidirectional=meta["bidirectional"],
            device="cpu",
        )
        state_dict: dict[str, TensorProtocol] = torch_mod.load(str(pt_path))
        model.load_state_dict(state_dict)
        model.eval()
        return _LSTMRegressorPrepared(model, sequence_length)

    def get_feature_importances(
        self,
        *,
        model: PreparedRegressor,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Feature importances not supported for LSTM.

        Args:
            model: A trained regressor (unused).
            feature_names: Feature names (unused).

        Returns:
            None (LSTM has no native feature importance).
        """
        _ = model, feature_names
        return None


def create_lstm_regressor_backend() -> LSTMRegressorBackend:
    """Create an LSTM regressor backend instance.

    Returns:
        A new LSTMRegressorBackend.
    """
    return LSTMRegressorBackend()


__all__ = ["LSTM_REGRESSOR_CAPABILITIES", "LSTMRegressorBackend", "create_lstm_regressor_backend"]
