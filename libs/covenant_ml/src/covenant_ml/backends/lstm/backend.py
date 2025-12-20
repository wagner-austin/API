"""Strict LSTM backend with PyTorch for temporal bankruptcy prediction.

Implements:
- Device + precision resolution via platform_ml
- LSTM-based temporal sequence modeling
- Stratified train/val/test splits
- Early stopping on val AUC with checkpoints
- Strict Protocol-based typing (no direct torch type annotations)

The LSTM backend processes temporal sequences of financial data. It supports:
1. Pre-sequenced data: Use build_sequences() to prepare data with entity/year info
2. Flat data: Automatically reshaped to pseudo-sequences using sequence_length

For proper temporal modeling with real company-year data, use the SequenceBuilder
utility from sequences.py before calling train().
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Protocol, TypedDict, TypeGuard

import numpy as np
from numpy.typing import NDArray
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

from ...metrics import compute_all_metrics
from ...trainer import preprocess_data_splits, stratified_split
from ...types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    LSTMConfig,
    TrainOutcome,
    TrainProgress,
)
from ..protocol import BackendCapabilities, ClassifierBackend, PreparedClassifier, ProgressCallback
from .sequences import reshape_flat_to_pseudo_sequences

_log = get_logger(__name__)


class _SplitsProtocol(Protocol):
    """Protocol for data splits."""

    x_train: NDArray[np.float64]
    y_train: NDArray[np.int64]
    x_val: NDArray[np.float64]
    y_val: NDArray[np.int64]
    x_test: NDArray[np.float64]
    y_test: NDArray[np.int64]

    @property
    def n_train(self) -> int: ...

    @property
    def n_val(self) -> int: ...

    @property
    def n_test(self) -> int: ...

    @property
    def n_total(self) -> int: ...


def _is_lstm_config(cfg: ClassifierTrainConfig) -> TypeGuard[LSTMConfig]:
    """Check if config is LSTMConfig by looking for LSTM-specific keys."""
    return (
        isinstance(cfg, dict)
        and "hidden_size" in cfg
        and "num_layers" in cfg
        and "bidirectional" in cfg
    )


# Protocols for nn.Module components (used via dynamic imports)
class _SequenceTensorProto(Protocol):
    """Protocol for 3D sequence tensor that supports select indexing.

    Provides select() method for extracting specific timesteps.
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


class _SoftmaxCtor(Protocol):
    """Protocol for Softmax constructor."""

    def __call__(self, *, dim: int) -> TrainableModel: ...


class _TensorCtor(Protocol):
    """Protocol for torch.tensor constructor."""

    def __call__(self, data: NDArray[np.float64], dtype: DTypeProtocol) -> TensorProtocol: ...


class _EnableGradFactory(Protocol):
    """Protocol for torch.enable_grad context manager factory."""

    def __call__(self) -> AbstractContextManager[None]: ...


class _LSTMPrepared:
    """Prepared LSTM model for inference and gradient computation.

    Implements both PredictorProtocol and GradientModelProtocol from
    platform_ml.explainers.protocol for use with feature importance explainers.
    """

    def __init__(self, model: TrainableModel, sequence_length: int) -> None:
        self._model = model
        self._sequence_length = sequence_length

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return class probabilities for input samples.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Class probabilities with shape (n_samples, n_classes).
        """
        m = self._model
        m.eval()

        # Reshape to (batch, seq_len, features_per_step)
        x_seq: NDArray[np.float64] = reshape_flat_to_pseudo_sequences(x, self._sequence_length)
        torch_mod = _import_torch()

        with torch_mod.no_grad():
            xt: TensorProtocol = torch_mod.tensor(x_seq, dtype=torch_mod.float32)
            logits: TensorProtocol = m(xt)
            nn_mod = __import__("torch.nn", fromlist=["Softmax"])
            softmax_ctor: _SoftmaxCtor = nn_mod.Softmax
            sm = softmax_ctor(dim=1)
            proba: TensorProtocol = sm(logits)
            proba_np: NDArray[np.float64] = proba.cpu().numpy().astype(np.float64)

        return proba_np

    def compute_gradients(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Compute gradients of output w.r.t. input features.

        Computes d(output[target_class]) / d(input) for each sample.
        Used by gradient-based explainers (GradientExplainer, IntegratedGradientsExplainer).

        The input is reshaped to sequences for LSTM processing, and gradients are
        computed through the full forward pass then reshaped back to flat format.

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
        float32: DTypeProtocol = torch_mod.float32

        # Put model in eval mode
        self._model.eval()

        # Create softmax function
        softmax: _SoftmaxCtor = nn_mod.Softmax
        softmax_fn = softmax(dim=1)

        n_samples = int(x.shape[0])
        n_features = int(x.shape[1])

        # Reshape to sequence format: (batch, seq_len, features_per_step)
        x_seq: NDArray[np.float64] = reshape_flat_to_pseudo_sequences(x, self._sequence_length)

        # Create input tensor and enable gradients
        x_tensor: TensorProtocol = tensor(x_seq, dtype=float32)
        x_tensor = x_tensor.requires_grad_(True)

        with enable_grad():
            # Forward pass through model
            logits: TensorProtocol = self._model(x_tensor)

            # Apply softmax
            proba: TensorProtocol = softmax_fn(logits)

            # Select target class probabilities using select()
            # proba shape: (n_samples, n_classes), select dim=1 (classes), index=target_class
            target_proba: TensorProtocol = proba.select(1, target_class)

            # Sum to get scalar for backward (gradients will be per-sample)
            scalar_output: TensorProtocol = target_proba.sum()

            # Backward pass to compute gradients w.r.t. input
            scalar_output.backward()

        # Extract gradients from input tensor (shape: batch, seq_len, features_per_step)
        # Note: grad is always populated since requires_grad=True and backward() was called
        grad_tensor = x_tensor.grad
        assert grad_tensor is not None, "Gradient tensor should not be None after backward()"
        grad_cpu: TensorProtocol = grad_tensor.cpu()
        grad_numpy = grad_cpu.numpy()
        grad_seq: NDArray[np.float64] = grad_numpy.astype(np.float64)

        # Reshape gradients back to flat format (n_samples, n_features)
        # grad_seq has shape (n_samples, seq_len, features_per_step)
        # Flatten the sequence dimensions
        seq_len = int(grad_seq.shape[1])
        features_per_step = int(grad_seq.shape[2])
        flat_seq_features = seq_len * features_per_step

        # Reshape to (n_samples, seq_len * features_per_step)
        grad_flat: NDArray[np.float64] = grad_seq.reshape(n_samples, flat_seq_features)

        # Trim to match original n_features
        # Note: flat_seq_features will always >= n_features because the forward pass
        # reshape only succeeds when n_features divides evenly by sequence_length
        gradients: NDArray[np.float64] = grad_flat[:, :n_features]

        return gradients


LSTM_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": True,
    "supports_early_stopping": True,
    "supports_feature_importance": False,
    "model_format": "pt",
}


class _LSTMClassifierWrapper:
    """LSTM classifier using composition instead of inheritance.

    This class implements the TrainableModel protocol by composing
    LSTM and Linear layers, avoiding the need to inherit from nn.Module
    which would require type: ignore for dynamic base class.
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
        """Forward pass: (batch, seq_len, input_size) -> (batch, 2)."""
        # LSTM output: (batch, seq_len, hidden*dirs), (h_n, c_n)
        out_tuple = self._lstm(x)
        lstm_out: _SequenceTensorProto = out_tuple[0]
        # Take the last timestep output using select (dim=1 is seq_len, index=-1 is last)
        last_out: TensorProtocol = lstm_out.select(1, -1)
        # Classification logits
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


def _build_model(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    device: str,
) -> TrainableModel:
    """Build LSTM classifier model.

    Uses composition to create an LSTM classifier that implements TrainableModel
    without needing to inherit from nn.Module (which would require type: ignore).
    """
    nn_mod = __import__("torch.nn", fromlist=["LSTM", "Linear"])

    # Get constructors via Protocol annotation
    lstm_ctor: _NNLSTMCtor = nn_mod.LSTM
    linear_ctor: _NNLinearCtor = nn_mod.Linear

    # Create LSTM layer
    lstm: _LSTMLayerProto = lstm_ctor(
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
    fc: _LinearLayerProto = linear_ctor(lstm_out_size, 2)

    # Build wrapper
    model: TrainableModel = _LSTMClassifierWrapper(lstm, fc)

    if device == "cuda":
        model = model.to("cuda")
    return model


class _OptimizerProto(Protocol):
    """Protocol for optimizer."""

    def zero_grad(self) -> None: ...
    def step(self) -> None: ...


class _LossProto(Protocol):
    """Protocol for loss function."""

    def __call__(self, logits: TensorProtocol, targets: TensorProtocol) -> TensorProtocol: ...


class _GradScalerProto(Protocol):
    """Protocol for gradient scaler."""

    def scale(self, loss: TensorProtocol) -> TensorProtocol: ...
    def step(self, optimizer: _OptimizerProto) -> None: ...
    def update(self) -> None: ...


class _TrainComponents(TypedDict):
    """Components needed for training."""

    model: TrainableModel
    optimizer: _OptimizerProto
    loss_fn: _LossProto
    scaler: _GradScalerProto | None
    scale_pos_weight_computed: float


def _compute_class_weight(y_train: NDArray[np.int64]) -> float:
    """Compute scale_pos_weight from training labels."""
    pos_mask: NDArray[np.bool_] = y_train == 1
    neg_mask: NDArray[np.bool_] = y_train == 0
    n_positive = int(np.count_nonzero(pos_mask))
    n_negative = int(np.count_nonzero(neg_mask))
    if n_positive == 0:
        raise ValueError("Training set has no positive samples")
    computed = float(n_negative) / float(n_positive)
    _log.info(
        "Auto-calculated scale_pos_weight for LSTM",
        extra={
            "n_positive": n_positive,
            "n_negative": n_negative,
            "scale_pos_weight": computed,
        },
    )
    return computed


class _CudnnConfigProto(Protocol):
    deterministic: bool
    benchmark: bool


class _WeightedLossCtor(Protocol):
    """Protocol for weighted loss constructor."""

    def __call__(self, weight: TensorProtocol) -> _LossProto: ...


class _OptimizerCtor(Protocol):
    """Protocol for optimizer constructor."""

    def __call__(self, params: TensorIterable, lr: float) -> _OptimizerProto: ...


def _prepare_components(
    *,
    y_train: NDArray[np.int64],
    cfg: LSTMConfig,
    device: str,
    precision: str,
    features_per_step: int,
) -> _TrainComponents:
    """Build model, optimizer, loss and AMP helpers for training."""
    torch_mod = _import_torch()
    nn_mod = __import__("torch.nn", fromlist=["CrossEntropyLoss"])
    loss_ctor: _WeightedLossCtor = nn_mod.CrossEntropyLoss

    # Compute class weights
    scale_pos_weight = _compute_class_weight(y_train)
    class_weights: TensorProtocol = torch_mod.tensor(
        [1.0, scale_pos_weight], dtype=torch_mod.float32
    )
    if device == "cuda":
        class_weights = class_weights.cuda()

    # Seed PyTorch RNG
    set_manual_seed(int(cfg["random_state"]))

    # Enable deterministic CUDA algorithms
    if device == "cuda":
        backends_mod = __import__("torch.backends", fromlist=["cudnn"])
        cudnn: _CudnnConfigProto = backends_mod.cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False

    scaler: _GradScalerProto | None = None
    if device == "cuda" and precision != "fp32":
        amp_mod = __import__("torch.amp", fromlist=["GradScaler"])
        scaler = amp_mod.GradScaler("cuda")

    model = _build_model(
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
    loss_fn: _LossProto = loss_ctor(weight=class_weights)

    return {
        "model": model,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "scaler": scaler,
        "scale_pos_weight_computed": scale_pos_weight,
    }


def _reshape_to_sequence(x: NDArray[np.float64], sequence_length: int) -> NDArray[np.float64]:
    """Reshape tabular data to sequence format."""
    return reshape_flat_to_pseudo_sequences(x, sequence_length)


class _AutocastFactory(Protocol):
    """Protocol for autocast context manager factory."""

    def __call__(
        self, device_type: str, *, dtype: DTypeProtocol
    ) -> AbstractContextManager[None]: ...


def _train_one_epoch(
    *,
    model: TrainableModel,
    optimizer: _OptimizerProto,
    loss_fn: _LossProto,
    scaler: _GradScalerProto | None,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    batch_size: int,
    device: str,
    train_scale: float,
    sequence_length: int,
) -> float:
    """Train model for one epoch and return average loss."""
    torch_mod = _import_torch()

    model.train()
    total_loss = 0.0
    total_count = 0
    n_train = int(x_train.shape[0])

    # Reshape to sequence format
    x_seq: NDArray[np.float64] = _reshape_to_sequence(x_train, sequence_length)

    for start in range(0, n_train, batch_size):
        end = min(n_train, start + batch_size)
        batch_len = end - start
        xb: TensorProtocol = torch_mod.tensor(x_seq[start:end], dtype=torch_mod.float32)
        yb: TensorProtocol = torch_mod.tensor(y_train[start:end], dtype=torch_mod.long)
        if device == "cuda":
            xb = xb.cuda()
            yb = yb.cuda()

        optimizer.zero_grad()
        if scaler is not None:
            amp_mod = __import__("torch.amp", fromlist=["autocast"])
            autocast: _AutocastFactory = amp_mod.autocast
            with autocast("cuda", dtype=torch_mod.float16):
                logits: TensorProtocol = model(xb)
                loss: TensorProtocol = loss_fn(logits, yb)
            scaled: TensorProtocol = scaler.scale(loss * float(train_scale))
            scaled.backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with nullcontext():
                logits = model(xb)
                loss = loss_fn(logits, yb)
                (loss * float(train_scale)).backward()
                optimizer.step()

        batch_loss: float = float(loss.item())
        total_loss += batch_loss * batch_len
        total_count += batch_len

    return total_loss / max(1, total_count)


def _validate_model(
    *,
    model: TrainableModel,
    loss_fn: _LossProto,
    x_val: NDArray[np.float64],
    y_val: NDArray[np.int64],
    batch_size: int,
    device: str,
    sequence_length: int,
) -> tuple[float, float]:
    """Validate model and return (val_loss, val_auc)."""
    torch_mod = _import_torch()
    nn_mod = __import__("torch.nn", fromlist=["Softmax"])
    softmax_ctor: _SoftmaxCtor = nn_mod.Softmax

    model.eval()
    v_probs: list[float] = []
    v_targets: list[int] = []
    v_loss_total = 0.0
    v_count = 0
    n_val = int(x_val.shape[0])

    # Reshape to sequence format
    x_seq: NDArray[np.float64] = _reshape_to_sequence(x_val, sequence_length)

    with torch_mod.no_grad():
        for start in range(0, n_val, batch_size):
            end = min(n_val, start + batch_size)
            batch_len = end - start
            xb: TensorProtocol = torch_mod.tensor(x_seq[start:end], dtype=torch_mod.float32)
            yb: TensorProtocol = torch_mod.tensor(y_val[start:end], dtype=torch_mod.long)
            if device == "cuda":
                xb = xb.cuda()
                yb = yb.cuda()

            logits: TensorProtocol = model(xb)
            loss_val: TensorProtocol = loss_fn(logits, yb)
            v_loss_total += float(loss_val.item()) * batch_len
            v_count += batch_len

            sm = softmax_ctor(dim=1)
            softmax_out: TensorProtocol = sm(logits)
            probs: NDArray[np.float64] = softmax_out.detach().cpu().numpy().astype(np.float64)
            prob_col: NDArray[np.float64] = probs[:, 1]
            v_probs.extend([float(v) for v in prob_col.flat])
            target_slice: NDArray[np.int64] = y_val[start:end]
            v_targets.extend([int(v) for v in target_slice.flat])

    val_metrics = compute_all_metrics(
        np.array(v_targets, dtype=np.int64), np.array(v_probs, dtype=np.float64)
    )
    val_auc = val_metrics["auc"]
    val_loss = v_loss_total / max(1, v_count)
    return val_loss, val_auc


class _EarlyStopState(TypedDict):
    """State for early stopping tracking."""

    best_val_auc: float
    best_state: dict[str, TensorProtocol] | None
    patience: int
    early_stopped: bool


def _run_training_loop(
    *,
    components: _TrainComponents,
    splits: _SplitsProtocol,
    cfg: LSTMConfig,
    device: str,
    output_dir: Path,
    progress: ProgressCallback | None,
    sequence_length: int,
) -> _EarlyStopState:
    """Run the training loop with early stopping."""
    torch_mod = _import_torch()
    batch_size = int(cfg["batch_size"])
    max_patience = int(cfg["early_stopping_patience"])
    n_epochs = int(cfg["n_epochs"])

    state: _EarlyStopState = {
        "best_val_auc": 0.0,
        "best_state": None,
        "patience": 0,
        "early_stopped": False,
    }

    model = components["model"]
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Warmup schedule
    warmup_epochs = 3
    lr_scale = 1.0

    for epoch in range(1, n_epochs + 1):
        warmup_scale = 1.0 if epoch > warmup_epochs else float(epoch) / float(warmup_epochs)
        train_scale = float(warmup_scale) * float(lr_scale)

        train_loss = _train_one_epoch(
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

        val_loss, val_auc = _validate_model(
            model=model,
            loss_fn=components["loss_fn"],
            x_val=splits.x_val,
            y_val=splits.y_val,
            batch_size=batch_size,
            device=device,
            sequence_length=sequence_length,
        )

        if progress is not None:
            prog: TrainProgress = {
                "round": epoch,
                "total_rounds": n_epochs,
                "train_loss": train_loss,
                "train_auc": 0.0,
                "val_loss": float(val_loss),
                "val_auc": val_auc,
            }
            progress(prog)

        # Save last checkpoint
        torch_mod.save(model.state_dict(), str(ckpt_dir / "last.pt"))

        # Check for improvement
        if val_auc > state["best_val_auc"]:
            state["best_val_auc"] = val_auc
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


def _finalize_metrics(
    *,
    model: TrainableModel,
    device: str,
    splits: _SplitsProtocol,
    sequence_length: int,
) -> tuple[EvalMetrics, EvalMetrics, EvalMetrics]:
    """Compute final metrics on train/val/test splits."""
    torch_mod = _import_torch()
    nn_mod = __import__("torch.nn", fromlist=["Softmax"])
    softmax_ctor: _SoftmaxCtor = nn_mod.Softmax
    softmax = softmax_ctor(dim=1)

    def _predict_prob(x: NDArray[np.float64]) -> NDArray[np.float64]:
        x_seq: NDArray[np.float64] = _reshape_to_sequence(x, sequence_length)
        with torch_mod.no_grad():
            xb: TensorProtocol = torch_mod.tensor(x_seq, dtype=torch_mod.float32)
            if device == "cuda":
                xb = xb.cuda()
            logits: TensorProtocol = model(xb)
            softmax_out: TensorProtocol = softmax(logits)
            proba: NDArray[np.float64] = softmax_out.detach().cpu().numpy().astype(np.float64)
            return proba[:, 1]

    train = compute_all_metrics(splits.y_train, _predict_prob(splits.x_train))
    val = compute_all_metrics(splits.y_val, _predict_prob(splits.x_val))
    test = compute_all_metrics(splits.y_test, _predict_prob(splits.x_test))
    return train, val, test


class LSTMBackend(ClassifierBackend):
    """LSTM backend for tabular binary classification."""

    def backend_name(self) -> BackendName:
        return "lstm"

    def capabilities(self) -> BackendCapabilities:
        return LSTM_CAPABILITIES

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> PreparedClassifier:
        """Prepare a minimal LSTM model for inference.

        Uses default sequence_length = n_features (each feature as one timestep).
        """
        default_sequence_length = n_features
        features_per_step = 1  # One feature per timestep
        model = _build_model(
            input_size=features_per_step,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            device="cpu",
        )
        return _LSTMPrepared(model, default_sequence_length)

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: ProgressCallback | None,
    ) -> TrainOutcome:
        """Train LSTM model on tabular data."""
        if not _is_lstm_config(config):
            raise RuntimeError("LSTMBackend requires LSTMConfig")

        cfg = config
        device = resolve_device(cfg["device"])
        precision = resolve_precision(cfg["precision"], device)

        raw_splits = stratified_split(
            x_features,
            y_labels,
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            test_ratio=cfg["test_ratio"],
            random_state=cfg["random_state"],
        )

        # Preprocess features (outlier capping, imputation, normalization)
        splits = preprocess_data_splits(raw_splits)
        n_features = int(splits.x_train.shape[1])
        sequence_length = int(cfg["sequence_length"])

        # Calculate features per timestep for LSTM input size
        features_per_step = (n_features + sequence_length - 1) // sequence_length

        components = _prepare_components(
            y_train=splits.y_train,
            cfg=cfg,
            device=device,
            precision=precision,
            features_per_step=features_per_step,
        )

        state = _run_training_loop(
            components=components,
            splits=splits,
            cfg=cfg,
            device=device,
            output_dir=output_dir,
            progress=progress,
            sequence_length=sequence_length,
        )

        # Restore best model
        model = components["model"]
        best_state = state["best_state"]
        if best_state is None:
            raise RuntimeError("Training completed with no best state; check n_epochs >= 1")
        model.load_state_dict(best_state)

        # Final metrics
        train_metrics, val_metrics, test_metrics = _finalize_metrics(
            model=model, device=device, splits=splits, sequence_length=sequence_length
        )

        # Save final model
        torch_mod = _import_torch()
        final_path = output_dir / "lstm_final.pt"
        torch_mod.save(model.state_dict(), str(final_path))

        return TrainOutcome(
            model_path=str(final_path),
            model_id="lstm",
            samples_total=splits.n_total,
            samples_train=splits.n_train,
            samples_val=splits.n_val,
            samples_test=splits.n_test,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_val_auc=state["best_val_auc"],
            best_round=0,
            total_rounds=int(cfg["n_epochs"]),
            early_stopped=state["early_stopped"],
            config=cfg,
            feature_importances=[],
            scale_pos_weight_computed=components["scale_pos_weight_computed"],
        )

    def evaluate(
        self,
        *,
        model: PreparedClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> EvalMetrics:
        """Evaluate LSTM model."""
        proba = model.predict_proba(x)
        return compute_all_metrics(y, proba[:, 1])

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        raise RuntimeError("LSTMBackend.save not supported; use TrainOutcome.model_path.")

    def load(self, *, path: str) -> PreparedClassifier:
        raise RuntimeError("LSTMBackend.load not supported; restore performed by pipeline.")

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        return None


def create_lstm_backend() -> LSTMBackend:
    """Create LSTM backend instance."""
    return LSTMBackend()


__all__ = ["LSTM_CAPABILITIES", "LSTMBackend", "create_lstm_backend"]
