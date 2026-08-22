"""Training internals for the LSTM classifier backend."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
from covenant_ml.backends.protocol import (
    ProgressCallback,
)
from covenant_ml.metrics import compute_all_metrics
from covenant_ml.types import (
    EvalMetrics,
    LSTMConfig,
    TrainProgress,
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

from covenant_nn.backends.lstm.backend_protocols import (
    _LinearLayerProto,
    _LSTMLayerProto,
    _NNLinearCtor,
    _NNLSTMCtor,
    _SequenceTensorProto,
    _SoftmaxCtor,
    _SplitsProtocol,
)

from .sequences import reshape_flat_to_pseudo_sequences

LSTM_STATE_PREFIX = "lstm."
FC_STATE_PREFIX = "fc."

_log = get_logger(__name__)


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
        """Return combined state dictionary keyed by the published prefixes."""
        combined: dict[str, TensorProtocol] = {}
        for k, v in self._lstm.state_dict().items():
            combined[f"{LSTM_STATE_PREFIX}{k}"] = v
        for k, v in self._fc.state_dict().items():
            combined[f"{FC_STATE_PREFIX}{k}"] = v
        return combined

    def load_state_dict(self, state_dict: dict[str, TensorProtocol]) -> None:
        """Load a state dictionary written by :meth:`state_dict`."""
        lstm_state: dict[str, TensorProtocol] = {}
        fc_state: dict[str, TensorProtocol] = {}
        for k, v in state_dict.items():
            if k.startswith(LSTM_STATE_PREFIX):
                lstm_state[k[len(LSTM_STATE_PREFIX) :]] = v
            elif k.startswith(FC_STATE_PREFIX):
                fc_state[k[len(FC_STATE_PREFIX) :]] = v
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
