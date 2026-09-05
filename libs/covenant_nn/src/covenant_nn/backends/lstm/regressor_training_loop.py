"""Epoch loop, validation, and early stopping for LSTM regression training."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import numpy as np
from covenant_ml.trainer import RegressionDataSplits
from covenant_ml.types import LSTMConfig
from covenant_ml.types_regression import (
    RegressionTrainProgress,
)
from numpy.typing import NDArray
from platform_ml.torch_types import (
    TensorProtocol,
    TrainableModel,
    _import_torch,
)

from covenant_nn.backends import _amp
from covenant_nn.backends.lstm.regressor_protocols import (
    _LossProto,
    _OptimizerProto,
)

from .sequences import reshape_flat_to_pseudo_sequences


class _RegressorTrainComponents(TypedDict):
    """Components assembled for LSTM regression training."""

    model: TrainableModel
    optimizer: _OptimizerProto
    loss_fn: _LossProto
    scaler: _amp.GradScalerProto | None


def _reshape_to_sequence(x: NDArray[np.float64], sequence_length: int) -> NDArray[np.float64]:
    """Reshape tabular data to sequence format."""
    return reshape_flat_to_pseudo_sequences(x, sequence_length)


def _train_one_epoch_regression(
    *,
    model: TrainableModel,
    optimizer: _OptimizerProto,
    loss_fn: _LossProto,
    scaler: _amp.GradScalerProto | None,
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
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        # One forward pass, in whichever context this run calls for -- it was
        # written twice, identically, so the fp16 copy only ever ran on a GPU.
        with _amp.amp_context(scaler, torch_mod.float16):
            logits: TensorProtocol = model(xb)
            preds: TensorProtocol = logits.select(1, 0)
            loss: TensorProtocol = loss_fn(preds, yb)
        _amp.backward_step(scaler=scaler, optimizer=optimizer, loss=loss, train_scale=train_scale)

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
            xb = xb.to(device)
            yb = yb.to(device)

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
    progress: Callable[[RegressionTrainProgress], None] | None,
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
