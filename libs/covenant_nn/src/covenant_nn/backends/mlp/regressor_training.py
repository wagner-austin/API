"""Training internals for the MLP regressor backend."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
from covenant_ml.metrics_regression import compute_all_regression_metrics
from covenant_ml.preprocessing import AutoPreprocessor
from covenant_ml.trainer import RegressionDataSplits
from covenant_ml.types import MLPConfig
from covenant_ml.types_regression import (
    RegressionMetrics,
    RegressionTrainProgress,
)
from numpy.typing import NDArray
from platform_core.logging import get_logger
from platform_ml.torch_types import (
    DTypeProtocol,
    TensorIterable,
    TensorProtocol,
    TrainableModel,
    _import_torch,
    set_manual_seed,
)

from covenant_nn.backends import _amp
from covenant_nn.backends.mlp.regressor_protocols import (
    _LossCtor,
    _LossProto,
    _NNBatchNorm1dCtor,
    _NNDropoutCtor,
    _NNLinearCtor,
    _NNReLUCtor,
    _NNSequentialCtor,
    _NoGradFactory,
    _OptimizerCtor,
    _OptimizerProto,
    _TensorCtor,
)

_log = get_logger(__name__)


def _build_regressor_model(
    n_in: int, hidden: tuple[int, ...], dropout: float, device: str
) -> TrainableModel:
    """Build MLP model with 1 output neuron for regression.

    Architecture: [Linear-BN-ReLU-Dropout] x N then Linear(1)

    Args:
        n_in: Number of input features.
        hidden: Tuple of hidden layer widths.
        dropout: Dropout probability (0.0 to disable).
        device: Target device ('cpu' or 'cuda').

    Returns:
        A TrainableModel ready for training.
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
    parts: list[TrainableModel] = []
    in_f = int(n_in)
    for width in hidden:
        parts.append(linear(in_f, int(width)))
        parts.append(bn(int(width)))
        parts.append(relu())
        if dropout > 0.0:
            parts.append(drop(dropout))
        in_f = int(width)
    parts.append(linear(in_f, 1))  # Single output for regression
    model = sequential(*parts)
    _ = model.to(device)
    return model


def _get_optimizer(name: str, params: TensorIterable, lr: float) -> _OptimizerProto:
    """Create optimizer by name.

    Args:
        name: Optimizer name ('adamw', 'adam', or 'sgd').
        params: Model parameters to optimize.
        lr: Learning rate.

    Returns:
        An optimizer instance.
    """
    optim = __import__("torch.optim", fromlist=["AdamW", "Adam", "SGD"])
    sym_map: Final[dict[str, str]] = {"adamw": "AdamW", "adam": "Adam", "sgd": "SGD"}
    ctor: _OptimizerCtor = getattr(optim, sym_map[name])
    return ctor(params, lr=float(lr))


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
    # y_train is unused in fit() — pass zero array to satisfy type signature
    dummy_y: NDArray[np.int64] = np.zeros(splits.n_train, dtype=np.int64)
    state = preprocessor.fit(splits.x_train, dummy_y)

    _log.info(
        "Preprocessing regression splits",
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
    """Components assembled for regression training."""

    model: TrainableModel
    optimizer: _OptimizerProto
    loss_fn: _LossProto
    scaler: _amp.GradScalerProto | None


def _prepare_regression_components(
    *,
    n_features: int,
    cfg: MLPConfig,
    device: str,
    precision: str,
) -> _RegressorTrainComponents:
    """Build model, optimizer, MSELoss, and AMP helpers for regression.

    Seeds PyTorch RNG and configures deterministic CUDA when on GPU.

    Args:
        n_features: Number of input features.
        cfg: MLP training configuration.
        device: Resolved device string ('cpu' or 'cuda').
        precision: Resolved precision string ('fp32' or 'fp16').

    Returns:
        _RegressorTrainComponents with all training objects.
    """
    _ = _import_torch()  # Ensure torch is loaded before using nn
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
        n_in=int(n_features),
        hidden=cfg["hidden_sizes"],
        dropout=float(cfg["dropout"]),
        device=device,
    )
    opt = _get_optimizer(cfg["optimizer"], model.parameters(), float(cfg["learning_rate"]))
    return {
        "model": model,
        "optimizer": opt,
        "loss_fn": loss_ctor(),
        "scaler": scaler,
    }


# =============================================================================
# Training loop
# =============================================================================


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
) -> float:
    """Train model for one epoch and return average MSE loss.

    Args:
        model: The MLP model to train.
        optimizer: Optimizer instance.
        loss_fn: MSELoss instance.
        autocast: AMP autocast factory.
        scaler: Optional GradScaler for mixed precision.
        x_train: Training features.
        y_train: Training targets (float64).
        batch_size: Mini-batch size.
        device: Device string ('cpu' or 'cuda').
        train_scale: Loss scaling factor (warmup + LR decay).

    Returns:
        Average MSE loss over the epoch.
    """
    torch = _import_torch()
    tensor: _TensorCtor = torch.tensor
    float32: DTypeProtocol = torch.float32
    fp16: DTypeProtocol = torch.float16

    model.train()
    total_loss = 0.0
    total_count = 0
    n_train: int = int(x_train.shape[0])

    for start in range(0, n_train, batch_size):
        end: int = min(n_train, start + batch_size)
        batch_len: int = end - start
        xb = tensor(x_train[start:end], dtype=float32)
        yb = tensor(y_train[start:end], dtype=float32)
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        # One forward pass, in whichever context this run calls for -- it was
        # written twice, identically, so the fp16 copy only ever ran on a GPU.
        with _amp.amp_context(scaler, fp16):
            logits = model(xb)
            preds: TensorProtocol = logits.select(1, 0)
            loss = loss_fn(preds, yb)
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
) -> tuple[float, float]:
    """Validate model and return (val_loss, val_rmse).

    Args:
        model: The trained MLP model.
        loss_fn: MSELoss instance.
        x_val: Validation features.
        y_val: Validation targets (float64).
        batch_size: Mini-batch size.
        device: Device string.

    Returns:
        Tuple of (average MSE loss, RMSE on validation set).
    """
    torch = _import_torch()
    tensor: _TensorCtor = torch.tensor
    no_grad: _NoGradFactory = torch.no_grad
    float32: DTypeProtocol = torch.float32

    model.eval()
    v_preds: list[float] = []
    v_targets: list[float] = []
    v_loss_total = 0.0
    v_count = 0
    n_val: int = int(x_val.shape[0])

    with no_grad():
        for start in range(0, n_val, batch_size):
            end: int = min(n_val, start + batch_size)
            batch_len: int = end - start
            xb = tensor(x_val[start:end], dtype=float32)
            yb = tensor(y_val[start:end], dtype=float32)
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
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
    cfg: MLPConfig,
    device: str,
    output_dir: Path,
    progress: Callable[[RegressionTrainProgress], None] | None,
) -> _EarlyStopState:
    """Run the regression training loop with early stopping on val RMSE.

    Args:
        components: Model, optimizer, loss, AMP helpers.
        splits: Preprocessed regression data splits.
        cfg: MLP training configuration.
        device: Resolved device string.
        output_dir: Directory for checkpoints.
        progress: Optional callback for progress updates.

    Returns:
        _EarlyStopState with best model state and training metadata.
    """
    torch = _import_torch()

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

    # Tiny linear warmup over first few epochs to stabilize updates
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
        )

        _val_loss, val_rmse = _validate_regression_model(
            model=model,
            loss_fn=components["loss_fn"],
            x_val=splits.x_val,
            y_val=splits.y_val,
            batch_size=batch_size,
            device=device,
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
        torch.save(model.state_dict(), str(ckpt_dir / "last.pt"))

        # Check for improvement (RMSE: lower is better)
        if val_rmse < state["best_val_rmse"]:
            state["best_val_rmse"] = val_rmse
            state["best_round"] = epoch
            state["best_state"] = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), str(ckpt_dir / "best.pt"))
            state["patience"] = 0
        else:
            state["patience"] += 1
            # Reduce-on-plateau: shrink effective LR when patience accrues
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
) -> tuple[RegressionMetrics, RegressionMetrics, RegressionMetrics]:
    """Compute final regression metrics on train/val/test splits.

    Args:
        model: Trained MLP model with best weights loaded.
        device: Device string.
        splits: Preprocessed regression data splits.

    Returns:
        Tuple of (train_metrics, val_metrics, test_metrics).
    """
    torch = _import_torch()
    tensor: _TensorCtor = torch.tensor
    float32: DTypeProtocol = torch.float32
    no_grad: _NoGradFactory = torch.no_grad

    def _predict(x: NDArray[np.float64]) -> NDArray[np.float64]:
        with no_grad():
            xb = tensor(x, dtype=float32)
            xb = xb.to(device)
            logits = model(xb)
            preds: TensorProtocol = logits.select(1, 0)
            return preds.detach().cpu().numpy().astype(np.float64)

    train = compute_all_regression_metrics(splits.y_train, _predict(splits.x_train))
    val = compute_all_regression_metrics(splits.y_val, _predict(splits.x_val))
    test = compute_all_regression_metrics(splits.y_test, _predict(splits.x_test))
    return train, val, test


# =============================================================================
# Prepared model for inference
# =============================================================================
