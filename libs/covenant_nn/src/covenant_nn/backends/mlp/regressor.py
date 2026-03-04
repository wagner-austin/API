"""MLP regressor backend for continuous target prediction.

Parallel to MLP classifier backend (backend.py). Key differences:
- Output layer: nn.Linear(last_hidden, 1) instead of 2
- Loss: nn.MSELoss() instead of nn.CrossEntropyLoss()
- No class weighting or scale_pos_weight
- No softmax — direct scalar output via select(1, 0)
- Early stopping on val RMSE (lower is better, not AUC higher is better)
- predict() returns 1D float64 array (not 2D class probabilities)
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Final, Protocol, TypedDict, TypeGuard

import numpy as np
from covenant_ml.backends.protocol import BackendCapabilities
from covenant_ml.backends.regressor_protocol import (
    PreparedRegressor,
    RegressorProgressCallback,
)
from covenant_ml.metrics import compute_all_regression_metrics
from covenant_ml.preprocessing import AutoPreprocessor
from covenant_ml.trainer import RegressionDataSplits, regression_split
from covenant_ml.types import (
    FeatureImportance,
    MLPConfig,
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
    narrow_json_to_int,
    require_float,
    require_int,
    require_list,
)
from platform_core.logging import get_logger
from platform_ml.device_selector import resolve_device, resolve_precision
from platform_ml.torch_types import (
    DTypeProtocol,
    TensorIterable,
    TensorProtocol,
    TrainableModel,
    _import_torch,
    set_manual_seed,
)

_log = get_logger(__name__)


# =============================================================================
# Protocols for PyTorch dynamic imports
# =============================================================================


class _OptimizerProto(Protocol):
    """Protocol for torch optimizer."""

    def zero_grad(self) -> None: ...
    def step(self) -> None: ...


class _OptimizerCtor(Protocol):
    """Protocol for torch optimizer constructor."""

    def __call__(self, params: TensorIterable, lr: float) -> _OptimizerProto: ...


class _LossProto(Protocol):
    """Protocol for a loss function (MSELoss)."""

    def __call__(self, input: TensorProtocol, target: TensorProtocol) -> TensorProtocol: ...


class _LossCtor(Protocol):
    """Protocol for MSELoss constructor (no arguments)."""

    def __call__(self) -> _LossProto: ...


class _AutocastFactory(Protocol):
    """Protocol for torch.amp.autocast factory."""

    def __call__(
        self, *, device_type: str, dtype: DTypeProtocol
    ) -> AbstractContextManager[None]: ...


class _GradScalerProto(Protocol):
    """Protocol for torch.amp.GradScaler."""

    def scale(self, loss: TensorProtocol) -> TensorProtocol: ...
    def step(self, optimizer: _OptimizerProto) -> None: ...
    def update(self) -> None: ...


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

    def __call__(self, *modules: TrainableModel) -> TrainableModel: ...


class _TensorCtor(Protocol):
    """Protocol for torch.tensor constructor (float64 input only for regression)."""

    def __call__(self, data: NDArray[np.float64], dtype: DTypeProtocol) -> TensorProtocol: ...


class _NoGradFactory(Protocol):
    """Protocol for torch.no_grad context manager factory."""

    def __call__(self) -> AbstractContextManager[None]: ...


class _CudnnConfigProto(Protocol):
    """Protocol for torch.backends.cudnn config."""

    deterministic: bool
    benchmark: bool


# =============================================================================
# Constants and type guard
# =============================================================================


MLP_REGRESSOR_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": True,
    "supports_early_stopping": True,
    "supports_feature_importance": False,
    "model_format": "pt",
}


def _is_mlp_config(cfg: RegressorTrainConfig) -> TypeGuard[MLPConfig]:
    """Check if config is MLPConfig by looking for MLP-specific keys.

    Args:
        cfg: Regressor training configuration to check.

    Returns:
        True if config contains hidden_sizes key (MLPConfig discriminator).
    """
    return isinstance(cfg, dict) and "hidden_sizes" in cfg


# =============================================================================
# Model metadata for save/load
# =============================================================================


class _MLPRegressorMeta(TypedDict, total=True):
    """Architecture metadata saved alongside .pt state dict.

    Contains the minimal parameters needed to reconstruct the MLP
    architecture before loading the state dict.

    Args:
        n_features: Number of input features the model was trained on.
        hidden_sizes: Hidden layer widths (JSON list, not tuple).
        dropout: Dropout probability used in the architecture.
    """

    n_features: int
    hidden_sizes: list[int]
    dropout: float


def _encode_mlp_regressor_meta(meta: _MLPRegressorMeta) -> str:
    """Encode MLP regressor metadata to JSON string.

    Args:
        meta: Metadata TypedDict to encode.

    Returns:
        JSON string representation.
    """
    return dump_json_str(
        {
            "n_features": meta["n_features"],
            "hidden_sizes": meta["hidden_sizes"],
            "dropout": meta["dropout"],
        }
    )


def _decode_mlp_regressor_meta(raw: JSONValue) -> _MLPRegressorMeta:
    """Decode and validate MLP regressor metadata from parsed JSON.

    Args:
        raw: Parsed JSON value (from load_json_str).

    Returns:
        Validated _MLPRegressorMeta TypedDict.

    Raises:
        JSONTypeError: If structure or types are invalid.
    """
    obj = narrow_json_to_dict(raw)
    n_features = require_int(obj, "n_features")
    hidden_raw = require_list(obj, "hidden_sizes")
    hidden_sizes: list[int] = []
    for item in hidden_raw:
        hidden_sizes.append(narrow_json_to_int(item))
    dropout = require_float(obj, "dropout")
    return {"n_features": n_features, "hidden_sizes": hidden_sizes, "dropout": dropout}


# =============================================================================
# Model building
# =============================================================================


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
    if device == "cuda":
        _ = model.to("cuda")
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
    autocast: _AutocastFactory
    scaler: _GradScalerProto | None


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

    if device == "cuda":
        backends_mod = __import__("torch.backends", fromlist=["cudnn"])
        cudnn: _CudnnConfigProto = backends_mod.cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False

    amp = __import__("torch.amp", fromlist=["autocast", "GradScaler"])
    autocast: _AutocastFactory = amp.autocast
    scaler: _GradScalerProto | None = None
    if device == "cuda" and precision != "fp32":
        grad_scaler: _GradScalerProto = amp.GradScaler()
        scaler = grad_scaler

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
        "autocast": autocast,
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
    autocast: _AutocastFactory,
    scaler: _GradScalerProto | None,
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
        if device == "cuda":
            xb = xb.cuda()
            yb = yb.cuda()

        optimizer.zero_grad()
        if scaler is not None:
            with autocast(device_type="cuda", dtype=fp16):
                logits = model(xb)
                preds: TensorProtocol = logits.select(1, 0)
                loss = loss_fn(preds, yb)
            scaled = scaler.scale(loss * float(train_scale))
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
            if device == "cuda":
                xb = xb.cuda()
                yb = yb.cuda()

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
    progress: RegressorProgressCallback | None,
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
            autocast=components["autocast"],
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

        train_rmse = float(np.sqrt(max(0.0, train_loss)))

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
            if device == "cuda":
                xb = xb.cuda()
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


class _MLPRegressorPrepared:
    """Prepared MLP regressor for inference.

    Returns 1D continuous predictions via select(1, 0) on the single
    output neuron. No softmax, no class probabilities.
    """

    def __init__(self, model: TrainableModel) -> None:
        self._model = model

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict continuous values for input samples.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Predicted values with shape (n_samples,).
        """
        torch = _import_torch()
        tensor: _TensorCtor = torch.tensor
        no_grad: _NoGradFactory = torch.no_grad
        float32: DTypeProtocol = torch.float32
        m = self._model
        m.eval()
        with no_grad():
            xt = tensor(x, dtype=float32)
            logits = m(xt)
            preds: TensorProtocol = logits.select(1, 0)
            return preds.cpu().numpy().astype(np.float64)


# =============================================================================
# Backend class
# =============================================================================


class MLPRegressorBackend:
    """MLP regressor backend for continuous target prediction.

    Implements the RegressorBackend protocol. Parallel to MLPBackend
    (classifier). Uses MSELoss, no class weights, output dim=1,
    early stopping on validation RMSE.
    """

    def backend_name(self) -> RegressorBackendName:
        """Return the backend identifier.

        Returns:
            The backend name literal 'mlp_reg'.
        """
        return "mlp_reg"

    def capabilities(self) -> BackendCapabilities:
        """Return capability flags.

        Returns:
            BackendCapabilities for MLP regressor.
        """
        return MLP_REGRESSOR_CAPABILITIES

    def prepare(
        self,
        *,
        n_features: int,
        feature_names: list[str] | None,
    ) -> PreparedRegressor:
        """Prepare a minimal regressor for inference.

        Creates a simple Linear(n_features, 1) model. Primarily used
        for testing the predict/evaluate paths without full training.

        Args:
            n_features: Number of input features.
            feature_names: Optional feature names (unused).

        Returns:
            A PreparedRegressor with a simple linear model.
        """
        nn_mod = __import__("torch.nn", fromlist=["Linear", "Sequential"])
        linear: _NNLinearCtor = nn_mod.Linear
        seq: _NNSequentialCtor = nn_mod.Sequential
        _ = feature_names
        return _MLPRegressorPrepared(seq(linear(int(n_features), 1)))

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
        """Train an MLP regressor with early stopping on val RMSE.

        Splits data, preprocesses features (outlier capping, imputation,
        z-score normalization), trains with MSELoss, and returns complete
        regression metrics.

        Args:
            x_features: Feature matrix (n_samples, n_features).
            y_targets: Continuous target values (n_samples,).
            feature_names: Optional feature names (unused by MLP).
            config: MLPConfig with training hyperparameters.
            output_dir: Directory for model checkpoints and final model.
            progress: Optional callback for training progress.

        Returns:
            RegressionTrainOutcome with metrics from all splits.

        Raises:
            RuntimeError: If config is not MLPConfig.
            RuntimeError: If training produces no best state (n_epochs=0).
        """
        if not _is_mlp_config(config):
            raise RuntimeError(
                "MLPRegressorBackend requires MLPConfig (found RegressorTrainConfig)"
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

        components = _prepare_regression_components(
            n_features=int(splits.x_train.shape[1]),
            cfg=cfg,
            device=device,
            precision=precision,
        )

        state = _run_regression_training_loop(
            components=components,
            splits=splits,
            cfg=cfg,
            device=device,
            output_dir=output_dir,
            progress=progress,
        )

        model = components["model"]
        best_state = state["best_state"]
        if best_state is None:
            raise RuntimeError("Training completed with no best state; check n_epochs >= 1")
        model.load_state_dict(best_state)

        train_metrics, val_metrics, test_metrics = _finalize_regression_metrics(
            model=model, device=device, splits=splits
        )

        torch = _import_torch()
        final_path = output_dir / "mlp_reg_final.pt"
        torch.save(model.state_dict(), str(final_path))

        meta_path = output_dir / "mlp_reg_final.json"
        meta: _MLPRegressorMeta = {
            "n_features": int(splits.x_train.shape[1]),
            "hidden_sizes": [int(h) for h in cfg["hidden_sizes"]],
            "dropout": float(cfg["dropout"]),
        }
        meta_path.write_text(_encode_mlp_regressor_meta(meta), encoding="utf-8")

        return RegressionTrainOutcome(
            model_path=str(final_path),
            model_id="mlp_reg",
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
            "MLPRegressorBackend.save not supported; use RegressionTrainOutcome.model_path."
        )

    def load(self, *, path: str) -> PreparedRegressor:
        """Load a trained MLP regressor from saved state dict and metadata.

        Expects a JSON metadata file alongside the .pt file at the same
        path with a .json extension. The metadata contains architecture
        parameters needed to reconstruct the model before loading weights.

        Args:
            path: Path to the saved .pt state dict file.

        Returns:
            A PreparedRegressor wrapping the loaded MLP model.

        Raises:
            FileNotFoundError: If the .pt or .json file does not exist.
            ValueError: If metadata JSON is invalid.
        """
        torch = _import_torch()
        pt_path = Path(path)
        meta_path = pt_path.with_suffix(".json")
        raw = load_json_str(meta_path.read_text(encoding="utf-8"))
        meta = _decode_mlp_regressor_meta(raw)

        model = _build_regressor_model(
            n_in=meta["n_features"],
            hidden=tuple(meta["hidden_sizes"]),
            dropout=meta["dropout"],
            device="cpu",
        )
        state_dict: dict[str, TensorProtocol] = torch.load(str(pt_path))
        model.load_state_dict(state_dict)
        model.eval()
        return _MLPRegressorPrepared(model)

    def get_feature_importances(
        self,
        *,
        model: PreparedRegressor,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Feature importances not supported for MLP.

        Args:
            model: A trained regressor (unused).
            feature_names: Feature names (unused).

        Returns:
            None (MLP has no native feature importance).
        """
        _ = model, feature_names
        return None


def create_mlp_regressor_backend() -> MLPRegressorBackend:
    """Create an MLP regressor backend instance.

    Returns:
        A new MLPRegressorBackend.
    """
    return MLPRegressorBackend()


__all__ = ["MLP_REGRESSOR_CAPABILITIES", "MLPRegressorBackend", "create_mlp_regressor_backend"]
