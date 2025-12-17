"""Strict MLP backend with pluggable optimizer and CUDA autocast.

Implements:
- Device + precision resolution via platform_ml
- AdamW/Adam/SGD optimizer choice
- Stratified train/val/test splits
- Early stopping on val AUC with checkpoints (last/best/final)
- Strict Protocols/TypedDict usage (no Any/casts/ignores)
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Final, Protocol, TypedDict, TypeGuard

import numpy as np
from numpy.typing import NDArray
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

from ...metrics import compute_all_metrics
from ...trainer import normalize_data_splits, stratified_split
from ...types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    MLPConfig,
    TrainOutcome,
    TrainProgress,
)
from ..protocol import BackendCapabilities, ClassifierBackend, PreparedClassifier, ProgressCallback

_log = get_logger(__name__)


class _SplitsProtocol(Protocol):
    """Protocol for data splits (DataSplits or NormalizedDataSplits)."""

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


def _is_mlp_config(cfg: ClassifierTrainConfig) -> TypeGuard[MLPConfig]:
    """Check if config is MLPConfig by looking for MLP-specific keys."""
    return isinstance(cfg, dict) and "hidden_sizes" in cfg


class _OptimizerProto(Protocol):
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...


class _OptimizerCtor(Protocol):
    def __call__(self, params: TensorIterable, lr: float) -> _OptimizerProto: ...


class _LossProto(Protocol):
    def __call__(self, logits: TensorProtocol, targets: TensorProtocol) -> TensorProtocol: ...


class _WeightedLossCtor(Protocol):
    def __call__(self, weight: TensorProtocol) -> _LossProto: ...


class _LossCtor(Protocol):
    def __call__(self) -> _LossProto: ...


class _AutocastFactory(Protocol):
    def __call__(
        self, *, device_type: str, dtype: DTypeProtocol
    ) -> AbstractContextManager[None]: ...


class _GradScalerProto(Protocol):
    def scale(self, loss: TensorProtocol) -> TensorProtocol: ...
    def step(self, optimizer: _OptimizerProto) -> None: ...
    def update(self) -> None: ...


class _NNLinearCtor(Protocol):
    def __call__(self, in_features: int, out_features: int) -> TrainableModel: ...


class _NNBatchNorm1dCtor(Protocol):
    def __call__(self, num_features: int) -> TrainableModel: ...


class _NNReLUCtor(Protocol):
    def __call__(self) -> TrainableModel: ...


class _NNDropoutCtor(Protocol):
    def __call__(self, p: float) -> TrainableModel: ...


class _NNSequentialCtor(Protocol):
    def __call__(self, *modules: TrainableModel) -> TrainableModel: ...


class _SoftmaxCtor(Protocol):
    def __call__(self, *, dim: int) -> TrainableModel: ...


class _TensorCtor(Protocol):
    def __call__(
        self, data: NDArray[np.float64] | NDArray[np.int64], dtype: DTypeProtocol
    ) -> TensorProtocol: ...


class _NoGradFactory(Protocol):
    def __call__(self) -> AbstractContextManager[None]: ...


class _CudnnConfigProto(Protocol):
    """Protocol for torch.backends.cudnn config we rely on."""

    deterministic: bool
    benchmark: bool


class _EnableGradFactory(Protocol):
    """Protocol for torch.enable_grad context manager factory."""

    def __call__(self) -> AbstractContextManager[None]: ...


class _MLPPrepared:
    """Prepared MLP model for inference and gradient computation.

    Implements both PredictorProtocol and GradientModelProtocol from
    platform_ml.explainers.protocol for use with feature importance explainers.
    """

    def __init__(self, model: TrainableModel) -> None:
        self._model = model

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return class probabilities for input samples.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Class probabilities with shape (n_samples, n_classes).
        """
        torch = _import_torch()
        tensor: _TensorCtor = torch.tensor
        no_grad: _NoGradFactory = torch.no_grad
        nn_mod = __import__("torch.nn", fromlist=["Softmax"])
        softmax: _SoftmaxCtor = nn_mod.Softmax
        m = self._model
        m.eval()
        with no_grad():
            xt = tensor(x, dtype=torch.float32)
            logits = m(xt)
            sm = softmax(dim=1)
            proba = sm(logits).cpu().numpy()
        return proba.astype(np.float64)

    def compute_gradients(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Compute gradients of output w.r.t. input features.

        Computes d(output[target_class]) / d(input) for each sample.
        Used by gradient-based explainers (GradientExplainer, IntegratedGradientsExplainer).

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

        # Create input tensor and enable gradients
        x_tensor: TensorProtocol = tensor(x, dtype=float32)
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

        # Extract gradients from input tensor
        # Note: grad is always populated since requires_grad=True and backward() was called
        grad_tensor = x_tensor.grad
        assert grad_tensor is not None, "Gradient tensor should not be None after backward()"
        grad_cpu: TensorProtocol = grad_tensor.cpu()
        grad_numpy = grad_cpu.numpy()
        gradients: NDArray[np.float64] = grad_numpy.astype(np.float64)

        return gradients


MLP_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": True,
    "supports_early_stopping": True,
    "supports_feature_importance": False,
    "model_format": "pt",
}


def _build_model(n_in: int, hidden: tuple[int, ...], dropout: float, device: str) -> TrainableModel:
    nn_mod = __import__(
        "torch.nn",
        fromlist=[
            "Linear",
            "BatchNorm1d",
            "ReLU",
            "Dropout",
            "Sequential",
        ],
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
    parts.append(linear(in_f, 2))
    model = sequential(*parts)
    if device == "cuda":
        # to("cuda") exists at runtime; TrainableModel Protocol does not include it intentionally
        _ = model.to("cuda")
    return model


def _get_optimizer(name: str, params: TensorIterable, lr: float) -> _OptimizerProto:
    optim = __import__("torch.optim", fromlist=["AdamW", "Adam", "SGD"])
    sym_map: Final[dict[str, str]] = {"adamw": "AdamW", "adam": "Adam", "sgd": "SGD"}
    ctor: _OptimizerCtor = getattr(optim, sym_map[name])
    return ctor(params, lr=float(lr))


class _TrainComponents(TypedDict):
    """Components needed for training."""

    model: TrainableModel
    optimizer: _OptimizerProto
    loss_fn: _LossProto
    autocast: _AutocastFactory
    scaler: _GradScalerProto | None
    scale_pos_weight_computed: float


def _compute_class_weight(y_train: NDArray[np.int64]) -> float:
    """Compute scale_pos_weight from training labels.

    Returns:
        scale_pos_weight = n_negative / n_positive

    Raises:
        ValueError: If no positive samples exist
    """
    pos_mask: NDArray[np.bool_] = y_train == 1
    neg_mask: NDArray[np.bool_] = y_train == 0
    n_positive = int(np.count_nonzero(pos_mask))
    n_negative = int(np.count_nonzero(neg_mask))
    if n_positive == 0:
        raise ValueError("Training set has no positive samples")
    computed = float(n_negative) / float(n_positive)
    _log.info(
        "Auto-calculated scale_pos_weight",
        extra={
            "n_positive": n_positive,
            "n_negative": n_negative,
            "scale_pos_weight": computed,
        },
    )
    return computed


def _prepare_components(
    *,
    n_features: int,
    y_train: NDArray[np.int64],
    cfg: MLPConfig,
    device: str,
    precision: str,
) -> _TrainComponents:
    """Build model, optimizer, loss and AMP helpers for training.

    Also seeds PyTorch RNG for reproducibility and configures deterministic
    behavior on CUDA when feasible.
    """
    torch = _import_torch()
    nn_mod = __import__("torch.nn", fromlist=["CrossEntropyLoss"])
    loss_ctor: _WeightedLossCtor = nn_mod.CrossEntropyLoss

    # Compute class weights for imbalanced data
    scale_pos_weight = _compute_class_weight(y_train)
    # CrossEntropyLoss weight tensor: [class_0_weight, class_1_weight]
    class_weights = torch.tensor([1.0, scale_pos_weight], dtype=torch.float32)
    if device == "cuda":
        class_weights = class_weights.cuda()

    # Seed PyTorch RNG deterministically for reproducible training runs
    set_manual_seed(int(cfg["random_state"]))

    # Enable deterministic CUDA algorithms when using GPU
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

    model = _build_model(
        n_in=int(n_features),
        hidden=cfg["hidden_sizes"],
        dropout=float(cfg["dropout"]),
        device=device,
    )
    opt = _get_optimizer(cfg["optimizer"], model.parameters(), float(cfg["learning_rate"]))
    return {
        "model": model,
        "optimizer": opt,
        "loss_fn": loss_ctor(weight=class_weights),
        "autocast": autocast,
        "scaler": scaler,
        "scale_pos_weight_computed": scale_pos_weight,
    }


def _train_one_epoch(
    *,
    model: TrainableModel,
    optimizer: _OptimizerProto,
    loss_fn: _LossProto,
    autocast: _AutocastFactory,
    scaler: _GradScalerProto | None,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    batch_size: int,
    device: str,
    train_scale: float,
) -> float:
    """Train model for one epoch and return average loss."""
    torch = _import_torch()
    tensor: _TensorCtor = torch.tensor
    float32: DTypeProtocol = torch.float32
    long_dtype: DTypeProtocol = torch.long
    fp16: DTypeProtocol = torch.float16

    model.train()
    total_loss = 0.0
    total_count = 0
    n_train: int = int(x_train.shape[0])

    for start in range(0, n_train, batch_size):
        end: int = min(n_train, start + batch_size)
        batch_len: int = end - start
        xb = tensor(x_train[start:end], dtype=float32)
        yb = tensor(y_train[start:end], dtype=long_dtype)
        if device == "cuda":
            xb = xb.cuda()
            yb = yb.cuda()

        optimizer.zero_grad()
        if scaler is not None:
            with autocast(device_type="cuda", dtype=fp16):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            # Apply training scale by scaling loss (equivalent to scaling LR)
            scaled = scaler.scale(loss * float(train_scale))
            scaled.backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with nullcontext():
                logits = model(xb)
                loss = loss_fn(logits, yb)
                # Apply training scale by scaling loss (equivalent to scaling LR)
                (loss * float(train_scale)).backward()
                optimizer.step()

        total_loss += float(loss.item()) * batch_len
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
) -> tuple[float, float]:
    """Validate model and return (val_loss, val_auc)."""
    torch = _import_torch()
    tensor: _TensorCtor = torch.tensor
    no_grad: _NoGradFactory = torch.no_grad
    float32: DTypeProtocol = torch.float32
    long_dtype: DTypeProtocol = torch.long

    model.eval()
    v_probs: list[float] = []
    v_targets: list[int] = []
    v_loss_total = 0.0
    v_count = 0
    n_val: int = int(x_val.shape[0])

    with no_grad():
        for start in range(0, n_val, batch_size):
            end: int = min(n_val, start + batch_size)
            batch_len: int = end - start
            xb = tensor(x_val[start:end], dtype=float32)
            yb = tensor(y_val[start:end], dtype=long_dtype)
            if device == "cuda":
                xb = xb.cuda()
                yb = yb.cuda()

            logits = model(xb)
            v_loss_total += float(loss_fn(logits, yb).item()) * batch_len
            v_count += batch_len

            nn_mod = __import__("torch.nn", fromlist=["Softmax"])
            sm: _SoftmaxCtor = nn_mod.Softmax
            probs: NDArray[np.float64] = sm(dim=1)(logits).detach().cpu().numpy().astype(np.float64)
            prob_col: NDArray[np.float64] = probs[:, 1]
            v_probs.extend([float(v) for v in prob_col.flat])
            # Get targets from original numpy array slice
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
    cfg: MLPConfig,
    device: str,
    output_dir: Path,
    progress: ProgressCallback | None,
) -> _EarlyStopState:
    """Run the training loop with early stopping."""
    torch = _import_torch()

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

    # Tiny linear warmup over first few epochs to stabilize updates
    warmup_epochs: int = 3
    lr_scale: float = 1.0

    for epoch in range(1, n_epochs + 1):
        warmup_scale = 1.0 if epoch > warmup_epochs else float(epoch) / float(warmup_epochs)
        train_scale = float(warmup_scale) * float(lr_scale)
        train_loss = _train_one_epoch(
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

        val_loss, val_auc = _validate_model(
            model=model,
            loss_fn=components["loss_fn"],
            x_val=splits.x_val,
            y_val=splits.y_val,
            batch_size=batch_size,
            device=device,
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
        torch.save(model.state_dict(), str(ckpt_dir / "last.pt"))

        # Check for improvement
        if val_auc > state["best_val_auc"]:
            state["best_val_auc"] = val_auc
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


def _finalize_metrics(
    *,
    model: TrainableModel,
    device: str,
    splits: _SplitsProtocol,
) -> tuple[EvalMetrics, EvalMetrics, EvalMetrics]:
    """Compute final metrics on train/val/test splits for the given model."""
    torch = _import_torch()
    nn_mod = __import__("torch.nn", fromlist=["Softmax"])
    softmax: _SoftmaxCtor = nn_mod.Softmax
    tensor: _TensorCtor = torch.tensor
    float32: DTypeProtocol = torch.float32
    no_grad: _NoGradFactory = torch.no_grad

    def _predict_prob(x: NDArray[np.float64]) -> NDArray[np.float64]:
        with no_grad():
            xb = tensor(x, dtype=float32)
            if device == "cuda":
                xb = xb.cuda()
            logits = model(xb)
            return softmax(dim=1)(logits).detach().cpu().numpy().astype(np.float64)[:, 1]

    train = compute_all_metrics(splits.y_train, _predict_prob(splits.x_train))
    val = compute_all_metrics(splits.y_val, _predict_prob(splits.x_val))
    test = compute_all_metrics(splits.y_test, _predict_prob(splits.x_test))
    return train, val, test


class MLPBackend(ClassifierBackend):
    def backend_name(self) -> BackendName:
        return "mlp"

    def capabilities(self) -> BackendCapabilities:
        return MLP_CAPABILITIES

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> PreparedClassifier:
        nn_mod = __import__("torch.nn", fromlist=["Linear", "Sequential"])
        linear: _NNLinearCtor = nn_mod.Linear
        seq: _NNSequentialCtor = nn_mod.Sequential
        return _MLPPrepared(seq(linear(int(n_features), int(n_classes))))

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
        if not _is_mlp_config(config):
            raise RuntimeError("MLPBackend requires MLPConfig (found TrainConfig)")
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

        # Normalize features using training data statistics only (prevents data leakage)
        # This is critical for MLP performance - neural networks require normalized inputs
        splits = normalize_data_splits(raw_splits)

        components = _prepare_components(
            n_features=int(splits.x_train.shape[1]),
            y_train=splits.y_train,
            cfg=cfg,
            device=device,
            precision=precision,
        )

        state = _run_training_loop(
            components=components,
            splits=splits,
            cfg=cfg,
            device=device,
            output_dir=output_dir,
            progress=progress,
        )

        # Restore best model (best_state is always set after first epoch since val_auc > 0.0)
        model = components["model"]
        best_state = state["best_state"]
        if best_state is None:
            raise RuntimeError("Training completed with no best state; check n_epochs >= 1")
        model.load_state_dict(best_state)

        # Final metrics
        train_metrics, val_metrics, test_metrics = _finalize_metrics(
            model=model, device=device, splits=splits
        )

        # Save final model
        torch = _import_torch()
        final_path = output_dir / "mlp_final.pt"
        torch.save(model.state_dict(), str(final_path))

        return TrainOutcome(
            model_path=str(final_path),
            model_id="mlp",
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
        proba = model.predict_proba(x)
        return compute_all_metrics(y, proba[:, 1])

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        raise RuntimeError("MLPBackend.save not supported; use TrainOutcome.model_path.")

    def load(self, *, path: str) -> PreparedClassifier:
        raise RuntimeError("MLPBackend.load not supported; restore performed by pipeline.")

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        return None


def create_mlp_backend() -> MLPBackend:
    return MLPBackend()


__all__ = ["MLP_CAPABILITIES", "MLPBackend", "create_mlp_backend"]
