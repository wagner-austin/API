"""Strict MLP backend with pluggable optimizer and CUDA autocast.

Implements:
- Device + precision resolution via platform_ml
- AdamW/Adam/SGD optimizer choice
- Stratified train/val/test splits
- Early stopping on val AUC with checkpoints (last/best/final)
- Strict Protocols/TypedDict usage (no Any/casts/ignores)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeGuard

import numpy as np
from covenant_ml.backends.protocol import (
    BackendCapabilities,
    ClassifierBackend,
    PreparedClassifier,
)
from covenant_ml.metrics import compute_all_metrics
from covenant_ml.optimizer.search_spaces import (
    make_mlp_default_space,
    make_mlp_focused_space,
)
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SearchSpace,
)
from covenant_ml.trainer import preprocess_data_splits, stratified_split
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    MLPConfig,
    TrainOutcome,
    TrainProgress,
)
from numpy.typing import NDArray
from platform_core.logging import get_logger
from platform_ml.device_selector import resolve_device, resolve_precision
from platform_ml.torch_types import (
    DTypeProtocol,
    TensorProtocol,
    TrainableModel,
    _import_torch,
)

from covenant_nn.backends.mlp.backend_protocols import (
    _EnableGradFactory,
    _NNLinearCtor,
    _NNSequentialCtor,
    _NoGradFactory,
    _SoftmaxCtor,
    _TensorCtor,
)
from covenant_nn.backends.mlp.backend_training import (
    _build_model,
    _finalize_metrics,
    _prepare_components,
    _run_training_loop,
)

_log = get_logger(__name__)


def _is_mlp_config(cfg: ClassifierTrainConfig) -> TypeGuard[MLPConfig]:
    """Check if config is MLPConfig by looking for MLP-specific keys."""
    return isinstance(cfg, dict) and "hidden_sizes" in cfg


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


class MLPBackend(ClassifierBackend):
    def get_default_search_space(self) -> SearchSpace:
        """Return the backend's default hyperparameter search space.

        Returns:
            The mlp default SearchSpace.
        """
        return make_mlp_default_space()

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
            The mlp focused SearchSpace.
        """
        return make_mlp_focused_space(
            best_n_layers=best_int_params["n_layers"],
            best_hidden_size=best_int_params["hidden_size"],
            best_learning_rate=best_float_params["learning_rate"],
        )

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
        progress: Callable[[TrainProgress], None] | None,
        groups: NDArray[np.int64] | None = None,
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
            groups=groups,
        )

        # Preprocess features (outlier capping, imputation, normalization)
        # This is critical for MLP performance - neural networks require clean, normalized inputs
        splits = preprocess_data_splits(raw_splits)

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


def load_mlp_for_inference(
    *,
    path: str,
    n_features: int,
    hidden_sizes: tuple[int, ...],
    dropout: float,
) -> _MLPPrepared:
    """Restore a trained MLP from a checkpoint for inference and explanation.

    The architecture must be rebuilt before the weights can be loaded, and it
    must be rebuilt exactly as training built it -- the Sequential stack is
    Linear/BatchNorm/ReLU with Dropout emitted only when the rate is above
    zero, which shifts every later layer index in the state dict. Rebuilding
    it here, beside the code that trains and saves it, is what keeps the two
    from drifting; a caller that reimplements the stack silently loads a
    differently shaped model.

    Args:
        path: Path to the saved state dict.
        n_features: Number of input features the model was trained on.
        hidden_sizes: Hidden layer widths used at training time.
        dropout: Dropout rate used at training time.

    Returns:
        A prepared model exposing predict_proba and compute_gradients.
    """
    model = _build_model(n_features, hidden_sizes, dropout, "cpu")
    torch_mod = _import_torch()
    state: dict[str, TensorProtocol] = torch_mod.load(path, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return _MLPPrepared(model)


def create_mlp_backend() -> MLPBackend:
    return MLPBackend()


__all__ = [
    "MLP_CAPABILITIES",
    "MLPBackend",
    "create_mlp_backend",
    "load_mlp_for_inference",
]
