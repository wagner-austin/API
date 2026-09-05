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
    make_lstm_default_space,
    make_lstm_focused_space,
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
    LSTMConfig,
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

from covenant_nn.backends.lstm.backend_protocols import (
    _EnableGradFactory,
    _SoftmaxCtor,
    _TensorCtor,
)
from covenant_nn.backends.lstm.backend_training import (
    FC_STATE_PREFIX,
    LSTM_STATE_PREFIX,
    _build_model,
    _finalize_metrics,
    _prepare_components,
    _run_training_loop,
)

from .sequences import compute_features_per_step, reshape_flat_to_pseudo_sequences

_log = get_logger(__name__)


def _is_lstm_config(cfg: ClassifierTrainConfig) -> TypeGuard[LSTMConfig]:
    """Check if config is LSTMConfig by looking for LSTM-specific keys."""
    return (
        isinstance(cfg, dict)
        and "hidden_size" in cfg
        and "num_layers" in cfg
        and "bidirectional" in cfg
    )


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


# Key prefixes in a saved LSTM state dict. _LSTMClassifierWrapper composes the
# combined dict from two submodules, so these are the wire contract for every
# checkpoint this backend writes. They are published because any out-of-package
# reader must strip exactly these. Note they deliberately do NOT match the
# wrapper's attribute names (self._lstm, self._fc); deriving them from the
# attributes yields "_lstm."/"_fc.", matches nothing, and loads no weights.


LSTM_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": True,
    "supports_early_stopping": True,
    "supports_feature_importance": False,
    "model_format": "pt",
}


class LSTMBackend(ClassifierBackend):
    """LSTM backend for tabular binary classification."""

    def get_default_search_space(self) -> SearchSpace:
        """Return the backend's default hyperparameter search space.

        Returns:
            The lstm default SearchSpace.
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
            The lstm focused SearchSpace.
        """
        return make_lstm_focused_space(
            best_hidden_size=best_int_params["hidden_size"],
            best_num_layers=best_int_params["num_layers"],
            best_learning_rate=best_float_params["learning_rate"],
        )

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
        progress: Callable[[TrainProgress], None] | None,
        groups: NDArray[np.int64] | None = None,
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
            groups=groups,
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


def load_lstm_for_inference(
    *,
    path: str,
    n_features: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    sequence_length: int,
) -> _LSTMPrepared:
    """Restore a trained LSTM from a checkpoint for inference and explanation.

    Two details make this unsafe to reimplement elsewhere. The input size is a
    ceiling divide, because flat features are zero-padded up to a multiple of
    sequence_length before reshaping, so flooring builds a differently shaped
    model. And the checkpoint's keys are composed by hand as "lstm."/"fc.",
    which deliberately differ from the wrapper's attribute names; a caller
    that derives the prefixes from those attributes strips nothing and loads
    no weights at all.

    Args:
        path: Path to the saved state dict.
        n_features: Number of flat input features at training time.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between layers.
        bidirectional: Whether the LSTM was trained bidirectionally.
        sequence_length: Timesteps the flat features reshape into.

    Returns:
        A prepared model exposing predict_proba and compute_gradients.
    """
    input_size = compute_features_per_step(n_features, sequence_length)
    model = _build_model(
        input_size,
        hidden_size,
        num_layers,
        dropout,
        bidirectional,
        "cpu",
    )
    torch_mod = _import_torch()
    state: dict[str, TensorProtocol] = torch_mod.load(path, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return _LSTMPrepared(model, sequence_length)


def create_lstm_backend() -> LSTMBackend:
    """Create LSTM backend instance."""
    return LSTMBackend()


__all__ = [
    "FC_STATE_PREFIX",
    "LSTM_CAPABILITIES",
    "LSTM_STATE_PREFIX",
    "LSTMBackend",
    "create_lstm_backend",
    "load_lstm_for_inference",
]
