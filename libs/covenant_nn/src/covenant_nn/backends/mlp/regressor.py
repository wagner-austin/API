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

from pathlib import Path
from typing import TypedDict, TypeGuard

import numpy as np
from covenant_ml.backends.protocol import BackendCapabilities
from covenant_ml.backends.regressor_protocol import (
    PreparedRegressor,
    RegressorProgressCallback,
)
from covenant_ml.metrics_regression import compute_all_regression_metrics
from covenant_ml.optimizer.search_spaces import (
    make_mlp_default_space,
    make_mlp_focused_space,
)
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SearchSpace,
)
from covenant_ml.trainer import regression_split
from covenant_ml.types import FeatureImportance, MLPConfig
from covenant_ml.types_regression import (
    RegressionMetrics,
    RegressionTrainOutcome,
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
    TensorProtocol,
    TrainableModel,
    _import_torch,
)

from covenant_nn.backends.mlp.regressor_protocols import (
    _NNLinearCtor,
    _NNSequentialCtor,
    _NoGradFactory,
    _TensorCtor,
)
from covenant_nn.backends.mlp.regressor_training import (
    _build_regressor_model,
    _finalize_regression_metrics,
    _prepare_regression_components,
    _preprocess_regression_splits,
    _run_regression_training_loop,
)

_log = get_logger(__name__)


# =============================================================================
# Protocols for PyTorch dynamic imports
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

    def compute_regression_gradients(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute gradients of the prediction w.r.t. input features.

        Used by the regression gradient explainers. There is no target class
        to select: the model has a single output neuron, so the prediction
        itself is the scalar differentiated.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Gradients with shape (n_samples, n_features).
        """
        torch = _import_torch()
        tensor: _TensorCtor = torch.tensor
        float32: DTypeProtocol = torch.float32

        m = self._model
        m.eval()

        x_tensor: TensorProtocol = tensor(x, dtype=float32)
        x_tensor = x_tensor.requires_grad_(True)

        with torch.enable_grad():
            logits: TensorProtocol = m(x_tensor)
            preds: TensorProtocol = logits.select(1, 0)
            scalar_output: TensorProtocol = preds.sum()
            scalar_output.backward()

        grad_tensor = x_tensor.grad
        assert grad_tensor is not None, "Gradient tensor should not be None after backward()"
        grad_cpu: TensorProtocol = grad_tensor.cpu()
        gradients: NDArray[np.float64] = grad_cpu.numpy().astype(np.float64)
        return gradients


# =============================================================================
# Backend class
# =============================================================================


class MLPRegressorBackend:
    """MLP regressor backend for continuous target prediction.

    Implements the RegressorBackend protocol. Parallel to MLPBackend
    (classifier). Uses MSELoss, no class weights, output dim=1,
    early stopping on validation RMSE.
    """

    def get_default_search_space(self) -> SearchSpace:
        """Return the backend's default hyperparameter search space.

        Returns:
            The mlp_reg default SearchSpace.
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
            The mlp_reg focused SearchSpace.
        """
        return make_mlp_focused_space(
            best_n_layers=best_int_params["n_layers"],
            best_hidden_size=best_int_params["hidden_size"],
            best_learning_rate=best_float_params["learning_rate"],
        )

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

    def load(self, *, path: str) -> _MLPRegressorPrepared:
        """Load a trained MLP regressor from saved state dict and metadata.

        The concrete type is declared rather than PreparedRegressor so callers
        can reach compute_regression_gradients, which the gradient explainers
        require and which the tree regressors do not have. Narrowing a return
        type still satisfies the RegressorBackend protocol.

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
