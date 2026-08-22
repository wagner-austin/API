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

from pathlib import Path
from typing import TypedDict, TypeGuard

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
from covenant_ml.trainer import regression_split
from covenant_ml.types import (
    FeatureImportance,
    LSTMConfig,
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
    require_bool,
    require_float,
    require_int,
)
from platform_core.logging import get_logger
from platform_ml.device_selector import resolve_device, resolve_precision
from platform_ml.torch_types import (
    TensorProtocol,
    _import_torch,
)

from covenant_nn.backends.lstm.regressor_training import (
    _build_regressor_model,
    _finalize_regression_metrics,
    _LSTMRegressorPrepared,
    _prepare_regression_components,
    _preprocess_regression_splits,
    _run_regression_training_loop,
)

_log = get_logger(__name__)


# =============================================================================
# Protocols for PyTorch dynamic imports
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

    def load(self, *, path: str) -> _LSTMRegressorPrepared:
        """Load a trained LSTM regressor from saved state dict and metadata.

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
