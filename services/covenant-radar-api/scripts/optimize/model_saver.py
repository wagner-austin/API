"""Model saving after optimization with best-model tracking.

Trains and saves the best model after optimization completes.
Only overwrites existing models if the new model has better validation AUC.

Builds backend-specific training configs from unified sampled parameters.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from covenant_ml.features import (
    FeaturePreset,
    engineer_features,
    get_feature_config_for_preset,
)
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SampledStringParams
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    ClearGBMConfig,
    LightGBMConfig,
    LogRegConfig,
    LogRegPenalty,
    LogRegSolver,
    LSTMConfig,
    MLPConfig,
    RandomForestConfig,
    TrainConfig,
    TrainOutcome,
)
from numpy.typing import NDArray
from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    require_float,
)
from platform_core.logging import get_logger, get_rich_console

import scripts._test_hooks as _hooks
from scripts._test_hooks import UnifiedOptimizationResult
from scripts.optimize.cli import DatasetName

_log = get_logger(__name__)


# =============================================================================
# Save Result TypedDict
# =============================================================================


class SaveModelResult(TypedDict, total=True):
    """Result of save_best_model operation."""

    saved: bool
    reason: str
    model_path: str | None
    meta_path: str | None
    train_outcome: TrainOutcome | None


# =============================================================================
# Model File Extensions
# =============================================================================

MODEL_EXTENSIONS: dict[BackendName, str] = {
    "xgboost": "ubj",
    "mlp": "pt",
    "lightgbm": "txt",
    "lstm": "pt",
    "cleargbm": "json",
    "logreg": "joblib",
    "random_forest": "joblib",
}


# =============================================================================
# Per-Backend Config Builders
# =============================================================================


def _build_xgboost_config(
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
) -> TrainConfig:
    """Build XGBoost training config from sampled parameters.

    Args:
        int_params: Sampled integer parameters from optimization.
        float_params: Sampled float parameters from optimization.

    Returns:
        XGBoost training configuration.
    """
    return TrainConfig(
        device="auto",
        learning_rate=float_params["learning_rate"],
        max_depth=int_params["max_depth"],
        n_estimators=int_params["n_estimators"],
        subsample=float_params["subsample"],
        colsample_bytree=float_params["colsample_bytree"],
        reg_alpha=float_params["reg_alpha"],
        reg_lambda=float_params["reg_lambda"],
        random_state=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        early_stopping_rounds=10,
    )


def _build_mlp_config(
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
) -> MLPConfig:
    """Build MLP training config from sampled parameters.

    Args:
        int_params: Sampled integer parameters from optimization.
        float_params: Sampled float parameters from optimization.

    Returns:
        MLP training configuration.
    """
    n_layers = int_params["n_layers"]
    hidden_size = int_params["hidden_size"]
    return MLPConfig(
        device="auto",
        precision="fp32",
        optimizer="adamw",
        hidden_sizes=tuple(hidden_size for _ in range(n_layers)),
        learning_rate=float_params["learning_rate"],
        batch_size=int_params["batch_size"],
        n_epochs=50,
        dropout=float_params["dropout"],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_patience=10,
    )


def _build_lstm_config(
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
) -> LSTMConfig:
    """Build LSTM training config from sampled parameters.

    Args:
        int_params: Sampled integer parameters from optimization.
        float_params: Sampled float parameters from optimization.

    Returns:
        LSTM training configuration.
    """
    return LSTMConfig(
        device="auto",
        precision="fp32",
        hidden_size=int_params["hidden_size"],
        num_layers=int_params["num_layers"],
        dropout=float_params["dropout"],
        bidirectional=False,
        sequence_length=5,
        learning_rate=float_params["learning_rate"],
        batch_size=int_params["batch_size"],
        n_epochs=50,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_patience=10,
    )


def _build_lightgbm_config(
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
) -> LightGBMConfig:
    """Build LightGBM training config from sampled parameters.

    Args:
        int_params: Sampled integer parameters from optimization.
        float_params: Sampled float parameters from optimization.

    Returns:
        LightGBM training configuration.
    """
    return LightGBMConfig(
        device="auto",
        learning_rate=float_params["learning_rate"],
        max_depth=int_params["max_depth"],
        n_estimators=int_params["n_estimators"],
        num_leaves=int_params["num_leaves"],
        min_child_samples=int_params["min_child_samples"],
        subsample=float_params["subsample"],
        colsample_bytree=float_params["colsample_bytree"],
        reg_alpha=float_params["reg_alpha"],
        reg_lambda=float_params["reg_lambda"],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_rounds=10,
    )


def _build_cleargbm_config(
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
) -> ClearGBMConfig:
    """Build ClearGBM training config from sampled parameters.

    Args:
        int_params: Sampled integer parameters from optimization.
        float_params: Sampled float parameters from optimization.

    Returns:
        ClearGBM training configuration.
    """
    return ClearGBMConfig(
        n_estimators=int_params["n_estimators"],
        max_depth=int_params["max_depth"],
        learning_rate=float_params["learning_rate"],
        min_samples_split=int_params["min_samples_split"],
        min_samples_leaf=int_params["min_samples_leaf"],
        max_features=None,
        max_bins=int_params["max_bins"],
        subsample=float_params["subsample"],
        random_state=42,
        track_contributions=False,
        monotonic_constraints=None,
        reg_alpha=float_params["reg_alpha"],
        reg_lambda=float_params["reg_lambda"],
        n_jobs=-1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        early_stopping_rounds=10,
    )


def _narrow_logreg_solver(raw: str) -> LogRegSolver:
    """Narrow string to LogRegSolver literal type.

    Args:
        raw: Solver name string.

    Returns:
        Validated LogRegSolver literal.

    Raises:
        ValueError: If solver name is not valid.
    """
    if raw == "lbfgs":
        return "lbfgs"
    if raw == "liblinear":
        return "liblinear"
    if raw == "newton-cg":
        return "newton-cg"
    if raw == "newton-cholesky":
        return "newton-cholesky"
    if raw == "sag":
        return "sag"
    if raw == "saga":
        return "saga"
    raise ValueError(f"Invalid LogReg solver: {raw}")


def _narrow_logreg_penalty(raw: str) -> LogRegPenalty:
    """Narrow string to LogRegPenalty literal type.

    Args:
        raw: Penalty name string.

    Returns:
        Validated LogRegPenalty literal.

    Raises:
        ValueError: If penalty name is not valid.
    """
    if raw == "l1":
        return "l1"
    if raw == "l2":
        return "l2"
    if raw == "elasticnet":
        return "elasticnet"
    if raw == "none":
        return "none"
    raise ValueError(f"Invalid LogReg penalty: {raw}")


def _build_logreg_config(
    float_params: SampledFloatParams,
    string_params: SampledStringParams,
) -> LogRegConfig:
    """Build Logistic Regression training config from sampled parameters.

    Args:
        float_params: Sampled float parameters from optimization.
        string_params: Sampled string parameters from optimization.

    Returns:
        LogReg training configuration.
    """
    solver = _narrow_logreg_solver(string_params["solver"])
    penalty = _narrow_logreg_penalty(string_params["penalty"])

    return LogRegConfig(
        solver=solver,
        penalty=penalty,
        C=float_params["C"],
        max_iter=1000,
        tol=float_params["tol"],
        class_weight_balanced=True,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        l1_ratio=float_params["l1_ratio"],
    )


def _build_random_forest_config(
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
    string_params: SampledStringParams,
) -> RandomForestConfig:
    """Build Random Forest training config from sampled parameters.

    Args:
        int_params: Sampled integer parameters from optimization.
        float_params: Sampled float parameters from optimization.
        string_params: Sampled string parameters from optimization.

    Returns:
        RandomForest training configuration.
    """
    # Parse max_features — narrow to the literal union type
    max_features_val: Literal["sqrt", "log2"] | float | int | None
    max_features_str = string_params.get("max_features")
    if max_features_str == "sqrt":
        max_features_val = "sqrt"
    elif max_features_str == "log2":
        max_features_val = "log2"
    elif "max_features_float" in float_params:
        max_features_val = float_params["max_features_float"]
    else:
        max_features_val = "sqrt"

    return RandomForestConfig(
        n_estimators=int_params["n_estimators"],
        max_depth=int_params.get("max_depth"),
        min_samples_split=int_params["min_samples_split"],
        min_samples_leaf=int_params["min_samples_leaf"],
        max_features=max_features_val,
        bootstrap=True,
        class_weight_balanced=True,
        n_jobs=-1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        oob_score=False,
    )


def build_train_config(
    backend: str,
    result: UnifiedOptimizationResult,
) -> ClassifierTrainConfig:
    """Build backend-specific training config from optimization result.

    Extracts sampled parameters from the unified result and constructs
    the appropriate config TypedDict for the target backend.

    Args:
        backend: Backend name determining config structure.
        result: Optimization result with best sampled parameters.

    Returns:
        Backend-specific training configuration.

    Raises:
        ValueError: If backend name is not recognized.
    """
    int_params = result["best_int_params"]
    float_params = result["best_float_params"]
    string_params = result["best_string_params"]

    if backend == "xgboost":
        return _build_xgboost_config(int_params, float_params)
    if backend == "mlp":
        return _build_mlp_config(int_params, float_params)
    if backend == "lstm":
        return _build_lstm_config(int_params, float_params)
    if backend == "lightgbm":
        return _build_lightgbm_config(int_params, float_params)
    if backend == "cleargbm":
        return _build_cleargbm_config(int_params, float_params)
    if backend == "logreg":
        return _build_logreg_config(float_params, string_params)
    if backend == "random_forest":
        return _build_random_forest_config(int_params, float_params, string_params)

    raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
# Metadata Loading
# =============================================================================


def _get_meta_path(output_dir: Path, dataset: str, backend: BackendName) -> Path:
    """Get the path for the metadata JSON file.

    Args:
        output_dir: Directory where models are saved.
        dataset: Dataset name (taiwan, us, polish).
        backend: Backend name.

    Returns:
        Path to the metadata JSON file.
    """
    return output_dir / f"{dataset}_{backend}_best_meta.json"


def _get_model_path(output_dir: Path, dataset: str, backend: BackendName) -> Path:
    """Get the path for the model file.

    Args:
        output_dir: Directory where models are saved.
        dataset: Dataset name (taiwan, us, polish).
        backend: Backend name.

    Returns:
        Path to the model file.
    """
    ext = MODEL_EXTENSIONS[backend]
    return output_dir / f"{dataset}_{backend}_best.{ext}"


def load_existing_auc(
    output_dir: Path,
    dataset: str,
    backend: BackendName,
) -> float | None:
    """Load existing model's validation AUC if metadata file exists.

    Only loads the AUC field, not the full config, to avoid complex decoding.

    Args:
        output_dir: Directory where models are saved.
        dataset: Dataset name (taiwan, us, polish).
        backend: Backend name.

    Returns:
        Best validation AUC if metadata file exists, None otherwise.
    """
    from platform_core.json_utils import narrow_json_to_dict

    meta_path = _get_meta_path(output_dir, dataset, backend)
    if not meta_path.exists():
        return None

    raw_str = meta_path.read_text(encoding="utf-8")
    raw_json = load_json_str(raw_str)
    raw = narrow_json_to_dict(raw_json)

    return require_float(raw, "best_val_auc")


def should_save_model(
    new_auc: float,
    existing_auc: float | None,
) -> bool:
    """Determine if the new model should be saved.

    The new model is saved only if:
    - No existing model exists, OR
    - The new model has strictly better validation AUC

    Args:
        new_auc: Validation AUC of the new model.
        existing_auc: Existing model's AUC, or None if no model exists.

    Returns:
        True if the new model should be saved.
    """
    if existing_auc is None:
        return True
    return new_auc > existing_auc


# =============================================================================
# Model Training and Saving
# =============================================================================


def _load_dataset_with_features(
    dataset: DatasetName,
    feature_preset: FeaturePreset,
    external_dir: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Load dataset and apply feature engineering.

    Args:
        dataset: Dataset name (taiwan, us, polish).
        feature_preset: Feature engineering preset.
        external_dir: Directory containing external datasets.

    Returns:
        Tuple of (features, labels, feature_names).
    """
    registry = _hooks.dataset_registry_factory()
    config = registry.get(dataset)
    loaded = _hooks.dataset_loader(config, external_dir)

    # Apply feature engineering using the preset
    fe_config = get_feature_config_for_preset(feature_preset)
    engineered = engineer_features(
        loaded["x"],
        list(loaded["meta"]["feature_names"]),
        fe_config,
    )

    return engineered["x"], loaded["y"], engineered["feature_names"]


def _save_metadata(
    meta_path: Path,
    backend: BackendName,
    dataset: str,
    feature_preset: str,
    best_val_auc: float,
    model_path: str,
    n_features: int,
    n_samples: int,
) -> None:
    """Save model metadata to JSON file.

    Args:
        meta_path: Path to save metadata.
        backend: Backend name.
        dataset: Dataset name.
        feature_preset: Feature preset used.
        best_val_auc: Best validation AUC achieved.
        model_path: Path to the saved model.
        n_features: Number of input features.
        n_samples: Number of training samples.
    """
    timestamp = datetime.now(UTC).isoformat()

    meta_dict: dict[str, str | float | int] = {
        "backend": backend,
        "dataset": dataset,
        "feature_preset": feature_preset,
        "best_val_auc": best_val_auc,
        "saved_at": timestamp,
        "model_path": model_path,
        "n_features": n_features,
        "n_samples": n_samples,
    }

    meta_json = dump_json_str(meta_dict)
    with meta_path.open("w", encoding="utf-8") as f:
        f.write(meta_json)


def save_best_model(
    result: UnifiedOptimizationResult,
    dataset: DatasetName,
    feature_preset: FeaturePreset,
    project_root: Path,
) -> SaveModelResult:
    """Train and save the best model after optimization.

    Builds the appropriate training config from the optimization result's
    sampled parameters, then trains and saves the model only if it's
    better than any existing saved model.

    Args:
        result: Optimization result with best sampled parameters.
        dataset: Dataset name (taiwan, us, polish).
        feature_preset: Feature engineering preset used.
        project_root: Project root directory.

    Returns:
        SaveModelResult with save status and paths.
    """
    console = get_rich_console()
    backend: BackendName = result["backend"]
    new_auc = result["best_value"]

    # Determine output directory
    output_dir = project_root / "models" / backend
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if we should save
    existing_auc = load_existing_auc(output_dir, dataset, backend)
    if not should_save_model(new_auc, existing_auc):
        existing_auc_val = existing_auc if existing_auc is not None else 0.0
        reason = f"New AUC ({new_auc:.6f}) not better than existing ({existing_auc_val:.6f})"
        console.print(f"[yellow]Skipping model save: {reason}[/yellow]")
        return SaveModelResult(
            saved=False,
            reason=reason,
            model_path=None,
            meta_path=None,
            train_outcome=None,
        )

    # Load dataset with features
    external_dir = project_root / "data" / "external"
    x_features, y_labels, feature_names = _load_dataset_with_features(
        dataset, feature_preset, external_dir
    )

    # Build backend-specific training config from sampled params
    config: ClassifierTrainConfig = build_train_config(backend, result)

    # Get backend and train
    registry = _hooks.backend_registry_factory()
    backend_impl = registry.get(backend)

    console.print("\n[cyan]Training final model with best hyperparameters...[/cyan]")

    # Train the model
    train_outcome: TrainOutcome = backend_impl.train(
        x_features=x_features,
        y_labels=y_labels,
        feature_names=feature_names,
        config=config,
        output_dir=output_dir,
        progress=None,
    )

    # Move/rename model to best location
    trained_model_path = Path(train_outcome["model_path"])
    best_model_path = _get_model_path(output_dir, dataset, backend)
    meta_path = _get_meta_path(output_dir, dataset, backend)

    # Copy trained model to best location (or rename if same dir)
    if trained_model_path != best_model_path:
        if best_model_path.exists():
            best_model_path.unlink()
        trained_model_path.rename(best_model_path)

    # Save metadata
    _save_metadata(
        meta_path=meta_path,
        backend=backend,
        dataset=dataset,
        feature_preset=feature_preset,
        best_val_auc=train_outcome["best_val_auc"],
        model_path=str(best_model_path),
        n_features=int(x_features.shape[1]),
        n_samples=int(x_features.shape[0]),
    )

    console.print(
        f"[green]Model saved: {best_model_path.name} "
        f"(AUC: {train_outcome['best_val_auc']:.6f})[/green]"
    )

    return SaveModelResult(
        saved=True,
        reason="New best model",
        model_path=str(best_model_path),
        meta_path=str(meta_path),
        train_outcome=train_outcome,
    )


__all__ = [
    "MODEL_EXTENSIONS",
    "SaveModelResult",
    "build_train_config",
    "load_existing_auc",
    "save_best_model",
    "should_save_model",
]
