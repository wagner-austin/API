"""Model saving after optimization with best-model tracking.

Trains and saves the best model after optimization completes.
Only overwrites existing models if the new model has better validation AUC.

Builds backend-specific training configs from unified sampled parameters.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import numpy as np
from covenant_ml.features import (
    FeaturePreset,
    engineer_features,
    get_feature_config_for_preset,
)
from covenant_ml.types import BackendName, ClassifierTrainConfig, TrainOutcome
from numpy.typing import NDArray
from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    require_float,
)
from platform_core.logging import get_logger
from platform_core.rich_logging import get_rich_console

import scripts._test_hooks as _hooks
from scripts._test_hooks import UnifiedOptimizationResult
from scripts.optimize._config_builders import (
    _build_cleargbm_config,
    _build_lightgbm_config,
    _build_logreg_config,
    _build_lstm_config,
    _build_mlp_config,
    _build_random_forest_config,
    _build_xgboost_config,
)
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
