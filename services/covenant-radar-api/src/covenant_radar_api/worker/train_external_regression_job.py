"""Background job for training regressors on external CSV data.

Orchestrates the regression training pipeline: parse config, load
regression dataset, train regressor via backend, copy active file,
and optionally upload to data-bank.

Separated from train_external_job.py (classifier) for clear separation
of concerns.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from covenant_ml.backends.regressor_registry import RegressorRegistry
from covenant_ml.types import (
    FeatureImportance,
    LightGBMConfig,
    TrainConfig,
)
from covenant_ml.types_regression import (
    RegressionMetrics,
    RegressionTrainOutcome,
    RegressorBackendName,
)
from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.logging import get_logger

from covenant_radar_api.worker._optimize_regression_common import (
    load_regression_dataset,
)
from covenant_radar_api.worker._train_external_regression_parsers import (
    RegressionParseResult,
    parse_external_regression_train_config,
)

_log = get_logger(__name__)


# =============================================================================
# JSON serialization helpers
# =============================================================================


def _regression_metrics_to_json(
    metrics: RegressionMetrics,
) -> dict[str, JSONValue]:
    """Convert RegressionMetrics to JSON-serializable dict.

    Args:
        metrics: Regression metrics to convert.

    Returns:
        Dict with all regression metric fields.
    """
    return {
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r_squared": metrics["r_squared"],
        "mape": metrics["mape"],
    }


def _importance_to_json(
    imp: FeatureImportance,
) -> dict[str, JSONValue]:
    """Convert FeatureImportance to JSON-serializable dict.

    Args:
        imp: Feature importance to convert.

    Returns:
        Dict with name, importance, and rank.
    """
    return {
        "name": imp["name"],
        "importance": imp["importance"],
        "rank": imp["rank"],
    }


# =============================================================================
# Per-backend config log builders
# =============================================================================


def _build_xgboost_reg_log(config: TrainConfig) -> dict[str, JSONValue]:
    """Build log dict for XGBoost regressor config.

    Args:
        config: XGBoost training configuration.

    Returns:
        Dict with key hyperparameters for logging.
    """
    return {
        "learning_rate": config["learning_rate"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
    }


def _build_lightgbm_reg_log(
    config: LightGBMConfig,
) -> dict[str, JSONValue]:
    """Build log dict for LightGBM regressor config.

    Args:
        config: LightGBM training configuration.

    Returns:
        Dict with key hyperparameters for logging.
    """
    return {
        "learning_rate": config["learning_rate"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "num_leaves": config["num_leaves"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
    }


# =============================================================================
# Active filename and metadata filename mappings
# =============================================================================


def _get_regression_active_filename(
    backend_name: RegressorBackendName,
) -> str:
    """Get active model filename for regressor backend.

    Args:
        backend_name: Regressor backend name.

    Returns:
        Active model filename with appropriate extension.

    Raises:
        ValueError: If backend_name is not recognized.
    """
    if backend_name == "xgboost_reg":
        return "active_xgb_reg.ubj"
    if backend_name == "lightgbm_reg":
        return "active_lgbm_reg.txt"
    raise ValueError(f"Unknown regressor backend: {backend_name}")


def _get_regression_meta_filename(
    backend_name: RegressorBackendName,
) -> str:
    """Get metadata filename for regressor backend.

    Args:
        backend_name: Regressor backend name.

    Returns:
        Filename for the metadata JSON file, or empty string if
        the backend's format is self-describing.
    """
    if backend_name == "lightgbm_reg":
        return "active_lgbm_reg_meta.json"
    # xgboost_reg uses self-describing .ubj format
    return ""


def _write_regression_model_metadata(
    backend_name: RegressorBackendName,
    output_dir: Path,
) -> Path | None:
    """Write regression model metadata JSON if backend needs it.

    Args:
        backend_name: Regressor backend name.
        output_dir: Directory where model is saved.

    Returns:
        Path to metadata file, or None if backend doesn't need metadata.
    """
    meta_filename = _get_regression_meta_filename(backend_name)
    if not meta_filename:
        return None

    meta: dict[str, JSONValue] = {"backend": backend_name}
    meta_path = output_dir / meta_filename
    json_str = dump_json_str(meta, compact=False, indent=2)
    meta_path.write_text(json_str, encoding="utf-8")

    _log.info(
        "Saved regression model metadata",
        extra={
            "backend": backend_name,
            "meta_path": str(meta_path),
        },
    )
    return meta_path


# =============================================================================
# Dispatch: build config log from ParseResult
# =============================================================================


def _dispatch_regression_backend(
    parse_result: RegressionParseResult,
) -> dict[str, JSONValue]:
    """Build config log dict from parsed regression result.

    Uses discriminated union narrowing to extract the typed config
    and build the config log dict.

    Args:
        parse_result: Parsed regression training config.

    Returns:
        Config log dict for structured logging.
    """
    if parse_result["backend"] == "xgboost_reg":
        return _build_xgboost_reg_log(parse_result["config"])
    # lightgbm_reg
    return _build_lightgbm_reg_log(parse_result["config"])


# =============================================================================
# Main training orchestration
# =============================================================================


def run_external_regression_training(
    config_json: str,
    external_dir: Path,
    output_dir: Path,
) -> dict[str, JSONValue]:
    """Run regression training on external CSV data.

    Args:
        config_json: JSON config with dataset name and hyperparameters.
        external_dir: Path to data/external directory with datasets.
        output_dir: Directory to save model artifacts.

    Returns:
        Training result with model info, metrics, and feature importances.

    Raises:
        JSONTypeError: If config JSON is invalid.
        ValueError: If dataset/backend/ratios are invalid.
    """
    parse_result = parse_external_regression_train_config(config_json)
    dataset_name = parse_result["dataset"]
    backend_name: RegressorBackendName = parse_result["backend"]

    # Load regression dataset
    dataset = load_regression_dataset(dataset_name, external_dir)

    # Build config log for structured logging
    config_log = _dispatch_regression_backend(parse_result)

    _log.info(
        "Starting external regression training",
        extra={
            "dataset": dataset_name,
            "n_samples": dataset["meta"]["n_samples"],
            "n_features": dataset["meta"]["n_features"],
            "backend": backend_name,
            "config": config_log,
        },
    )

    # Train via regressor registry
    from covenant_radar_api.worker import _regression_hooks as hooks

    registry: RegressorRegistry = hooks.regressor_registry_factory()
    backend = registry.get(backend_name)

    outcome: RegressionTrainOutcome = backend.train(
        x_features=dataset["x"],
        y_targets=dataset["y"],
        feature_names=list(dataset["meta"]["feature_names"]),
        config=parse_result["config"],
        output_dir=output_dir,
        progress=None,
    )

    # Copy to backend-specific active file
    active_filename = _get_regression_active_filename(backend_name)
    active_model_path = output_dir / active_filename
    shutil.copyfile(outcome["model_path"], active_model_path)

    # Write metadata if needed
    meta_path = _write_regression_model_metadata(backend_name, output_dir)

    # Log top features
    top_features = outcome["feature_importances"][:10]
    _log.info(
        "Regression training complete - top features by importance",
        extra={
            "model_id": outcome["model_id"],
            "test_rmse": outcome["test_metrics"]["rmse"],
            "test_r_squared": outcome["test_metrics"]["r_squared"],
            "top_10_features": [
                {
                    "rank": f["rank"],
                    "name": f["name"],
                    "importance": f"{f['importance']:.4f}",
                }
                for f in top_features
            ],
        },
    )

    # Build result
    result: dict[str, JSONValue] = {
        "status": "complete",
        "dataset": dataset_name,
        "backend": backend_name,
        "model_id": outcome["model_id"],
        "model_path": outcome["model_path"],
        "active_model_path": str(active_model_path),
        "active_meta_path": (str(meta_path) if meta_path is not None else None),
        "samples_total": outcome["samples_total"],
        "samples_train": outcome["samples_train"],
        "samples_val": outcome["samples_val"],
        "samples_test": outcome["samples_test"],
        "n_features": dataset["meta"]["n_features"],
        "best_val_rmse": outcome["best_val_rmse"],
        "best_round": outcome["best_round"],
        "total_rounds": outcome["total_rounds"],
        "early_stopped": outcome["early_stopped"],
        "train_metrics": _regression_metrics_to_json(outcome["train_metrics"]),
        "val_metrics": _regression_metrics_to_json(outcome["val_metrics"]),
        "test_metrics": _regression_metrics_to_json(outcome["test_metrics"]),
        "feature_importances": [_importance_to_json(f) for f in outcome["feature_importances"]],
    }

    return result


# =============================================================================
# Data-bank upload
# =============================================================================


def _upload_regression_model_to_data_bank(
    model_path: Path,
    data_bank_url: str,
    data_bank_key: str,
) -> str:
    """Upload trained regression model to data-bank-api.

    Args:
        model_path: Path to the model file.
        data_bank_url: URL for data-bank-api.
        data_bank_key: API key for data-bank-api.

    Returns:
        file_id from data-bank-api.
    """
    from covenant_radar_api.worker import _test_hooks as hooks

    return hooks.data_bank_uploader(model_path, data_bank_url, data_bank_key)


# =============================================================================
# RQ job entry point
# =============================================================================


def process_external_regression_train_job(
    config_json: str,
) -> dict[str, JSONValue]:
    """RQ job entry point for external regression data training.

    Args:
        config_json: JSON config with dataset name and hyperparameters.

    Returns:
        Training result with model info and feature importances.
    """
    import tempfile

    from covenant_radar_api.core.config import load_settings

    settings = load_settings()

    # Get data-bank config
    data_bank_url = settings["app"]["data_bank_api_url"]
    data_bank_key = settings["app"]["data_bank_api_key"]

    # Get directories from settings
    data_root = Path(settings["app"]["data_root"])
    external_dir = data_root / "external"

    if data_bank_url and data_bank_key:
        # TemporaryDirectory removes the tree on scope exit, including the
        # paths where run_external_regression_training raises.
        with tempfile.TemporaryDirectory(prefix="covenant_regression_model_") as temp_dir:
            result = run_external_regression_training(
                config_json,
                external_dir,
                Path(temp_dir),
            )
            result["model_file_id"] = _upload_regression_model_to_data_bank(
                Path(str(result["active_model_path"])),
                data_bank_url,
                data_bank_key,
            )
            return result

    output_dir = Path(settings["app"]["models_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_external_regression_training(config_json, external_dir, output_dir)


__all__ = [
    "process_external_regression_train_job",
    "run_external_regression_training",
]
