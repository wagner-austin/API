"""Background job for training on external CSV data.

Orchestrates the training pipeline: parse config, load dataset, train model,
copy active file, write metadata, and optionally upload to data-bank.
Config parsing lives in _train_external_parsers.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from covenant_ml.backends.registry import ClassifierRegistry
from covenant_ml.base_trainer import BaseTabularTrainer
from covenant_ml.datasets import LoadedDataset
from covenant_ml.types import (
    ClassifierTrainConfig,
    ClearGBMConfig,
    EvalMetrics,
    FeatureImportance,
    LightGBMConfig,
    LogRegConfig,
    LSTMConfig,
    MLPConfig,
    RandomForestConfig,
    TrainConfig,
    TrainOutcome,
)
from platform_core.json_utils import JSONValue
from platform_core.logging import get_logger

from covenant_radar_api.worker._train_external_parsers import (
    ParseResult,
    parse_external_train_config,
)
from covenant_radar_api.worker.train_external_metadata import (
    _get_active_filename,
    _LightgbmMetadataBuilder,
    _LogRegMetadataBuilder,
    _LstmMetadataBuilder,
    _MetadataBuilder,
    _MlpMetadataBuilder,
    _RandomForestMetadataBuilder,
    _write_model_metadata,
)

_log = get_logger(__name__)


# =============================================================================
# Dataset loading
# =============================================================================


def _load_dataset(dataset_name: str, external_dir: Path) -> LoadedDataset:
    """Load the specified dataset using pluggable loader.

    Args:
        dataset_name: Name of dataset in registry.
        external_dir: Path to data/external directory.

    Returns:
        LoadedDataset with feature matrix, labels, and metadata.

    Raises:
        KeyError: If dataset not in registry.
        FileNotFoundError: If dataset file doesn't exist.
        ValueError: If data doesn't match expected format.
    """
    from covenant_radar_api.worker import _test_hooks as hooks

    registry = hooks.dataset_registry_factory()
    config = registry.get(dataset_name)
    return hooks.dataset_loader(config, external_dir)


# =============================================================================
# JSON serialization helpers
# =============================================================================


def _metrics_to_json(metrics: EvalMetrics) -> dict[str, JSONValue]:
    """Convert EvalMetrics to JSON-serializable dict."""
    return {
        "loss": metrics["loss"],
        "ppl": metrics["ppl"],
        "auc": metrics["auc"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
    }


def _importance_to_json(
    imp: FeatureImportance,
) -> dict[str, JSONValue]:
    """Convert FeatureImportance to JSON-serializable dict."""
    return {
        "name": imp["name"],
        "importance": imp["importance"],
        "rank": imp["rank"],
    }


# =============================================================================
# Per-backend config log builders
# =============================================================================


def _build_xgboost_log(config: TrainConfig) -> dict[str, JSONValue]:
    """Build log dict for XGBoost config."""
    return {
        "learning_rate": config["learning_rate"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
    }


def _build_mlp_log(config: MLPConfig) -> dict[str, JSONValue]:
    """Build log dict for MLP config."""
    return {
        "learning_rate": config["learning_rate"],
        "hidden_sizes": list(config["hidden_sizes"]),
        "n_epochs": config["n_epochs"],
        "dropout": config["dropout"],
    }


def _build_lstm_log(config: LSTMConfig) -> dict[str, JSONValue]:
    """Build log dict for LSTM config."""
    return {
        "learning_rate": config["learning_rate"],
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "n_epochs": config["n_epochs"],
        "bidirectional": config["bidirectional"],
        "sequence_length": config["sequence_length"],
    }


def _build_lightgbm_log(
    config: LightGBMConfig,
) -> dict[str, JSONValue]:
    """Build log dict for LightGBM config."""
    return {
        "learning_rate": config["learning_rate"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "num_leaves": config["num_leaves"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
    }


def _build_cleargbm_log(
    config: ClearGBMConfig,
) -> dict[str, JSONValue]:
    """Build log dict for ClearGBM config."""
    return {
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "learning_rate": config["learning_rate"],
        "max_bins": config["max_bins"],
        "subsample": config["subsample"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
    }


def _build_logreg_log(config: LogRegConfig) -> dict[str, JSONValue]:
    """Build log dict for LogReg config."""
    return {
        "solver": config["solver"],
        "penalty": config["penalty"],
        "C": config["C"],
        "max_iter": config["max_iter"],
        "tol": config["tol"],
    }


def _build_random_forest_log(
    config: RandomForestConfig,
) -> dict[str, JSONValue]:
    """Build log dict for Random Forest config."""
    return {
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "min_samples_split": config["min_samples_split"],
        "max_features": config["max_features"],
        "bootstrap": config["bootstrap"],
    }


# =============================================================================
# Active filename and metadata filename mappings
# =============================================================================


def _dispatch_backend(
    parse_result: ParseResult,
) -> tuple[dict[str, JSONValue], ClassifierTrainConfig, _MetadataBuilder | None]:
    """Dispatch backend from parsed result.

    Uses discriminated union narrowing to extract the typed config,
    build the config log dict, and prepare a metadata builder.

    Args:
        parse_result: Parsed external training config.

    Returns:
        Tuple of (config_log, train_config, metadata_builder_or_none).
    """
    if parse_result["backend"] == "xgboost":
        return (
            _build_xgboost_log(parse_result["config"]),
            parse_result["config"],
            None,
        )
    if parse_result["backend"] == "mlp":
        return (
            _build_mlp_log(parse_result["config"]),
            parse_result["config"],
            _MlpMetadataBuilder(parse_result["config"]),
        )
    if parse_result["backend"] == "lstm":
        return (
            _build_lstm_log(parse_result["config"]),
            parse_result["config"],
            _LstmMetadataBuilder(parse_result["config"]),
        )
    if parse_result["backend"] == "lightgbm":
        return (
            _build_lightgbm_log(parse_result["config"]),
            parse_result["config"],
            _LightgbmMetadataBuilder(),
        )
    if parse_result["backend"] == "cleargbm":
        return (
            _build_cleargbm_log(parse_result["config"]),
            parse_result["config"],
            None,
        )
    if parse_result["backend"] == "logreg":
        return (
            _build_logreg_log(parse_result["config"]),
            parse_result["config"],
            _LogRegMetadataBuilder(parse_result["config"]),
        )
    # random_forest
    return (
        _build_random_forest_log(parse_result["config"]),
        parse_result["config"],
        _RandomForestMetadataBuilder(parse_result["config"]),
    )


# =============================================================================
# Main training orchestration
# =============================================================================


def run_external_training(
    config_json: str,
    external_dir: Path,
    output_dir: Path,
) -> dict[str, JSONValue]:
    """Run training on external CSV data with automatic feature selection.

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
    parse_result = parse_external_train_config(config_json)
    dataset_name = parse_result["dataset"]
    backend_name = parse_result["backend"]

    # Load raw dataset with all columns
    dataset = _load_dataset(dataset_name, external_dir)

    # Dispatch backend for config log, train config, metadata builder
    config_log, train_config, metadata_builder = _dispatch_backend(parse_result)

    _log.info(
        "Starting external training",
        extra={
            "dataset": dataset_name,
            "n_samples": dataset["meta"]["n_samples"],
            "n_features": dataset["meta"]["n_features"],
            "n_positive": dataset["meta"]["n_positive"],
            "n_negative": dataset["meta"]["n_negative"],
            "backend": backend_name,
            "config": config_log,
        },
    )

    # Train via unified base trainer using default registry
    from covenant_radar_api.worker import _test_hooks as hooks

    reg_factory = hooks.registry_factory
    registry: ClassifierRegistry = reg_factory()
    trainer = BaseTabularTrainer(registry)

    outcome: TrainOutcome = trainer.train(
        backend=backend_name,
        x_features=dataset["x"],
        y_labels=dataset["y"],
        feature_names=list(dataset["meta"]["feature_names"]),
        config=train_config,
        output_dir=output_dir,
        progress=None,
        # Grouped datasets (rw_matches) carry which rows are one match;
        # the split keeps whole matches together so correlated snapshots
        # cannot straddle train and test.
        groups=dataset["groups"],
    )

    # Copy to backend-specific active file
    active_filename = _get_active_filename(backend_name)
    active_model_path = output_dir / active_filename
    shutil.copyfile(outcome["model_path"], active_model_path)

    # Save model metadata for inference loading
    meta_path: Path | None = None
    if metadata_builder is not None:
        meta = metadata_builder.build(dataset["meta"]["n_features"])
        meta_path = _write_model_metadata(backend_name, meta, output_dir)

    # Log top features
    top_features = outcome["feature_importances"][:10]
    _log.info(
        "Training complete - top features by importance",
        extra={
            "model_id": outcome["model_id"],
            "test_auc": outcome["test_metrics"]["auc"],
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
        "model_id": outcome["model_id"],
        "model_path": outcome["model_path"],
        "active_model_path": str(active_model_path),
        "active_meta_path": (str(meta_path) if meta_path is not None else None),
        "samples_total": outcome["samples_total"],
        "samples_train": outcome["samples_train"],
        "samples_val": outcome["samples_val"],
        "samples_test": outcome["samples_test"],
        "n_features": dataset["meta"]["n_features"],
        "scale_pos_weight": outcome["scale_pos_weight_computed"],
        "best_val_auc": outcome["best_val_auc"],
        "best_round": outcome["best_round"],
        "total_rounds": outcome["total_rounds"],
        "early_stopped": outcome["early_stopped"],
        "train_metrics": _metrics_to_json(outcome["train_metrics"]),
        "val_metrics": _metrics_to_json(outcome["val_metrics"]),
        "test_metrics": _metrics_to_json(outcome["test_metrics"]),
        "feature_importances": [_importance_to_json(f) for f in outcome["feature_importances"]],
    }

    return result


# =============================================================================
# Data-bank upload
# =============================================================================


def _upload_model_to_data_bank(
    model_path: Path,
    data_bank_url: str,
    data_bank_key: str,
) -> str:
    """Upload trained model to data-bank-api.

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


def process_external_train_job(
    config_json: str,
) -> dict[str, JSONValue]:
    """RQ job entry point for external data training.

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
        # paths where run_external_training raises on an invalid config.
        with tempfile.TemporaryDirectory(prefix="covenant_model_") as temp_dir:
            result = run_external_training(config_json, external_dir, Path(temp_dir))
            result["model_file_id"] = _upload_model_to_data_bank(
                Path(str(result["active_model_path"])),
                data_bank_url,
                data_bank_key,
            )
            return result

    output_dir = Path(settings["app"]["models_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_external_training(config_json, external_dir, output_dir)


__all__ = ["process_external_train_job", "run_external_training"]
