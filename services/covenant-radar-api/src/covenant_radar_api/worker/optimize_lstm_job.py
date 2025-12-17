"""Background job for LSTM hyperparameter optimization using Optuna TPE.

Runs Bayesian optimization on external bankruptcy datasets to find
optimal LSTM hyperparameters. Results include best hyperparameters
and recommended LSTMConfig for subsequent training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, TypedDict

from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer import (
    LSTMSearchSpace,
    OptimizationSummary,
    TrialResult,
    create_lstm_objective,
    create_lstm_optimizer,
)
from covenant_ml.types import LSTMConfig, LSTMPrecision
from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    require_int,
    require_str,
)
from platform_core.logging import get_logger

from covenant_radar_api.worker._optimize_common import (
    build_optimization_config,
    load_dataset,
    optional_int,
    parse_device,
    parse_feature_preset,
)

_log = get_logger(__name__)


SpaceProfile = Literal["default"]


def _parse_space_profile(raw: JSONValue | None) -> SpaceProfile:
    """Parse space profile, defaulting to 'default'.

    LSTM optimization currently only supports the default profile.
    The focused profile requires initial values from a previous run.

    Args:
        raw: Raw JSON value.

    Returns:
        SpaceProfile literal.

    Raises:
        JSONTypeError: If value is not a string or invalid profile.
    """
    if raw is None:
        return "default"
    if not isinstance(raw, str):
        raise JSONTypeError("space_profile must be a string")
    if raw == "default":
        return "default"
    raise JSONTypeError("space_profile must be: default")


def _parse_precision(raw: JSONValue | None) -> LSTMPrecision:
    """Parse precision setting, defaulting to 'fp32'.

    Args:
        raw: Raw JSON value.

    Returns:
        LSTMPrecision literal.

    Raises:
        JSONTypeError: If value is not a string or invalid precision.
    """
    if raw is None:
        return "fp32"
    if not isinstance(raw, str):
        raise JSONTypeError("precision must be a string")
    if raw == "fp32":
        return "fp32"
    if raw == "fp16":
        return "fp16"
    if raw == "bf16":
        return "bf16"
    if raw == "auto":
        return "auto"
    raise JSONTypeError("precision must be one of: fp32, fp16, bf16, auto")


def _parse_bidirectional(raw: JSONValue | None) -> bool:
    """Parse bidirectional setting, defaulting to False.

    Args:
        raw: Raw JSON value.

    Returns:
        Boolean value.

    Raises:
        JSONTypeError: If value is not a boolean.
    """
    if raw is None:
        return False
    if not isinstance(raw, bool):
        raise JSONTypeError("bidirectional must be a boolean")
    return raw


class LSTMOptimizeParseResult(TypedDict, total=True):
    """Parsed LSTM optimization request."""

    dataset: str
    n_trials: int
    timeout_seconds: int | None
    device: Literal["cpu", "cuda", "auto"]
    space_profile: SpaceProfile
    feature_preset: FeaturePreset
    random_state: int
    precision: LSTMPrecision
    n_epochs: int
    early_stopping_patience: int
    sequence_length: int
    bidirectional: bool


def _parse_optimize_config(config_json: str) -> LSTMOptimizeParseResult:
    """Parse LSTM optimization config from JSON string.

    Args:
        config_json: JSON string with optimization parameters.

    Returns:
        LSTMOptimizeParseResult with all optimization parameters.

    Raises:
        JSONTypeError: If config is not a JSON object or has invalid fields.
        ValueError: If dataset name is invalid.
    """
    from covenant_radar_api.worker._optimize_common import parse_dataset_name

    raw = load_json_str(config_json)
    if not isinstance(raw, dict):
        raise JSONTypeError("config must be a JSON object")

    # Dataset selection (required)
    dataset = require_str(raw, "dataset")
    dataset_name = parse_dataset_name(dataset)

    n_trials = require_int(raw, "n_trials")

    timeout_raw = raw.get("timeout_seconds")
    timeout_seconds: int | None = None
    if timeout_raw is not None:
        if not isinstance(timeout_raw, int):
            raise JSONTypeError("timeout_seconds must be an integer or null")
        timeout_seconds = timeout_raw

    device = parse_device(raw.get("device"))
    space_profile = _parse_space_profile(raw.get("space_profile"))
    feature_preset = parse_feature_preset(raw.get("feature_preset"))
    random_state = optional_int(raw, "random_state", 42)
    precision = _parse_precision(raw.get("precision"))
    n_epochs = optional_int(raw, "n_epochs", 50)
    early_stopping_patience = optional_int(raw, "early_stopping_patience", 10)
    sequence_length = optional_int(raw, "sequence_length", 5)
    bidirectional = _parse_bidirectional(raw.get("bidirectional"))

    return LSTMOptimizeParseResult(
        dataset=dataset_name,
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        device=device,
        space_profile=space_profile,
        feature_preset=feature_preset,
        random_state=random_state,
        precision=precision,
        n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
        sequence_length=sequence_length,
        bidirectional=bidirectional,
    )


def _get_search_space(profile: SpaceProfile) -> LSTMSearchSpace:
    """Get LSTM search space based on profile name.

    Args:
        profile: Space profile name (currently only 'default' supported).

    Returns:
        LSTMSearchSpace with appropriate ranges.
    """
    from covenant_ml.optimizer import make_lstm_default_space

    # Currently only default space is supported
    # Focused space requires initial values from a previous optimization run
    _ = profile  # Used for future extension
    return make_lstm_default_space()


class LSTMOptimizationResult(TypedDict, total=True):
    """Result of an LSTM hyperparameter optimization run."""

    backend: Literal["lstm"]
    status: Literal["complete"]
    dataset: str
    n_samples: int
    n_features: int
    feature_preset: FeaturePreset
    n_trials_complete: int
    n_trials_pruned: int
    n_trials_failed: int
    best_trial_number: int
    best_val_auc: float
    best_hidden_size: int
    best_num_layers: int
    best_learning_rate: float
    best_dropout: float
    best_batch_size: int
    duration_seconds: float
    recommended_config: LSTMConfig


def _generate_lstm_config(
    summary: OptimizationSummary,
    device: Literal["cpu", "cuda", "auto"],
    precision: LSTMPrecision,
    n_epochs: int,
    early_stopping_patience: int,
    sequence_length: int,
    bidirectional: bool,
) -> LSTMConfig:
    """Generate an LSTMConfig from optimization summary.

    Args:
        summary: Optimization summary with best parameters.
        device: Device to use for training.
        precision: Precision mode.
        n_epochs: Number of training epochs.
        early_stopping_patience: Early stopping patience.
        sequence_length: Number of time periods in each sequence.
        bidirectional: Whether to use bidirectional LSTM.

    Returns:
        LSTMConfig ready for training.
    """
    best_int = summary["best_int_params"]
    best_float = summary["best_float_params"]
    hidden_size = best_int.get("hidden_size", 64)
    num_layers = best_int.get("num_layers", 2)
    batch_size = best_int.get("batch_size", 32)

    return LSTMConfig(
        device=device,
        precision=precision,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=best_float["dropout"],
        bidirectional=bidirectional,
        sequence_length=sequence_length,
        learning_rate=best_float["learning_rate"],
        batch_size=batch_size,
        n_epochs=n_epochs,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_patience=early_stopping_patience,
    )


class LSTMTrialProgressInfo(TypedDict):
    """Information about LSTM trial progress for CLI display."""

    trial_number: int
    n_trials_total: int
    current_auc: float
    best_auc: float
    best_trial: int
    is_best: bool
    best_learning_rate: float
    best_hidden_size: int
    best_num_layers: int
    best_dropout: float


class LSTMTrialProgressCallbackProtocol(Protocol):
    """Protocol for LSTM trial progress callback."""

    def __call__(self, info: LSTMTrialProgressInfo) -> None:
        """Called after each trial with progress info."""
        ...


def run_lstm_optimization(
    config_json: str,
    external_dir: Path,
    output_dir: Path,
    progress_callback: LSTMTrialProgressCallbackProtocol | None = None,
) -> LSTMOptimizationResult:
    """Run LSTM hyperparameter optimization on external dataset.

    Args:
        config_json: JSON config with dataset and optimization parameters.
        external_dir: Path to data/external directory with datasets.
        output_dir: Directory to save optimization results.
        progress_callback: Optional callback for trial progress updates.

    Returns:
        LSTMOptimizationResult with best hyperparameters and recommended config.
    """
    parse_result = _parse_optimize_config(config_json)
    dataset_name = parse_result["dataset"]

    # Load raw dataset
    dataset = load_dataset(dataset_name, external_dir)

    _log.info(
        "Starting LSTM hyperparameter optimization",
        extra={
            "dataset": dataset_name,
            "n_samples": dataset["meta"]["n_samples"],
            "n_features": dataset["meta"]["n_features"],
            "n_trials": parse_result["n_trials"],
            "space_profile": parse_result["space_profile"],
            "feature_preset": parse_result["feature_preset"],
            "device": parse_result["device"],
            "precision": parse_result["precision"],
            "sequence_length": parse_result["sequence_length"],
        },
    )

    # Build config and search space
    config = build_optimization_config(
        n_trials=parse_result["n_trials"],
        timeout_seconds=parse_result["timeout_seconds"],
        random_state=parse_result["random_state"],
    )
    search_space = _get_search_space(parse_result["space_profile"])

    # Create objective function (applies feature engineering if preset != "none")
    objective = create_lstm_objective(
        dataset["x"],
        dataset["y"],
        list(dataset["meta"]["feature_names"]),
        parse_result["device"],
        parse_result["precision"],
        parse_result["feature_preset"],
        parse_result["n_epochs"],
        parse_result["early_stopping_patience"],
        parse_result["sequence_length"],
        bidirectional=parse_result["bidirectional"],
    )

    # Track progress
    best_auc = 0.0
    best_trial_num = 0
    best_learning_rate = 0.0
    best_hidden_size = 0
    best_num_layers = 0
    best_dropout = 0.0
    trials_seen = 0
    n_trials_total = parse_result["n_trials"]

    def trial_callback(result: TrialResult) -> None:
        nonlocal best_auc
        nonlocal best_trial_num
        nonlocal trials_seen
        nonlocal best_learning_rate
        nonlocal best_hidden_size
        nonlocal best_num_layers
        nonlocal best_dropout
        trials_seen += 1
        auc = result["value"]
        is_best = auc > best_auc
        if is_best:
            best_auc = auc
            best_trial_num = result["trial_number"]
            trial_int_params = result["int_params"]
            trial_float_params = result["float_params"]
            best_learning_rate = trial_float_params["learning_rate"]
            best_hidden_size = trial_int_params.get("hidden_size", 64)
            best_num_layers = trial_int_params.get("num_layers", 2)
            best_dropout = trial_float_params["dropout"]
            _log.info(
                "New best LSTM trial",
                extra={
                    "trial": result["trial_number"],
                    "auc": f"{auc:.4f}",
                    "hidden_size": best_hidden_size,
                    "num_layers": best_num_layers,
                    "learning_rate": f"{trial_float_params['learning_rate']:.4f}",
                    "dropout": f"{best_dropout:.2f}",
                },
            )

        # Call progress callback if provided
        if progress_callback is not None:
            progress_info: LSTMTrialProgressInfo = {
                "trial_number": result["trial_number"],
                "n_trials_total": n_trials_total,
                "current_auc": auc,
                "best_auc": best_auc,
                "best_trial": best_trial_num,
                "is_best": is_best,
                "best_learning_rate": best_learning_rate,
                "best_hidden_size": best_hidden_size,
                "best_num_layers": best_num_layers,
                "best_dropout": best_dropout,
            }
            progress_callback(progress_info)

    # Run optimization
    optimizer = create_lstm_optimizer()
    summary: OptimizationSummary = optimizer.optimize(
        x_features=dataset["x"],
        y_labels=dataset["y"],
        feature_names=list(dataset["meta"]["feature_names"]),
        search_space=search_space,
        config=config,
        objective=objective,
        trial_callback=trial_callback,
    )

    _log.info(
        "LSTM optimization complete",
        extra={
            "dataset": dataset_name,
            "best_trial": summary["best_trial_number"],
            "best_auc": f"{summary['best_value']:.4f}",
            "n_trials_complete": summary["n_trials_complete"],
            "duration_seconds": f"{summary['total_duration_seconds']:.1f}",
        },
    )

    # Generate recommended config
    recommended_config = _generate_lstm_config(
        summary,
        parse_result["device"],
        parse_result["precision"],
        parse_result["n_epochs"],
        parse_result["early_stopping_patience"],
        parse_result["sequence_length"],
        parse_result["bidirectional"],
    )

    # Save results to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{dataset_name}_lstm_optuna_result.json"
    config_path = output_dir / f"{dataset_name}_lstm_optimal_config.json"

    # Build serializable result
    best_int_params = summary["best_int_params"]
    best_float_params = summary["best_float_params"]

    result_dict: dict[str, JSONValue] = {
        "dataset": dataset_name,
        "n_samples": dataset["meta"]["n_samples"],
        "n_features": objective.n_features,
        "best_trial": summary["best_trial_number"],
        "best_val_auc": summary["best_value"],
        "best_hidden_size": best_int_params.get("hidden_size", 64),
        "best_num_layers": best_int_params.get("num_layers", 2),
        "best_batch_size": best_int_params.get("batch_size", 32),
        "best_learning_rate": best_float_params["learning_rate"],
        "best_dropout": best_float_params["dropout"],
        "n_trials_complete": summary["n_trials_complete"],
        "duration_seconds": summary["total_duration_seconds"],
    }

    # Write results
    with open(result_path, "w") as f:
        f.write(dump_json_str(result_dict))

    config_dict: dict[str, JSONValue] = {
        "device": recommended_config["device"],
        "precision": recommended_config["precision"],
        "hidden_size": recommended_config["hidden_size"],
        "num_layers": recommended_config["num_layers"],
        "dropout": recommended_config["dropout"],
        "bidirectional": recommended_config["bidirectional"],
        "sequence_length": recommended_config["sequence_length"],
        "learning_rate": recommended_config["learning_rate"],
        "batch_size": recommended_config["batch_size"],
        "n_epochs": recommended_config["n_epochs"],
        "train_ratio": recommended_config["train_ratio"],
        "val_ratio": recommended_config["val_ratio"],
        "test_ratio": recommended_config["test_ratio"],
        "random_state": recommended_config["random_state"],
        "early_stopping_patience": recommended_config["early_stopping_patience"],
    }

    with open(config_path, "w") as f:
        f.write(dump_json_str(config_dict))

    _log.info(
        "Saved LSTM optimization results",
        extra={
            "result_path": str(result_path),
            "config_path": str(config_path),
        },
    )

    return LSTMOptimizationResult(
        backend="lstm",
        status="complete",
        dataset=dataset_name,
        n_samples=dataset["meta"]["n_samples"],
        n_features=objective.n_features,
        feature_preset=parse_result["feature_preset"],
        n_trials_complete=summary["n_trials_complete"],
        n_trials_pruned=summary["n_trials_pruned"],
        n_trials_failed=summary["n_trials_failed"],
        best_trial_number=summary["best_trial_number"],
        best_val_auc=summary["best_value"],
        best_hidden_size=best_int_params.get("hidden_size", 64),
        best_num_layers=best_int_params.get("num_layers", 2),
        best_learning_rate=best_float_params["learning_rate"],
        best_dropout=best_float_params["dropout"],
        best_batch_size=best_int_params.get("batch_size", 32),
        duration_seconds=summary["total_duration_seconds"],
        recommended_config=recommended_config,
    )


def process_lstm_optimize_job(config_json: str) -> dict[str, JSONValue]:
    """RQ job entry point for LSTM hyperparameter optimization.

    Args:
        config_json: JSON config with dataset and optimization parameters.

    Returns:
        Optimization result with best hyperparameters.
    """
    from covenant_radar_api.core.config import settings_from_env

    settings = settings_from_env()

    # Get directories from settings
    data_root = Path(settings["app"]["data_root"])
    external_dir = data_root / "external"
    output_dir = Path(settings["app"]["models_root"]) / "optuna" / "lstm"

    result = run_lstm_optimization(config_json, external_dir, output_dir)

    # Convert to JSON-serializable dict
    config_json_dict: dict[str, JSONValue] = {
        "device": result["recommended_config"]["device"],
        "precision": result["recommended_config"]["precision"],
        "hidden_size": result["recommended_config"]["hidden_size"],
        "num_layers": result["recommended_config"]["num_layers"],
        "dropout": result["recommended_config"]["dropout"],
        "bidirectional": result["recommended_config"]["bidirectional"],
        "sequence_length": result["recommended_config"]["sequence_length"],
        "learning_rate": result["recommended_config"]["learning_rate"],
        "batch_size": result["recommended_config"]["batch_size"],
        "n_epochs": result["recommended_config"]["n_epochs"],
        "train_ratio": result["recommended_config"]["train_ratio"],
        "val_ratio": result["recommended_config"]["val_ratio"],
        "test_ratio": result["recommended_config"]["test_ratio"],
        "random_state": result["recommended_config"]["random_state"],
        "early_stopping_patience": result["recommended_config"]["early_stopping_patience"],
    }

    return {
        "status": result["status"],
        "dataset": result["dataset"],
        "n_samples": result["n_samples"],
        "n_features": result["n_features"],
        "feature_preset": result["feature_preset"],
        "n_trials_complete": result["n_trials_complete"],
        "n_trials_pruned": result["n_trials_pruned"],
        "n_trials_failed": result["n_trials_failed"],
        "best_trial_number": result["best_trial_number"],
        "best_val_auc": result["best_val_auc"],
        "best_hidden_size": result["best_hidden_size"],
        "best_num_layers": result["best_num_layers"],
        "best_learning_rate": result["best_learning_rate"],
        "best_dropout": result["best_dropout"],
        "best_batch_size": result["best_batch_size"],
        "duration_seconds": result["duration_seconds"],
        "recommended_config": config_json_dict,
    }


__all__ = [
    "LSTMOptimizationResult",
    "LSTMTrialProgressCallbackProtocol",
    "LSTMTrialProgressInfo",
    "process_lstm_optimize_job",
    "run_lstm_optimization",
]
