"""Background job for MLP hyperparameter optimization using Optuna TPE.

Runs Bayesian optimization on external bankruptcy datasets to find
optimal MLP hyperparameters. Results include best hyperparameters
and recommended MLPConfig for subsequent training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, TypedDict

from covenant_ml.backends.protocol import ProgressCallback
from covenant_ml.datasets.types import LoadPhase, LoadProgress
from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer import (
    MLPSearchSpace,
    OptimizationSummary,
    TrialResult,
    create_mlp_objective,
    create_mlp_optimizer,
)
from covenant_ml.types import MLPConfig, MLPOptimizer, MLPPrecision, TrainProgress
from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    load_json_str,
    require_int,
    require_str,
)
from platform_core.logging import get_logger

from covenant_radar_api.worker._optimize_common import (
    build_optimization_config,
    load_any_dataset,
    optional_int,
    parse_device,
    parse_feature_preset,
    save_optimization_results,
)

_log = get_logger(__name__)


SpaceProfile = Literal["default"]


def _parse_space_profile(raw: JSONValue | None) -> SpaceProfile:
    """Parse space profile, defaulting to 'default'.

    MLP optimization currently only supports the default profile.
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


def _parse_precision(raw: JSONValue | None) -> MLPPrecision:
    """Parse precision setting, defaulting to 'fp32'.

    Args:
        raw: Raw JSON value.

    Returns:
        MLPPrecision literal.

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


def _parse_optimizer(raw: JSONValue | None) -> MLPOptimizer:
    """Parse optimizer setting, defaulting to 'adamw'.

    Args:
        raw: Raw JSON value.

    Returns:
        MLPOptimizer literal.

    Raises:
        JSONTypeError: If value is not a string or invalid optimizer.
    """
    if raw is None:
        return "adamw"
    if not isinstance(raw, str):
        raise JSONTypeError("optimizer must be a string")
    if raw == "adamw":
        return "adamw"
    if raw == "adam":
        return "adam"
    if raw == "sgd":
        return "sgd"
    raise JSONTypeError("optimizer must be one of: adamw, adam, sgd")


class MLPOptimizeParseResult(TypedDict, total=True):
    """Parsed MLP optimization request."""

    dataset: str
    n_trials: int
    timeout_seconds: int | None
    device: Literal["cpu", "cuda", "auto"]
    space_profile: SpaceProfile
    feature_preset: FeaturePreset
    random_state: int
    precision: MLPPrecision
    optimizer: MLPOptimizer
    n_epochs: int
    early_stopping_patience: int


def _parse_optimize_config(config_json: str) -> MLPOptimizeParseResult:
    """Parse MLP optimization config from JSON string.

    Args:
        config_json: JSON string with optimization parameters.

    Returns:
        MLPOptimizeParseResult with all optimization parameters.

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
    optimizer = _parse_optimizer(raw.get("optimizer"))
    n_epochs = optional_int(raw, "n_epochs", 50)
    early_stopping_patience = optional_int(raw, "early_stopping_patience", 10)

    return MLPOptimizeParseResult(
        dataset=dataset_name,
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        device=device,
        space_profile=space_profile,
        feature_preset=feature_preset,
        random_state=random_state,
        precision=precision,
        optimizer=optimizer,
        n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
    )


def _get_search_space(profile: SpaceProfile) -> MLPSearchSpace:
    """Get MLP search space based on profile name.

    Args:
        profile: Space profile name (currently only 'default' supported).

    Returns:
        MLPSearchSpace with appropriate ranges.
    """
    from covenant_ml.optimizer import make_mlp_default_space

    # Currently only default space is supported
    # Focused space requires initial values from a previous optimization run
    _ = profile  # Used for future extension
    return make_mlp_default_space()


class MLPOptimizationResult(TypedDict, total=True):
    """Result of an MLP hyperparameter optimization run."""

    backend: Literal["mlp"]
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
    best_n_layers: int
    best_hidden_size: int
    best_learning_rate: float
    best_dropout: float
    best_batch_size: int
    duration_seconds: float
    recommended_config: MLPConfig


def _generate_mlp_config(
    summary: OptimizationSummary,
    device: Literal["cpu", "cuda", "auto"],
    precision: MLPPrecision,
    optimizer: MLPOptimizer,
    n_epochs: int,
    early_stopping_patience: int,
) -> MLPConfig:
    """Generate an MLPConfig from optimization summary.

    Args:
        summary: Optimization summary with best parameters.
        device: Device to use for training.
        precision: Precision mode.
        optimizer: Optimizer type.
        n_epochs: Number of training epochs.
        early_stopping_patience: Early stopping patience.

    Returns:
        MLPConfig ready for training.
    """
    best_int = summary["best_int_params"]
    best_float = summary["best_float_params"]
    n_layers = best_int.get("n_layers", 2)
    hidden_size = best_int.get("hidden_size", 64)
    batch_size = best_int.get("batch_size", 32)

    # Create hidden_sizes tuple with n_layers of hidden_size
    hidden_sizes = tuple([hidden_size] * n_layers)

    return MLPConfig(
        device=device,
        precision=precision,
        optimizer=optimizer,
        hidden_sizes=hidden_sizes,
        learning_rate=best_float["learning_rate"],
        batch_size=batch_size,
        n_epochs=n_epochs,
        dropout=best_float["dropout"],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_patience=early_stopping_patience,
    )


class MLPTrialProgressInfo(TypedDict):
    """Information about MLP trial progress for CLI display."""

    trial_number: int
    n_trials_total: int
    current_auc: float
    best_auc: float
    best_trial: int
    is_best: bool
    best_learning_rate: float
    best_n_layers: int
    best_hidden_size: int
    best_dropout: float


class MLPTrialProgressCallbackProtocol(Protocol):
    """Protocol for MLP trial progress callback."""

    def __call__(self, info: MLPTrialProgressInfo) -> None:
        """Called after each trial with progress info."""
        ...


class MLPPhaseInfo(TypedDict):
    """Information about optimization phase for CLI display."""

    phase: Literal["loading_data", "feature_engineering", "optimizing", "saving"]
    dataset: str
    n_samples: int
    n_features: int


class MLPPhaseCallbackProtocol(Protocol):
    """Protocol for MLP phase progress callback."""

    def __call__(self, info: MLPPhaseInfo) -> None:
        """Called when entering a new optimization phase."""
        ...


class MLPLoadingProgressInfo(TypedDict):
    """Progress information during dataset loading.

    Provides granular progress updates during the loading_data phase.
    """

    dataset: str
    phase: LoadPhase
    percent_complete: float
    rows_processed: int
    rows_total: int
    message: str


class MLPLoadingProgressCallbackProtocol(Protocol):
    """Protocol for loading progress callback during dataset loading."""

    def __call__(self, info: MLPLoadingProgressInfo) -> None:
        """Called with progress updates during dataset loading."""
        ...


class MLPEpochProgressInfo(TypedDict):
    """Progress information during epoch training within a trial.

    Provides per-epoch updates during MLP training to show
    training progress within each trial.
    """

    trial_number: int
    epoch: int
    total_epochs: int
    train_loss: float
    train_auc: float
    val_loss: float | None
    val_auc: float | None


class MLPEpochProgressCallbackProtocol(Protocol):
    """Protocol for epoch progress callback during MLP training."""

    def __call__(self, info: MLPEpochProgressInfo) -> None:
        """Called with progress updates during epoch training."""
        ...


def _default_mlp_epoch_callback(info: MLPEpochProgressInfo) -> None:
    """Default epoch callback that logs progress.

    Args:
        info: Epoch progress information.
    """
    val_auc_str = f"{info['val_auc']:.4f}" if info["val_auc"] is not None else "N/A"
    _log.debug(
        "MLP epoch progress",
        extra={
            "trial": info["trial_number"],
            "epoch": info["epoch"],
            "total_epochs": info["total_epochs"],
            "train_auc": f"{info['train_auc']:.4f}",
            "val_auc": val_auc_str,
        },
    )


def _report_mlp_phase(
    callback: MLPPhaseCallbackProtocol | None,
    phase: Literal["loading_data", "feature_engineering", "optimizing", "saving"],
    dataset: str,
    n_samples: int,
    n_features: int,
) -> None:
    """Report phase transition if callback is provided.

    Args:
        callback: Optional phase callback.
        phase: Phase name (loading_data, feature_engineering, optimizing).
        dataset: Dataset name.
        n_samples: Number of samples (0 during loading).
        n_features: Number of features (0 during loading).
    """
    if callback is not None:
        callback(
            MLPPhaseInfo(
                phase=phase,
                dataset=dataset,
                n_samples=n_samples,
                n_features=n_features,
            )
        )


def run_mlp_optimization(
    config_json: str,
    external_dir: Path,
    output_dir: Path,
    progress_callback: MLPTrialProgressCallbackProtocol | None = None,
    phase_callback: MLPPhaseCallbackProtocol | None = None,
    loading_progress_callback: MLPLoadingProgressCallbackProtocol | None = None,
    epoch_callback: MLPEpochProgressCallbackProtocol | None = None,
) -> MLPOptimizationResult:
    """Run MLP hyperparameter optimization on external dataset.

    Args:
        config_json: JSON config with dataset and optimization parameters.
        external_dir: Path to data/external directory with datasets.
        output_dir: Directory to save optimization results.
        progress_callback: Optional callback for trial progress updates.
        phase_callback: Optional callback for phase transitions (loading, optimizing, etc).
        loading_progress_callback: Optional callback for granular loading progress.
        epoch_callback: Optional callback for per-epoch training progress within trials.

    Returns:
        MLPOptimizationResult with best hyperparameters and recommended config.
    """
    parse_result = _parse_optimize_config(config_json)
    dataset_name = parse_result["dataset"]

    # Report loading phase
    _report_mlp_phase(phase_callback, "loading_data", dataset_name, 0, 0)

    # Create loading progress adapter - only used when loading_progress_callback is not None
    def _loading_progress_adapter(progress: LoadProgress) -> None:
        # Assertion to satisfy type narrowing - adapter only called when callback exists
        assert loading_progress_callback is not None
        loading_progress_callback(
            MLPLoadingProgressInfo(
                dataset=dataset_name,
                phase=progress["phase"],
                percent_complete=progress["percent_complete"],
                rows_processed=progress["rows_processed"],
                rows_total=progress["rows_total"],
                message=progress["message"],
            )
        )

    # Load raw dataset with progress reporting
    dataset = load_any_dataset(
        dataset_name, external_dir, _loading_progress_adapter if loading_progress_callback else None
    )

    # Report feature engineering phase
    _report_mlp_phase(
        phase_callback,
        "feature_engineering",
        dataset_name,
        dataset["meta"]["n_samples"],
        dataset["meta"]["n_features"],
    )

    _log.info(
        "Starting MLP hyperparameter optimization",
        extra={
            "dataset": dataset_name,
            "n_samples": dataset["meta"]["n_samples"],
            "n_features": dataset["meta"]["n_features"],
            "n_trials": parse_result["n_trials"],
            "space_profile": parse_result["space_profile"],
            "feature_preset": parse_result["feature_preset"],
            "device": parse_result["device"],
            "precision": parse_result["precision"],
        },
    )

    # Build config and search space
    config = build_optimization_config(
        n_trials=parse_result["n_trials"],
        timeout_seconds=parse_result["timeout_seconds"],
        random_state=parse_result["random_state"],
    )
    search_space = _get_search_space(parse_result["space_profile"])

    # Track current trial for epoch callback
    current_trial_number = 0

    # Create epoch callback adapter - uses default logging if none provided
    def _make_epoch_adapter() -> ProgressCallback:
        effective_callback = (
            epoch_callback if epoch_callback is not None else _default_mlp_epoch_callback
        )

        def _epoch_adapter(progress: TrainProgress) -> None:
            effective_callback(
                MLPEpochProgressInfo(
                    trial_number=current_trial_number,
                    epoch=progress["round"],
                    total_epochs=progress["total_rounds"],
                    train_loss=progress["train_loss"],
                    train_auc=progress["train_auc"],
                    val_loss=progress["val_loss"],
                    val_auc=progress["val_auc"],
                )
            )

        return _epoch_adapter

    epoch_adapter = _make_epoch_adapter()

    # Create objective function (applies feature engineering if preset != "none")
    objective = create_mlp_objective(
        dataset["x"],
        dataset["y"],
        list(dataset["meta"]["feature_names"]),
        parse_result["device"],
        parse_result["precision"],
        parse_result["feature_preset"],
        parse_result["n_epochs"],
        parse_result["early_stopping_patience"],
        optimizer_name=parse_result["optimizer"],
        epoch_callback=epoch_adapter,
    )

    # Report optimizing phase
    _report_mlp_phase(
        phase_callback,
        "optimizing",
        dataset_name,
        dataset["meta"]["n_samples"],
        objective.n_features,
    )

    # Track progress
    best_auc = 0.0
    best_trial_num = 0
    best_learning_rate = 0.0
    best_n_layers = 0
    best_hidden_size = 0
    best_dropout = 0.0
    trials_seen = 0
    n_trials_total = parse_result["n_trials"]

    def trial_callback(result: TrialResult) -> None:
        nonlocal best_auc
        nonlocal best_trial_num
        nonlocal trials_seen
        nonlocal best_learning_rate
        nonlocal best_n_layers
        nonlocal best_hidden_size
        nonlocal best_dropout
        nonlocal current_trial_number
        trials_seen += 1
        # Update trial number for next trial's epoch callbacks
        current_trial_number = result["trial_number"] + 1
        auc = result["value"]
        is_best = auc > best_auc
        if is_best:
            best_auc = auc
            best_trial_num = result["trial_number"]
            trial_int_params = result["int_params"]
            trial_float_params = result["float_params"]
            best_learning_rate = trial_float_params["learning_rate"]
            best_n_layers = trial_int_params.get("n_layers", 2)
            best_hidden_size = trial_int_params.get("hidden_size", 64)
            best_dropout = trial_float_params["dropout"]
            _log.info(
                "New best MLP trial",
                extra={
                    "trial": result["trial_number"],
                    "auc": f"{auc:.4f}",
                    "n_layers": best_n_layers,
                    "hidden_size": best_hidden_size,
                    "learning_rate": f"{trial_float_params['learning_rate']:.4f}",
                    "dropout": f"{best_dropout:.2f}",
                },
            )

        # Call progress callback if provided
        if progress_callback is not None:
            progress_info: MLPTrialProgressInfo = {
                "trial_number": result["trial_number"],
                "n_trials_total": n_trials_total,
                "current_auc": auc,
                "best_auc": best_auc,
                "best_trial": best_trial_num,
                "is_best": is_best,
                "best_learning_rate": best_learning_rate,
                "best_n_layers": best_n_layers,
                "best_hidden_size": best_hidden_size,
                "best_dropout": best_dropout,
            }
            progress_callback(progress_info)

    # Run optimization
    optimizer = create_mlp_optimizer()
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
        "MLP optimization complete",
        extra={
            "dataset": dataset_name,
            "best_trial": summary["best_trial_number"],
            "best_auc": f"{summary['best_value']:.4f}",
            "n_trials_complete": summary["n_trials_complete"],
            "duration_seconds": f"{summary['total_duration_seconds']:.1f}",
        },
    )

    # Generate recommended config
    recommended_config = _generate_mlp_config(
        summary,
        parse_result["device"],
        parse_result["precision"],
        parse_result["optimizer"],
        parse_result["n_epochs"],
        parse_result["early_stopping_patience"],
    )

    # Build serializable result and config
    best_int_params = summary["best_int_params"]
    best_float_params = summary["best_float_params"]

    result_dict: dict[str, JSONValue] = {
        "dataset": dataset_name,
        "n_samples": dataset["meta"]["n_samples"],
        "n_features": objective.n_features,
        "best_trial": summary["best_trial_number"],
        "best_val_auc": summary["best_value"],
        "best_n_layers": best_int_params.get("n_layers", 2),
        "best_hidden_size": best_int_params.get("hidden_size", 64),
        "best_batch_size": best_int_params.get("batch_size", 32),
        "best_learning_rate": best_float_params["learning_rate"],
        "best_dropout": best_float_params["dropout"],
        "n_trials_complete": summary["n_trials_complete"],
        "duration_seconds": summary["total_duration_seconds"],
    }

    # Convert hidden_sizes tuple to list for JSON serialization
    hidden_sizes_json: list[JSONValue] = [int(s) for s in recommended_config["hidden_sizes"]]

    config_dict: dict[str, JSONValue] = {
        "device": recommended_config["device"],
        "precision": recommended_config["precision"],
        "optimizer": recommended_config["optimizer"],
        "hidden_sizes": hidden_sizes_json,
        "learning_rate": recommended_config["learning_rate"],
        "batch_size": recommended_config["batch_size"],
        "n_epochs": recommended_config["n_epochs"],
        "dropout": recommended_config["dropout"],
        "train_ratio": recommended_config["train_ratio"],
        "val_ratio": recommended_config["val_ratio"],
        "test_ratio": recommended_config["test_ratio"],
        "random_state": recommended_config["random_state"],
        "early_stopping_patience": recommended_config["early_stopping_patience"],
    }

    # Save results to output directory
    result_path, config_path = save_optimization_results(
        output_dir, dataset_name, "mlp", result_dict, config_dict
    )

    _log.info(
        "Saved MLP optimization results",
        extra={
            "result_path": str(result_path),
            "config_path": str(config_path),
        },
    )

    return MLPOptimizationResult(
        backend="mlp",
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
        best_n_layers=best_int_params.get("n_layers", 2),
        best_hidden_size=best_int_params.get("hidden_size", 64),
        best_learning_rate=best_float_params["learning_rate"],
        best_dropout=best_float_params["dropout"],
        best_batch_size=best_int_params.get("batch_size", 32),
        duration_seconds=summary["total_duration_seconds"],
        recommended_config=recommended_config,
    )


def process_mlp_optimize_job(config_json: str) -> dict[str, JSONValue]:
    """RQ job entry point for MLP hyperparameter optimization.

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
    output_dir = Path(settings["app"]["models_root"]) / "optuna" / "mlp"

    result = run_mlp_optimization(config_json, external_dir, output_dir)

    # Convert hidden_sizes tuple to list for JSON serialization
    rec_cfg = result["recommended_config"]
    hidden_sizes_json: list[JSONValue] = [int(s) for s in rec_cfg["hidden_sizes"]]

    # Convert to JSON-serializable dict
    config_json_dict: dict[str, JSONValue] = {
        "device": result["recommended_config"]["device"],
        "precision": result["recommended_config"]["precision"],
        "optimizer": result["recommended_config"]["optimizer"],
        "hidden_sizes": hidden_sizes_json,
        "learning_rate": result["recommended_config"]["learning_rate"],
        "batch_size": result["recommended_config"]["batch_size"],
        "n_epochs": result["recommended_config"]["n_epochs"],
        "dropout": result["recommended_config"]["dropout"],
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
        "best_n_layers": result["best_n_layers"],
        "best_hidden_size": result["best_hidden_size"],
        "best_learning_rate": result["best_learning_rate"],
        "best_dropout": result["best_dropout"],
        "best_batch_size": result["best_batch_size"],
        "duration_seconds": result["duration_seconds"],
        "recommended_config": config_json_dict,
    }


__all__ = [
    "MLPEpochProgressCallbackProtocol",
    "MLPEpochProgressInfo",
    "MLPLoadingProgressCallbackProtocol",
    "MLPLoadingProgressInfo",
    "MLPOptimizationResult",
    "MLPPhaseCallbackProtocol",
    "MLPPhaseInfo",
    "MLPTrialProgressCallbackProtocol",
    "MLPTrialProgressInfo",
    "process_mlp_optimize_job",
    "run_mlp_optimization",
]
