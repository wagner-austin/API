"""Background job for LightGBM hyperparameter optimization using Optuna TPE.

Runs Bayesian optimization on external bankruptcy datasets to find
optimal LightGBM hyperparameters. Results include best hyperparameters
and recommended LightGBMConfig for subsequent training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, TypedDict

from covenant_ml.datasets.types import LoadPhase, LoadProgress
from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer import (
    LightGBMSearchSpace,
    OptimizationSummary,
    TrialResult,
    create_lightgbm_objective,
    create_lightgbm_optimizer,
)
from covenant_ml.types import LightGBMConfig
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
    load_any_dataset,
    optional_int,
    parse_device,
    parse_feature_preset,
)

_log = get_logger(__name__)


SpaceProfile = Literal["default"]


def _parse_space_profile(raw: JSONValue | None) -> SpaceProfile:
    """Parse space profile, defaulting to 'default'.

    LightGBM optimization currently only supports the default profile.
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


class LightGBMOptimizeParseResult(TypedDict, total=True):
    """Parsed LightGBM optimization request."""

    dataset: str
    n_trials: int
    timeout_seconds: int | None
    device: Literal["cpu", "cuda", "auto"]
    space_profile: SpaceProfile
    feature_preset: FeaturePreset
    random_state: int
    early_stopping_rounds: int
    n_jobs: int


def _parse_optimize_config(config_json: str) -> LightGBMOptimizeParseResult:
    """Parse LightGBM optimization config from JSON string.

    Args:
        config_json: JSON string with optimization parameters.

    Returns:
        LightGBMOptimizeParseResult with all optimization parameters.

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
    early_stopping_rounds = optional_int(raw, "early_stopping_rounds", 10)
    n_jobs = optional_int(raw, "n_jobs", -1)

    return LightGBMOptimizeParseResult(
        dataset=dataset_name,
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        device=device,
        space_profile=space_profile,
        feature_preset=feature_preset,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
    )


def _get_search_space(profile: SpaceProfile) -> LightGBMSearchSpace:
    """Get LightGBM search space based on profile name.

    Args:
        profile: Space profile name (currently only 'default' supported).

    Returns:
        LightGBMSearchSpace with appropriate ranges.
    """
    from covenant_ml.optimizer import make_lightgbm_default_space

    # Currently only default space is supported
    # Focused space requires initial values from a previous optimization run
    _ = profile  # Used for future extension
    return make_lightgbm_default_space()


class LightGBMOptimizationResult(TypedDict, total=True):
    """Result of a LightGBM hyperparameter optimization run.

    Note: best_max_depth is always -1 (unlimited). LightGBM optimization uses
    num_leaves as the primary complexity control to avoid constraint conflicts.
    """

    backend: Literal["lightgbm"]
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
    best_max_depth: Literal[-1]  # Fixed: unlimited depth, num_leaves controls complexity
    best_n_estimators: int
    best_num_leaves: int
    best_learning_rate: float
    best_reg_alpha: float
    best_reg_lambda: float
    best_subsample: float
    best_colsample_bytree: float
    duration_seconds: float
    recommended_config: LightGBMConfig


def _generate_lightgbm_config(
    summary: OptimizationSummary,
    device: Literal["cpu", "cuda", "auto"],
    early_stopping_rounds: int,
) -> LightGBMConfig:
    """Generate a LightGBMConfig from optimization summary.

    Args:
        summary: Optimization summary with best parameters.
        device: Device to use for training.
        early_stopping_rounds: Early stopping rounds for training.

    Returns:
        LightGBMConfig ready for training.
    """
    best_int = summary["best_int_params"]
    best_float = summary["best_float_params"]
    # Handle optional num_leaves - use default if not present
    num_leaves = best_int.get("num_leaves", 31)
    min_child_samples = best_int.get("min_child_samples", 20)

    # max_depth is fixed at -1 (unlimited) - num_leaves controls tree complexity
    return LightGBMConfig(
        device=device,
        learning_rate=best_float["learning_rate"],
        max_depth=-1,
        n_estimators=best_int["n_estimators"],
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=best_float["subsample"],
        colsample_bytree=best_float["colsample_bytree"],
        reg_alpha=best_float["reg_alpha"],
        reg_lambda=best_float["reg_lambda"],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_rounds=early_stopping_rounds,
    )


class LightGBMTrialProgressInfo(TypedDict):
    """Information about LightGBM trial progress for CLI display.

    Note: max_depth is not tracked since it's fixed at -1 (unlimited).
    """

    trial_number: int
    n_trials_total: int
    current_auc: float
    best_auc: float
    best_trial: int
    is_best: bool
    best_learning_rate: float
    best_n_estimators: int
    best_num_leaves: int


class LightGBMTrialProgressCallbackProtocol(Protocol):
    """Protocol for LightGBM trial progress callback."""

    def __call__(self, info: LightGBMTrialProgressInfo) -> None:
        """Called after each trial with progress info."""
        ...


class LightGBMPhaseInfo(TypedDict):
    """Information about optimization phase for CLI display."""

    phase: Literal["loading_data", "feature_engineering", "optimizing", "saving"]
    dataset: str
    n_samples: int
    n_features: int


class LightGBMPhaseCallbackProtocol(Protocol):
    """Protocol for LightGBM phase progress callback."""

    def __call__(self, info: LightGBMPhaseInfo) -> None:
        """Called when entering a new optimization phase."""
        ...


class LightGBMLoadingProgressInfo(TypedDict):
    """Progress information during dataset loading.

    Provides granular progress updates during the loading_data phase.
    """

    dataset: str
    phase: LoadPhase
    percent_complete: float
    rows_processed: int
    rows_total: int
    message: str


class LightGBMLoadingProgressCallbackProtocol(Protocol):
    """Protocol for loading progress callback during dataset loading."""

    def __call__(self, info: LightGBMLoadingProgressInfo) -> None:
        """Called with progress updates during dataset loading."""
        ...


def run_lightgbm_optimization(
    config_json: str,
    external_dir: Path,
    output_dir: Path,
    progress_callback: LightGBMTrialProgressCallbackProtocol | None = None,
    phase_callback: LightGBMPhaseCallbackProtocol | None = None,
    loading_progress_callback: LightGBMLoadingProgressCallbackProtocol | None = None,
) -> LightGBMOptimizationResult:
    """Run LightGBM hyperparameter optimization on external dataset.

    Args:
        config_json: JSON config with dataset and optimization parameters.
        external_dir: Path to data/external directory with datasets.
        output_dir: Directory to save optimization results.
        progress_callback: Optional callback for trial progress updates.
        phase_callback: Optional callback for phase transitions (loading, optimizing, etc).
        loading_progress_callback: Optional callback for granular loading progress.

    Returns:
        LightGBMOptimizationResult with best hyperparameters and recommended config.
    """
    parse_result = _parse_optimize_config(config_json)
    dataset_name = parse_result["dataset"]

    # Report loading phase
    if phase_callback is not None:
        phase_callback(
            LightGBMPhaseInfo(
                phase="loading_data",
                dataset=dataset_name,
                n_samples=0,
                n_features=0,
            )
        )

    # Create loading progress adapter - only used when loading_progress_callback is not None
    def _loading_progress_adapter(progress: LoadProgress) -> None:
        # Assertion to satisfy type narrowing - adapter only called when callback exists
        assert loading_progress_callback is not None
        loading_progress_callback(
            LightGBMLoadingProgressInfo(
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
    if phase_callback is not None:
        phase_callback(
            LightGBMPhaseInfo(
                phase="feature_engineering",
                dataset=dataset_name,
                n_samples=dataset["meta"]["n_samples"],
                n_features=dataset["meta"]["n_features"],
            )
        )

    _log.info(
        "Starting LightGBM hyperparameter optimization",
        extra={
            "dataset": dataset_name,
            "n_samples": dataset["meta"]["n_samples"],
            "n_features": dataset["meta"]["n_features"],
            "n_trials": parse_result["n_trials"],
            "space_profile": parse_result["space_profile"],
            "feature_preset": parse_result["feature_preset"],
            "device": parse_result["device"],
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
    objective = create_lightgbm_objective(
        dataset["x"],
        dataset["y"],
        list(dataset["meta"]["feature_names"]),
        parse_result["device"],
        parse_result["feature_preset"],
        early_stopping_rounds=parse_result["early_stopping_rounds"],
        n_jobs=parse_result["n_jobs"],
    )

    # Report optimizing phase
    if phase_callback is not None:
        phase_callback(
            LightGBMPhaseInfo(
                phase="optimizing",
                dataset=dataset_name,
                n_samples=dataset["meta"]["n_samples"],
                n_features=objective.n_features,
            )
        )

    # Track progress
    # Note: max_depth is not tracked since it's fixed at -1 (unlimited)
    best_auc = 0.0
    best_trial_num = 0
    best_learning_rate = 0.0
    best_n_estimators = 0
    best_num_leaves = 0
    trials_seen = 0
    n_trials_total = parse_result["n_trials"]

    def trial_callback(result: TrialResult) -> None:
        nonlocal best_auc
        nonlocal best_trial_num
        nonlocal trials_seen
        nonlocal best_learning_rate
        nonlocal best_n_estimators
        nonlocal best_num_leaves
        trials_seen += 1
        auc = result["value"]
        is_best = auc > best_auc
        if is_best:
            best_auc = auc
            best_trial_num = result["trial_number"]
            trial_int_params = result["int_params"]
            trial_float_params = result["float_params"]
            best_learning_rate = trial_float_params["learning_rate"]
            best_n_estimators = trial_int_params["n_estimators"]
            best_num_leaves = trial_int_params.get("num_leaves", 31)
            _log.info(
                "New best LightGBM trial",
                extra={
                    "trial": result["trial_number"],
                    "auc": f"{auc:.4f}",
                    "num_leaves": best_num_leaves,
                    "learning_rate": f"{trial_float_params['learning_rate']:.4f}",
                    "n_estimators": trial_int_params["n_estimators"],
                },
            )

        # Call progress callback if provided
        if progress_callback is not None:
            progress_info: LightGBMTrialProgressInfo = {
                "trial_number": result["trial_number"],
                "n_trials_total": n_trials_total,
                "current_auc": auc,
                "best_auc": best_auc,
                "best_trial": best_trial_num,
                "is_best": is_best,
                "best_learning_rate": best_learning_rate,
                "best_n_estimators": best_n_estimators,
                "best_num_leaves": best_num_leaves,
            }
            progress_callback(progress_info)

    # Run optimization
    optimizer = create_lightgbm_optimizer()
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
        "LightGBM optimization complete",
        extra={
            "dataset": dataset_name,
            "best_trial": summary["best_trial_number"],
            "best_auc": f"{summary['best_value']:.4f}",
            "n_trials_complete": summary["n_trials_complete"],
            "duration_seconds": f"{summary['total_duration_seconds']:.1f}",
        },
    )

    # Generate recommended config
    recommended_config = _generate_lightgbm_config(
        summary, parse_result["device"], parse_result["early_stopping_rounds"]
    )

    # Save results to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{dataset_name}_lightgbm_optuna_result.json"
    config_path = output_dir / f"{dataset_name}_lightgbm_optimal_config.json"

    # Build serializable result (use objective.n_features for actual engineered count)
    best_int_params = summary["best_int_params"]
    best_float_params = summary["best_float_params"]

    # max_depth is fixed at -1 (unlimited) - num_leaves controls tree complexity
    result_dict: dict[str, JSONValue] = {
        "dataset": dataset_name,
        "n_samples": dataset["meta"]["n_samples"],
        "n_features": objective.n_features,
        "best_trial": summary["best_trial_number"],
        "best_val_auc": summary["best_value"],
        "best_max_depth": -1,
        "best_n_estimators": best_int_params["n_estimators"],
        "best_num_leaves": best_int_params.get("num_leaves", 31),
        "best_learning_rate": best_float_params["learning_rate"],
        "best_reg_alpha": best_float_params["reg_alpha"],
        "best_reg_lambda": best_float_params["reg_lambda"],
        "best_subsample": best_float_params["subsample"],
        "best_colsample_bytree": best_float_params["colsample_bytree"],
        "n_trials_complete": summary["n_trials_complete"],
        "duration_seconds": summary["total_duration_seconds"],
    }

    # Write results
    with open(result_path, "w") as f:
        f.write(dump_json_str(result_dict))

    config_dict: dict[str, JSONValue] = {
        "device": recommended_config["device"],
        "learning_rate": recommended_config["learning_rate"],
        "max_depth": recommended_config["max_depth"],
        "n_estimators": recommended_config["n_estimators"],
        "num_leaves": recommended_config["num_leaves"],
        "min_child_samples": recommended_config["min_child_samples"],
        "subsample": recommended_config["subsample"],
        "colsample_bytree": recommended_config["colsample_bytree"],
        "reg_alpha": recommended_config["reg_alpha"],
        "reg_lambda": recommended_config["reg_lambda"],
        "train_ratio": recommended_config["train_ratio"],
        "val_ratio": recommended_config["val_ratio"],
        "test_ratio": recommended_config["test_ratio"],
        "random_state": recommended_config["random_state"],
        "early_stopping_rounds": recommended_config["early_stopping_rounds"],
    }

    with open(config_path, "w") as f:
        f.write(dump_json_str(config_dict))

    _log.info(
        "Saved LightGBM optimization results",
        extra={
            "result_path": str(result_path),
            "config_path": str(config_path),
        },
    )

    return LightGBMOptimizationResult(
        backend="lightgbm",
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
        best_max_depth=-1,  # Fixed: unlimited depth, num_leaves controls complexity
        best_n_estimators=best_int_params["n_estimators"],
        best_num_leaves=best_int_params.get("num_leaves", 31),
        best_learning_rate=best_float_params["learning_rate"],
        best_reg_alpha=best_float_params["reg_alpha"],
        best_reg_lambda=best_float_params["reg_lambda"],
        best_subsample=best_float_params["subsample"],
        best_colsample_bytree=best_float_params["colsample_bytree"],
        duration_seconds=summary["total_duration_seconds"],
        recommended_config=recommended_config,
    )


def process_lightgbm_optimize_job(config_json: str) -> dict[str, JSONValue]:
    """RQ job entry point for LightGBM hyperparameter optimization.

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
    output_dir = Path(settings["app"]["models_root"]) / "optuna" / "lightgbm"

    result = run_lightgbm_optimization(config_json, external_dir, output_dir)

    # Convert to JSON-serializable dict
    config_json_dict: dict[str, JSONValue] = {
        "device": result["recommended_config"]["device"],
        "learning_rate": result["recommended_config"]["learning_rate"],
        "max_depth": result["recommended_config"]["max_depth"],
        "n_estimators": result["recommended_config"]["n_estimators"],
        "num_leaves": result["recommended_config"]["num_leaves"],
        "min_child_samples": result["recommended_config"]["min_child_samples"],
        "subsample": result["recommended_config"]["subsample"],
        "colsample_bytree": result["recommended_config"]["colsample_bytree"],
        "reg_alpha": result["recommended_config"]["reg_alpha"],
        "reg_lambda": result["recommended_config"]["reg_lambda"],
        "train_ratio": result["recommended_config"]["train_ratio"],
        "val_ratio": result["recommended_config"]["val_ratio"],
        "test_ratio": result["recommended_config"]["test_ratio"],
        "random_state": result["recommended_config"]["random_state"],
        "early_stopping_rounds": result["recommended_config"]["early_stopping_rounds"],
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
        "best_max_depth": result["best_max_depth"],
        "best_n_estimators": result["best_n_estimators"],
        "best_num_leaves": result["best_num_leaves"],
        "best_learning_rate": result["best_learning_rate"],
        "best_reg_alpha": result["best_reg_alpha"],
        "best_reg_lambda": result["best_reg_lambda"],
        "best_subsample": result["best_subsample"],
        "best_colsample_bytree": result["best_colsample_bytree"],
        "duration_seconds": result["duration_seconds"],
        "recommended_config": config_json_dict,
    }


__all__ = [
    "LightGBMLoadingProgressCallbackProtocol",
    "LightGBMLoadingProgressInfo",
    "LightGBMOptimizationResult",
    "LightGBMPhaseCallbackProtocol",
    "LightGBMPhaseInfo",
    "LightGBMTrialProgressCallbackProtocol",
    "LightGBMTrialProgressInfo",
    "process_lightgbm_optimize_job",
    "run_lightgbm_optimization",
]
