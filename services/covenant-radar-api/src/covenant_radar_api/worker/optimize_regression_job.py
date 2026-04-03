"""Background job for regression hyperparameter optimization using Optuna TPE.

Runs Bayesian optimization on regression datasets for regressor backends.
Parallel to optimize_job.py (classifier) with clear separation of concerns.

The backend's search space comes from RegressorBackend.get_default_search_space(),
and objectives are created via the _regression_hooks.regressor_objective_factory hook.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from covenant_ml.datasets.types import LoadProgress
from covenant_ml.optimizer import OptimizationSummary, SearchSpace, TrialResult
from covenant_ml.optimizer.protocol import TrialCallbackProtocol as _TrialCallbackProtocol
from covenant_ml.types import RegressorBackendName
from numpy.typing import NDArray
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
    optional_int,
    parse_bidirectional,
    parse_device,
    parse_feature_preset,
    parse_nn_optimizer,
    parse_precision,
    save_optimization_results,
)
from covenant_radar_api.worker._optimize_regression_common import (
    load_regression_dataset,
    parse_regression_dataset_name,
    parse_regressor_backend_name,
)
from covenant_radar_api.worker.optimize_regression_types import (
    RegressionLoadingProgressCallbackProtocol,
    RegressionLoadingProgressInfo,
    RegressionOptimizePhase,
    RegressionPhaseProgressCallbackProtocol,
    RegressionPhaseProgressInfo,
    RegressionTrialProgressCallbackProtocol,
    RegressionTrialProgressInfo,
    UnifiedRegressionOptimizationResult,
    UnifiedRegressionOptimizeParseResult,
    encode_unified_regression_optimization_result,
)

_log = get_logger(__name__)


# =============================================================================
# Config Parser
# =============================================================================


def _parse_regression_optimize_config(
    config_json: str,
) -> UnifiedRegressionOptimizeParseResult:
    """Parse regression optimization config from JSON string.

    Args:
        config_json: JSON string with optimization parameters.

    Returns:
        UnifiedRegressionOptimizeParseResult with all parameters.

    Raises:
        JSONTypeError: If config is not a JSON object or has invalid fields.
        ValueError: If dataset or backend name is invalid.
    """
    raw = load_json_str(config_json)
    if not isinstance(raw, dict):
        raise JSONTypeError("config must be a JSON object")

    backend = parse_regressor_backend_name(raw.get("backend"))
    dataset = require_str(raw, "dataset")
    dataset_name = parse_regression_dataset_name(dataset)
    n_trials = require_int(raw, "n_trials")

    timeout_raw = raw.get("timeout_seconds")
    timeout_seconds: int | None = None
    if timeout_raw is not None:
        if not isinstance(timeout_raw, int):
            raise JSONTypeError("timeout_seconds must be an integer or null")
        timeout_seconds = timeout_raw

    device = parse_device(raw.get("device"))
    feature_preset = parse_feature_preset(raw.get("feature_preset"))
    random_state = optional_int(raw, "random_state", 42)
    early_stopping_rounds = optional_int(raw, "early_stopping_rounds", 10)
    n_jobs = optional_int(raw, "n_jobs", -1)

    # Neural network fields (defaults match classifier optimize_job)
    precision = parse_precision(raw.get("precision"))
    nn_optimizer = parse_nn_optimizer(raw.get("nn_optimizer"))
    n_epochs = optional_int(raw, "n_epochs", 50)
    early_stopping_patience = optional_int(raw, "early_stopping_patience", 10)
    sequence_length = optional_int(raw, "sequence_length", 5)
    bidirectional = parse_bidirectional(raw.get("bidirectional"))

    return UnifiedRegressionOptimizeParseResult(
        backend=backend,
        dataset=dataset_name,
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        device=device,
        feature_preset=feature_preset,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
        precision=precision,
        nn_optimizer=nn_optimizer,
        n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
        sequence_length=sequence_length,
        bidirectional=bidirectional,
    )


def _report_regression_phase(
    phase_callback: RegressionPhaseProgressCallbackProtocol | None,
    phase: RegressionOptimizePhase,
    backend_name: RegressorBackendName,
    dataset_name: str,
    n_samples: int,
    n_features: int,
) -> None:
    """Send phase progress update if callback is provided.

    Args:
        phase_callback: Optional callback for phase transitions.
        phase: Current optimization phase.
        backend_name: Regressor backend being optimized.
        dataset_name: Dataset being used.
        n_samples: Number of samples in dataset.
        n_features: Number of features (0 if not yet known).
    """
    if phase_callback is not None:
        phase_callback(
            RegressionPhaseProgressInfo(
                phase=phase,
                backend=backend_name,
                dataset=dataset_name,
                n_samples=n_samples,
                n_features=n_features,
            )
        )


def _make_regression_trial_callback(
    backend_name: RegressorBackendName,
    n_trials_total: int,
    progress_callback: RegressionTrialProgressCallbackProtocol | None,
) -> _TrialCallbackProtocol:
    """Create a trial callback that tracks best value and reports progress.

    Args:
        backend_name: Regressor backend being optimized.
        n_trials_total: Total number of trials configured.
        progress_callback: Optional external callback for trial progress.

    Returns:
        Trial callback function for the optimizer.
    """
    best_value = -float("inf")
    best_trial_num = 0

    def _callback(result: TrialResult) -> None:
        nonlocal best_value
        nonlocal best_trial_num
        value = result["value"]
        is_best = value > best_value
        if is_best:
            best_value = value
            best_trial_num = result["trial_number"]
            _log.info(
                "New best regression trial",
                extra={
                    "backend": backend_name,
                    "trial": result["trial_number"],
                    "value": f"{value:.6f}",
                },
            )

        if progress_callback is not None:
            progress_callback(
                RegressionTrialProgressInfo(
                    backend=backend_name,
                    trial_number=result["trial_number"],
                    n_trials_total=n_trials_total,
                    current_value=value,
                    best_value=best_value,
                    best_trial=best_trial_num,
                    is_best=is_best,
                )
            )

    return _callback


# =============================================================================
# Core Optimization Function
# =============================================================================


def run_regression_optimization(
    config_json: str,
    external_dir: Path,
    output_dir: Path,
    progress_callback: RegressionTrialProgressCallbackProtocol | None = None,
    phase_callback: RegressionPhaseProgressCallbackProtocol | None = None,
    loading_progress_callback: RegressionLoadingProgressCallbackProtocol | None = None,
) -> UnifiedRegressionOptimizationResult:
    """Run hyperparameter optimization for a regressor backend.

    Dispatches to the correct backend via the regressor registry and
    regression objective factory hooks.

    Args:
        config_json: JSON config with dataset and optimization parameters.
        external_dir: Path to data/external directory with datasets.
        output_dir: Directory to save optimization results.
        progress_callback: Optional callback for trial progress updates.
        phase_callback: Optional callback for phase transitions.
        loading_progress_callback: Optional callback for granular loading progress.

    Returns:
        UnifiedRegressionOptimizationResult with best hyperparameters.
    """
    from covenant_radar_api.worker import _regression_hooks as hooks
    from covenant_radar_api.worker import _test_hooks as classifier_hooks

    parse_result = _parse_regression_optimize_config(config_json)
    backend_name = parse_result["backend"]
    dataset_name = parse_result["dataset"]

    # Get search space from regressor backend protocol
    regressor_registry = hooks.regressor_registry_factory()
    backend = regressor_registry.get(backend_name)
    search_space: SearchSpace = backend.get_default_search_space()

    # Report loading phase
    _report_regression_phase(phase_callback, "loading_data", backend_name, dataset_name, 0, 0)

    # Create loading progress adapter
    def _loading_progress_adapter(progress: LoadProgress) -> None:
        assert loading_progress_callback is not None
        loading_progress_callback(
            RegressionLoadingProgressInfo(
                dataset=dataset_name,
                phase=progress["phase"],
                percent_complete=progress["percent_complete"],
                rows_processed=progress["rows_processed"],
                rows_total=progress["rows_total"],
                message=progress["message"],
            )
        )

    # Load regression dataset (float64 y)
    dataset = load_regression_dataset(
        dataset_name,
        external_dir,
        _loading_progress_adapter if loading_progress_callback else None,
    )

    n_samples = dataset["meta"]["n_samples"]
    n_features_raw = dataset["meta"]["n_features"]

    # Report feature engineering phase
    _report_regression_phase(
        phase_callback,
        "feature_engineering",
        backend_name,
        dataset_name,
        n_samples,
        n_features_raw,
    )

    # Create regressor objective via hook (passes float64 y to constructor)
    objective = hooks.regressor_objective_factory(
        backend_name,
        dataset["x"],
        dataset["y"],
        list(dataset["meta"]["feature_names"]),
        parse_result,
    )

    _log.info(
        "Starting regression hyperparameter optimization",
        extra={
            "backend": backend_name,
            "dataset": dataset_name,
            "n_samples": n_samples,
            "n_features": objective.n_features,
            "n_trials": parse_result["n_trials"],
            "feature_preset": parse_result["feature_preset"],
            "device": parse_result["device"],
        },
    )

    # Report optimizing phase
    _report_regression_phase(
        phase_callback,
        "optimizing",
        backend_name,
        dataset_name,
        n_samples,
        objective.n_features,
    )

    # Build optimization config
    config = build_optimization_config(
        n_trials=parse_result["n_trials"],
        timeout_seconds=parse_result["timeout_seconds"],
        random_state=parse_result["random_state"],
    )

    # Create trial callback (best_value starts at -inf for negative RMSE)
    trial_callback = _make_regression_trial_callback(
        backend_name,
        parse_result["n_trials"],
        progress_callback,
    )

    # Create int64 zeros for optimizer protocol compatibility
    # (objective ignores y in __call__, uses pre-split data from __init__)
    y_zeros: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)

    # Run optimization via strategy registry
    optimizer = classifier_hooks.optimizer_registry_factory().get("optuna_tpe")
    summary: OptimizationSummary = optimizer.optimize(
        x_features=dataset["x"],
        y_labels=y_zeros,
        feature_names=list(dataset["meta"]["feature_names"]),
        search_space=search_space,
        config=config,
        objective=objective,
        trial_callback=trial_callback,
    )

    _log.info(
        "Regression optimization complete",
        extra={
            "backend": backend_name,
            "dataset": dataset_name,
            "best_trial": summary["best_trial_number"],
            "best_value": f"{summary['best_value']:.6f}",
            "n_trials_complete": summary["n_trials_complete"],
            "duration_seconds": f"{summary['total_duration_seconds']:.1f}",
        },
    )

    # Build result dict for saving
    result_dict: dict[str, JSONValue] = {
        "backend": backend_name,
        "dataset": dataset_name,
        "n_samples": n_samples,
        "n_features": objective.n_features,
        "feature_preset": parse_result["feature_preset"],
        "best_trial": summary["best_trial_number"],
        "best_value": summary["best_value"],
        "n_trials_complete": summary["n_trials_complete"],
        "n_trials_pruned": summary["n_trials_pruned"],
        "n_trials_failed": summary["n_trials_failed"],
        "duration_seconds": summary["total_duration_seconds"],
    }
    config_dict: dict[str, JSONValue] = dict(result_dict)

    # Save results
    save_optimization_results(
        output_dir,
        dataset_name,
        backend_name,
        result_dict,
        config_dict,
    )

    # Report saving phase
    _report_regression_phase(
        phase_callback,
        "saving",
        backend_name,
        dataset_name,
        n_samples,
        objective.n_features,
    )

    return UnifiedRegressionOptimizationResult(
        backend=backend_name,
        status="complete",
        dataset=dataset_name,
        n_samples=n_samples,
        n_features=objective.n_features,
        feature_preset=parse_result["feature_preset"],
        n_trials_complete=summary["n_trials_complete"],
        n_trials_pruned=summary["n_trials_pruned"],
        n_trials_failed=summary["n_trials_failed"],
        best_trial_number=summary["best_trial_number"],
        best_value=summary["best_value"],
        best_int_params=summary["best_int_params"],
        best_float_params=summary["best_float_params"],
        best_string_params=summary["best_string_params"],
        duration_seconds=summary["total_duration_seconds"],
    )


# =============================================================================
# RQ Entry Point
# =============================================================================


def process_regression_optimize_job(config_json: str) -> dict[str, JSONValue]:
    """RQ job entry point for regression hyperparameter optimization.

    Reads settings from environment, calls run_regression_optimization(),
    and serializes the result to a JSON-compatible dict.

    Args:
        config_json: JSON config with dataset and optimization parameters.

    Returns:
        JSON-serializable regression optimization result dict.
    """
    from covenant_radar_api.core.config import settings_from_env

    settings = settings_from_env()

    data_root = Path(settings["app"]["data_root"])
    external_dir = data_root / "external"
    output_dir = Path(settings["app"]["models_root"]) / "optuna"

    result = run_regression_optimization(config_json, external_dir, output_dir)

    return encode_unified_regression_optimization_result(result)


__all__ = [
    "process_regression_optimize_job",
    "run_regression_optimization",
]
