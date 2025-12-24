"""Random search optimizer strategy.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Implements random sampling from search space for hyperparameter optimization.
"""

from __future__ import annotations

import math
import time

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from ..protocol import ObjectiveProtocol, TrialCallbackProtocol
from ..strategy_protocol import OptimizerStrategyCapabilities, OptimizerStrategyName
from ..type_guards import (
    is_lightgbm_search_space,
    is_lstm_search_space,
    is_mlp_search_space,
    is_xgboost_search_space,
)
from ..types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    CategoricalStringSpec,
    FloatRangeSpec,
    IntRangeSpec,
    LightGBMSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    OptimizationConfig,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    SearchSpace,
    TrialResult,
    XGBoostSearchSpace,
)

_log = get_logger(__name__)


# =============================================================================
# Random Sampling Functions
# =============================================================================


def _sample_int_random(
    rng: np.random.Generator,
    spec: IntRangeSpec | CategoricalIntSpec,
) -> int:
    """Sample integer parameter randomly."""
    if spec["param_type"] == "int":
        int_spec: IntRangeSpec = spec
        if int_spec["log_scale"]:
            log_low: float = math.log(int_spec["low"])
            log_high: float = math.log(int_spec["high"])
            log_val: float = float(rng.uniform(log_low, log_high))
            return round(math.exp(log_val))
        return int(rng.integers(int_spec["low"], int_spec["high"] + 1))
    cat_spec: CategoricalIntSpec = spec
    idx: int = int(rng.integers(0, len(cat_spec["choices"])))
    return cat_spec["choices"][idx]


def _sample_float_random(
    rng: np.random.Generator,
    spec: FloatRangeSpec | CategoricalFloatSpec,
) -> float:
    """Sample float parameter randomly."""
    if spec["param_type"] == "float":
        float_spec: FloatRangeSpec = spec
        if float_spec["log_scale"]:
            log_low: float = math.log(float_spec["low"])
            log_high: float = math.log(float_spec["high"])
            log_val: float = float(rng.uniform(log_low, log_high))
            return math.exp(log_val)
        return float(rng.uniform(float_spec["low"], float_spec["high"]))
    cat_spec: CategoricalFloatSpec = spec
    idx: int = int(rng.integers(0, len(cat_spec["choices"])))
    return cat_spec["choices"][idx]


def _sample_string_random(
    rng: np.random.Generator,
    spec: CategoricalStringSpec,
) -> str:
    """Sample string parameter randomly."""
    idx = int(rng.integers(0, len(spec["choices"])))
    return spec["choices"][idx]


def _sample_xgboost_random(
    rng: np.random.Generator,
    space: XGBoostSearchSpace,
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Sample XGBoost hyperparameters randomly."""
    int_params: SampledIntParams = {
        "max_depth": _sample_int_random(rng, space["max_depth"]),
        "n_estimators": _sample_int_random(rng, space["n_estimators"]),
    }

    float_params: SampledFloatParams = {
        "learning_rate": _sample_float_random(rng, space["learning_rate"]),
        "reg_alpha": _sample_float_random(rng, space["reg_alpha"]),
        "reg_lambda": _sample_float_random(rng, space["reg_lambda"]),
        "subsample": _sample_float_random(rng, space["subsample"]),
        "colsample_bytree": _sample_float_random(rng, space["colsample_bytree"]),
    }

    string_params: SampledStringParams = {}

    if "booster" in space:
        booster = _sample_string_random(rng, space["booster"])
        string_params["booster"] = booster
        if booster == "dart":
            if "rate_drop" in space:
                float_params["rate_drop"] = _sample_float_random(rng, space["rate_drop"])
            if "skip_drop" in space:
                float_params["skip_drop"] = _sample_float_random(rng, space["skip_drop"])

    return int_params, float_params, string_params


def _sample_mlp_random(
    rng: np.random.Generator,
    space: MLPSearchSpace,
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Sample MLP hyperparameters randomly."""
    int_params: SampledIntParams = {
        "n_layers": _sample_int_random(rng, space["n_layers"]),
        "hidden_size": _sample_int_random(rng, space["hidden_size"]),
        "batch_size": _sample_int_random(rng, space["batch_size"]),
    }

    float_params: SampledFloatParams = {
        "learning_rate": _sample_float_random(rng, space["learning_rate"]),
        "dropout": _sample_float_random(rng, space["dropout"]),
    }

    return int_params, float_params, {}


def _sample_lstm_random(
    rng: np.random.Generator,
    space: LSTMSearchSpace,
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Sample LSTM hyperparameters randomly."""
    int_params: SampledIntParams = {
        "hidden_size": _sample_int_random(rng, space["hidden_size"]),
        "num_layers": _sample_int_random(rng, space["num_layers"]),
        "batch_size": _sample_int_random(rng, space["batch_size"]),
    }

    float_params: SampledFloatParams = {
        "learning_rate": _sample_float_random(rng, space["learning_rate"]),
        "dropout": _sample_float_random(rng, space["dropout"]),
    }

    return int_params, float_params, {}


def _sample_lightgbm_random(
    rng: np.random.Generator,
    space: LightGBMSearchSpace,
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Sample LightGBM hyperparameters randomly."""
    int_params: SampledIntParams = {
        "n_estimators": _sample_int_random(rng, space["n_estimators"]),
        "num_leaves": _sample_int_random(rng, space["num_leaves"]),
    }

    float_params: SampledFloatParams = {
        "learning_rate": _sample_float_random(rng, space["learning_rate"]),
        "subsample": _sample_float_random(rng, space["subsample"]),
        "colsample_bytree": _sample_float_random(rng, space["colsample_bytree"]),
        "reg_alpha": _sample_float_random(rng, space["reg_alpha"]),
        "reg_lambda": _sample_float_random(rng, space["reg_lambda"]),
    }

    string_params: SampledStringParams = {}

    if "boosting_type" in space:
        boosting_type = _sample_string_random(rng, space["boosting_type"])
        string_params["boosting_type"] = boosting_type
        if boosting_type == "dart":
            if "drop_rate" in space:
                float_params["drop_rate"] = _sample_float_random(rng, space["drop_rate"])
            if "skip_drop" in space:
                float_params["skip_drop"] = _sample_float_random(rng, space["skip_drop"])
            if "feature_fraction" in space:
                float_params["feature_fraction"] = _sample_float_random(
                    rng, space["feature_fraction"]
                )

    return int_params, float_params, string_params


def _sample_random(
    rng: np.random.Generator,
    search_space: SearchSpace,
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Sample parameters randomly based on search space type."""
    if is_xgboost_search_space(search_space):
        return _sample_xgboost_random(rng, search_space)
    if is_mlp_search_space(search_space):
        return _sample_mlp_random(rng, search_space)
    if is_lstm_search_space(search_space):
        return _sample_lstm_random(rng, search_space)
    # LightGBM is the remaining type after other guards
    assert is_lightgbm_search_space(search_space)
    return _sample_lightgbm_random(rng, search_space)


# =============================================================================
# Random Search Optimizer
# =============================================================================


class RandomSearchOptimizer:
    """Random search hyperparameter optimizer.

    Randomly samples hyperparameters from the search space. Simple but
    effective baseline that often performs surprisingly well compared
    to more sophisticated methods.

    Benefits:
    - Embarrassingly parallel
    - No sequential dependencies
    - Covers the search space uniformly
    - Reproducible with fixed seed
    """

    def __init__(self) -> None:
        """Initialize optimizer."""
        pass

    def strategy_name(self) -> OptimizerStrategyName:
        """Return the strategy name.

        Returns:
            The literal string 'random_search'.
        """
        return "random_search"

    def capabilities(self) -> OptimizerStrategyCapabilities:
        """Return the capabilities of this strategy.

        Returns:
            Capabilities indicating random search is deterministic and parallel.
        """
        return OptimizerStrategyCapabilities(
            supports_pruning=False,
            supports_parallel=True,
            is_deterministic=True,
            requires_bounds=True,
        )

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: SearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run random search hyperparameter optimization.

        Args:
            x_features: Feature matrix (n_samples, n_features).
            y_labels: Binary labels (n_samples,).
            feature_names: Names for each feature column.
            search_space: Parameter ranges to search.
            config: Optimization settings.
            objective: Function to evaluate hyperparameters.
            trial_callback: Optional callback after each trial.

        Returns:
            Summary with best hyperparameters and trial statistics.
        """
        rng = np.random.default_rng(config["random_state"])
        start_time = time.perf_counter()

        _log.info(
            "Starting random search optimization",
            extra={
                "n_trials": config["n_trials"],
                "random_state": config["random_state"],
            },
        )

        best_value = float("-inf") if config["direction"] == "maximize" else float("inf")
        best_trial_number = 0
        best_int_params: SampledIntParams = {}
        best_float_params: SampledFloatParams = {}
        best_string_params: SampledStringParams = {}

        trials_complete = 0
        trials_failed = 0

        for trial_num in range(config["n_trials"]):
            # Check timeout
            if config["timeout_seconds"] is not None:
                elapsed = time.perf_counter() - start_time
                if elapsed > config["timeout_seconds"]:
                    _log.info(
                        "Random search stopped due to timeout",
                        extra={"elapsed": elapsed, "trials_complete": trials_complete},
                    )
                    break

            trial_start = time.perf_counter()

            int_params, float_params, string_params = _sample_random(rng, search_space)

            val_auc = objective(
                x_features,
                y_labels,
                feature_names,
                int_params,
                float_params,
                string_params,
                config["train_ratio"],
                config["val_ratio"],
                config["test_ratio"],
                config["random_state"],
            )

            trial_duration = time.perf_counter() - trial_start
            trials_complete += 1

            # Check if this is the best trial
            is_better = (config["direction"] == "maximize" and val_auc > best_value) or (
                config["direction"] == "minimize" and val_auc < best_value
            )

            if is_better:
                best_value = val_auc
                best_trial_number = trial_num
                best_int_params = int_params
                best_float_params = float_params
                best_string_params = string_params

            result: TrialResult = {
                "trial_number": trial_num,
                "int_params": int_params,
                "float_params": float_params,
                "string_params": string_params,
                "value": val_auc,
                "state": "complete",
                "duration_seconds": trial_duration,
            }

            if trial_callback is not None:
                trial_callback(result)

            _log.debug(
                "Random search trial complete",
                extra={
                    "trial": trial_num,
                    "val_auc": val_auc,
                    "is_best": is_better,
                    "duration_sec": trial_duration,
                },
            )

        total_duration = time.perf_counter() - start_time

        summary: OptimizationSummary = {
            "best_trial_number": best_trial_number,
            "best_value": best_value,
            "best_int_params": best_int_params,
            "best_float_params": best_float_params,
            "best_string_params": best_string_params,
            "n_trials_total": config["n_trials"],
            "n_trials_complete": trials_complete,
            "n_trials_pruned": 0,
            "n_trials_failed": trials_failed,
            "total_duration_seconds": total_duration,
        }

        _log.info(
            "Random search optimization complete",
            extra={
                "best_value": summary["best_value"],
                "n_trials_complete": summary["n_trials_complete"],
                "total_duration_sec": summary["total_duration_seconds"],
            },
        )

        return summary


def create_random_search_optimizer() -> RandomSearchOptimizer:
    """Factory function to create a RandomSearchOptimizer.

    Returns:
        A new RandomSearchOptimizer instance.
    """
    return RandomSearchOptimizer()


__all__ = [
    "RandomSearchOptimizer",
    "create_random_search_optimizer",
]
