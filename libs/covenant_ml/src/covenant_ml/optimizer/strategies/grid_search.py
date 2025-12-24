"""Grid search optimizer strategy.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Implements exhaustive grid search for hyperparameter optimization.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Protocol

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
# Build Grid Hook (for dependency injection)
# =============================================================================

GridTuple = tuple[SampledIntParams, SampledFloatParams, SampledStringParams]


class BuildGridProtocol(Protocol):
    """Protocol for grid building function."""

    def __call__(
        self,
        search_space: SearchSpace,
        n_points: int,
    ) -> list[GridTuple]: ...


_build_grid_hook: BuildGridProtocol | None = None


def set_build_grid_hook(hook: BuildGridProtocol | None) -> None:
    """Set custom grid builder for testing.

    Args:
        hook: Custom grid builder or None to use default.
    """
    global _build_grid_hook
    _build_grid_hook = hook


def _get_build_grid_hook() -> BuildGridProtocol:
    """Get current grid builder.

    Returns:
        Custom hook if set, otherwise default _build_grid.
    """
    if _build_grid_hook is not None:
        return _build_grid_hook
    return _build_grid


# =============================================================================
# Grid Point Generation
# =============================================================================


def _generate_int_grid(
    spec: IntRangeSpec | CategoricalIntSpec,
    n_points: int = 5,
) -> tuple[int, ...]:
    """Generate grid points for integer parameter."""
    if spec["param_type"] == "categorical_int":
        cat_spec: CategoricalIntSpec = spec
        return cat_spec["choices"]

    int_spec: IntRangeSpec = spec
    if int_spec["log_scale"]:
        log_low: float = math.log(int_spec["low"])
        log_high: float = math.log(int_spec["high"])
        n_grid_points = min(n_points, int_spec["high"] - int_spec["low"] + 1)
        log_values: list[int] = []
        for i in range(n_grid_points):
            frac = i / max(1, n_grid_points - 1)
            log_val = log_low + frac * (log_high - log_low)
            log_values.append(round(math.exp(log_val)))
        values: list[int] = sorted(set(log_values))
        return tuple(values)

    step = max(1, (int_spec["high"] - int_spec["low"]) // (n_points - 1))
    int_values: list[int] = list(range(int_spec["low"], int_spec["high"] + 1, step))
    if int_spec["high"] not in int_values:
        int_values.append(int_spec["high"])
    return tuple(int_values)


def _generate_float_grid(
    spec: FloatRangeSpec | CategoricalFloatSpec,
    n_points: int = 5,
) -> tuple[float, ...]:
    """Generate grid points for float parameter."""
    if spec["param_type"] == "categorical_float":
        cat_spec: CategoricalFloatSpec = spec
        return cat_spec["choices"]

    float_spec: FloatRangeSpec = spec
    if float_spec["log_scale"]:
        log_low: float = math.log(float_spec["low"])
        log_high: float = math.log(float_spec["high"])
        log_values: list[float] = []
        for i in range(n_points):
            frac = i / max(1, n_points - 1)
            log_val = log_low + frac * (log_high - log_low)
            log_values.append(math.exp(log_val))
        return tuple(log_values)

    # Linear spacing
    linear_values: list[float] = []
    for i in range(n_points):
        frac = i / max(1, n_points - 1)
        val = float_spec["low"] + frac * (float_spec["high"] - float_spec["low"])
        linear_values.append(val)
    return tuple(linear_values)


def _generate_string_grid(
    spec: CategoricalStringSpec,
) -> tuple[str, ...]:
    """Generate grid points for string parameter."""
    return spec["choices"]


def _build_xgboost_grid(
    space: XGBoostSearchSpace,
    n_points: int,
) -> list[tuple[SampledIntParams, SampledFloatParams, SampledStringParams]]:
    """Build parameter grid for XGBoost."""
    max_depth_vals = _generate_int_grid(space["max_depth"], n_points)
    n_estimators_vals = _generate_int_grid(space["n_estimators"], n_points)
    learning_rate_vals = _generate_float_grid(space["learning_rate"], n_points)
    reg_alpha_vals = _generate_float_grid(space["reg_alpha"], n_points)
    reg_lambda_vals = _generate_float_grid(space["reg_lambda"], n_points)
    subsample_vals = _generate_float_grid(space["subsample"], n_points)
    colsample_vals = _generate_float_grid(space["colsample_bytree"], n_points)

    # Build base grid
    base_combinations = list(
        itertools.product(
            max_depth_vals,
            n_estimators_vals,
            learning_rate_vals,
            reg_alpha_vals,
            reg_lambda_vals,
            subsample_vals,
            colsample_vals,
        )
    )

    grid: list[tuple[SampledIntParams, SampledFloatParams, SampledStringParams]] = []

    for combo in base_combinations:
        int_params: SampledIntParams = {
            "max_depth": combo[0],
            "n_estimators": combo[1],
        }
        float_params: SampledFloatParams = {
            "learning_rate": combo[2],
            "reg_alpha": combo[3],
            "reg_lambda": combo[4],
            "subsample": combo[5],
            "colsample_bytree": combo[6],
        }

        # Handle booster if present
        if "booster" in space:
            for booster in _generate_string_grid(space["booster"]):
                new_float: SampledFloatParams = {
                    "learning_rate": float_params["learning_rate"],
                    "reg_alpha": float_params["reg_alpha"],
                    "reg_lambda": float_params["reg_lambda"],
                    "subsample": float_params["subsample"],
                    "colsample_bytree": float_params["colsample_bytree"],
                }
                new_string: SampledStringParams = {"booster": booster}

                if booster == "dart":
                    if "rate_drop" in space:
                        new_float["rate_drop"] = _generate_float_grid(space["rate_drop"], 3)[0]
                    if "skip_drop" in space:
                        new_float["skip_drop"] = _generate_float_grid(space["skip_drop"], 3)[0]

                grid.append((int_params, new_float, new_string))
        else:
            grid.append((int_params, float_params, {}))

    return grid


def _build_mlp_grid(
    space: MLPSearchSpace,
    n_points: int,
) -> list[tuple[SampledIntParams, SampledFloatParams, SampledStringParams]]:
    """Build parameter grid for MLP."""
    n_layers_vals = _generate_int_grid(space["n_layers"], n_points)
    hidden_size_vals = _generate_int_grid(space["hidden_size"], n_points)
    batch_size_vals = _generate_int_grid(space["batch_size"], n_points)
    learning_rate_vals = _generate_float_grid(space["learning_rate"], n_points)
    dropout_vals = _generate_float_grid(space["dropout"], n_points)

    grid: list[tuple[SampledIntParams, SampledFloatParams, SampledStringParams]] = []

    for combo in itertools.product(
        n_layers_vals,
        hidden_size_vals,
        batch_size_vals,
        learning_rate_vals,
        dropout_vals,
    ):
        int_params: SampledIntParams = {
            "n_layers": combo[0],
            "hidden_size": combo[1],
            "batch_size": combo[2],
        }
        float_params: SampledFloatParams = {
            "learning_rate": combo[3],
            "dropout": combo[4],
        }
        grid.append((int_params, float_params, {}))

    return grid


def _build_lstm_grid(
    space: LSTMSearchSpace,
    n_points: int,
) -> list[tuple[SampledIntParams, SampledFloatParams, SampledStringParams]]:
    """Build parameter grid for LSTM."""
    hidden_size_vals = _generate_int_grid(space["hidden_size"], n_points)
    num_layers_vals = _generate_int_grid(space["num_layers"], n_points)
    batch_size_vals = _generate_int_grid(space["batch_size"], n_points)
    learning_rate_vals = _generate_float_grid(space["learning_rate"], n_points)
    dropout_vals = _generate_float_grid(space["dropout"], n_points)

    grid: list[tuple[SampledIntParams, SampledFloatParams, SampledStringParams]] = []

    for combo in itertools.product(
        hidden_size_vals,
        num_layers_vals,
        batch_size_vals,
        learning_rate_vals,
        dropout_vals,
    ):
        int_params: SampledIntParams = {
            "hidden_size": combo[0],
            "num_layers": combo[1],
            "batch_size": combo[2],
        }
        float_params: SampledFloatParams = {
            "learning_rate": combo[3],
            "dropout": combo[4],
        }
        grid.append((int_params, float_params, {}))

    return grid


def _add_lightgbm_dart_params(
    space: LightGBMSearchSpace,
    float_params: SampledFloatParams,
) -> None:
    """Add DART-specific parameters to float_params dict in place."""
    if "drop_rate" in space:
        # Take first value for grid simplification
        float_params["drop_rate"] = _generate_float_grid(space["drop_rate"], 3)[0]
    if "skip_drop" in space:
        float_params["skip_drop"] = _generate_float_grid(space["skip_drop"], 3)[0]
    if "feature_fraction" in space:
        float_params["feature_fraction"] = _generate_float_grid(space["feature_fraction"], 3)[0]


def _build_lightgbm_grid(
    space: LightGBMSearchSpace,
    n_points: int,
) -> list[tuple[SampledIntParams, SampledFloatParams, SampledStringParams]]:
    """Build parameter grid for LightGBM."""
    n_estimators_vals = _generate_int_grid(space["n_estimators"], n_points)
    num_leaves_vals = _generate_int_grid(space["num_leaves"], n_points)
    learning_rate_vals = _generate_float_grid(space["learning_rate"], n_points)
    subsample_vals = _generate_float_grid(space["subsample"], n_points)
    colsample_vals = _generate_float_grid(space["colsample_bytree"], n_points)
    reg_alpha_vals = _generate_float_grid(space["reg_alpha"], n_points)
    reg_lambda_vals = _generate_float_grid(space["reg_lambda"], n_points)

    base_combinations = list(
        itertools.product(
            n_estimators_vals,
            num_leaves_vals,
            learning_rate_vals,
            subsample_vals,
            colsample_vals,
            reg_alpha_vals,
            reg_lambda_vals,
        )
    )

    grid: list[tuple[SampledIntParams, SampledFloatParams, SampledStringParams]] = []

    for combo in base_combinations:
        int_params: SampledIntParams = {
            "n_estimators": combo[0],
            "num_leaves": combo[1],
        }
        float_params: SampledFloatParams = {
            "learning_rate": combo[2],
            "subsample": combo[3],
            "colsample_bytree": combo[4],
            "reg_alpha": combo[5],
            "reg_lambda": combo[6],
        }

        if "boosting_type" in space:
            for bt in _generate_string_grid(space["boosting_type"]):
                new_float: SampledFloatParams = {
                    "learning_rate": float_params["learning_rate"],
                    "subsample": float_params["subsample"],
                    "colsample_bytree": float_params["colsample_bytree"],
                    "reg_alpha": float_params["reg_alpha"],
                    "reg_lambda": float_params["reg_lambda"],
                }
                new_string: SampledStringParams = {"boosting_type": bt}

                if bt == "dart":
                    _add_lightgbm_dart_params(space, new_float)

                grid.append((int_params, new_float, new_string))
        else:
            grid.append((int_params, float_params, {}))

    return grid


def _build_grid(
    search_space: SearchSpace,
    n_points: int,
) -> list[tuple[SampledIntParams, SampledFloatParams, SampledStringParams]]:
    """Build parameter grid based on search space type."""
    if is_xgboost_search_space(search_space):
        return _build_xgboost_grid(search_space, n_points)
    if is_mlp_search_space(search_space):
        return _build_mlp_grid(search_space, n_points)
    if is_lstm_search_space(search_space):
        return _build_lstm_grid(search_space, n_points)
    # LightGBM is the remaining type after other guards
    assert is_lightgbm_search_space(search_space)
    return _build_lightgbm_grid(search_space, n_points)


# =============================================================================
# Grid Search Optimizer
# =============================================================================


class GridSearchOptimizer:
    """Exhaustive grid search hyperparameter optimizer.

    Evaluates all combinations of hyperparameters from a discretized grid.
    Best for small search spaces or when you need exhaustive coverage.

    Warning: Grid size grows exponentially with number of parameters.
    For large spaces, consider random search or Bayesian optimization.

    Attributes:
        grid_points: Number of points per continuous dimension (default 3).
    """

    def __init__(self, grid_points: int = 3) -> None:
        """Initialize optimizer.

        Args:
            grid_points: Number of points to sample per continuous dimension.
                Lower values = faster but coarser search.
        """
        self._grid_points = grid_points

    @property
    def grid_points(self) -> int:
        """Get number of grid points per dimension."""
        return self._grid_points

    def strategy_name(self) -> OptimizerStrategyName:
        """Return the strategy name.

        Returns:
            The literal string 'grid_search'.
        """
        return "grid_search"

    def capabilities(self) -> OptimizerStrategyCapabilities:
        """Return the capabilities of this strategy.

        Returns:
            Capabilities indicating grid search is deterministic and parallel.
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
        """Run grid search hyperparameter optimization.

        Args:
            x_features: Feature matrix (n_samples, n_features).
            y_labels: Binary labels (n_samples,).
            feature_names: Names for each feature column.
            search_space: Parameter ranges to search.
            config: Optimization settings (n_trials limits grid evaluation).
            objective: Function to evaluate hyperparameters.
            trial_callback: Optional callback after each trial.

        Returns:
            Summary with best hyperparameters and trial statistics.
        """
        start_time = time.perf_counter()

        # Build the grid
        grid = _get_build_grid_hook()(search_space, self._grid_points)
        n_grid = len(grid)

        # Limit to n_trials if specified
        max_trials = min(config["n_trials"], n_grid)

        _log.info(
            "Starting grid search optimization",
            extra={
                "grid_size": n_grid,
                "max_trials": max_trials,
                "grid_points": self._grid_points,
            },
        )

        best_value = float("-inf") if config["direction"] == "maximize" else float("inf")
        best_trial_number = 0
        best_int_params: SampledIntParams = {}
        best_float_params: SampledFloatParams = {}
        best_string_params: SampledStringParams = {}

        trials_complete = 0

        for trial_num, (int_params, float_params, string_params) in enumerate(grid):
            if trial_num >= max_trials:
                break

            # Check timeout
            if config["timeout_seconds"] is not None:
                elapsed = time.perf_counter() - start_time
                if elapsed > config["timeout_seconds"]:
                    _log.info(
                        "Grid search stopped due to timeout",
                        extra={"elapsed": elapsed, "trials_complete": trials_complete},
                    )
                    break

            trial_start = time.perf_counter()

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
                "Grid search trial complete",
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
            "n_trials_total": max_trials,
            "n_trials_complete": trials_complete,
            "n_trials_pruned": 0,
            "n_trials_failed": 0,
            "total_duration_seconds": total_duration,
        }

        _log.info(
            "Grid search optimization complete",
            extra={
                "best_value": summary["best_value"],
                "n_trials_complete": summary["n_trials_complete"],
                "grid_coverage": trials_complete / n_grid if n_grid > 0 else 0,
                "total_duration_sec": summary["total_duration_seconds"],
            },
        )

        return summary


def create_grid_search_optimizer() -> GridSearchOptimizer:
    """Factory function to create a GridSearchOptimizer.

    Returns:
        A new GridSearchOptimizer instance with default 3 points per dimension.
    """
    return GridSearchOptimizer(grid_points=3)


__all__ = [
    "BuildGridProtocol",
    "GridSearchOptimizer",
    "GridTuple",
    "create_grid_search_optimizer",
    "set_build_grid_hook",
]
