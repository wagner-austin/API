"""Core optimization runner with backend routing and history tracking.

Routes optimization requests to backend-specific runners based on backend selection.
Supports XGBoost, MLP, LightGBM, and LSTM backends.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from covenant_ml.types import BackendName
from platform_core.json_utils import dump_json_str

import scripts._test_hooks as _hooks
from scripts._test_hooks import (
    ClearGBMOptimizationResult,
    LightGBMOptimizationResult,
    LSTMOptimizationResult,
    MLPOptimizationResult,
    XGBoostOptimizationResult,
)
from scripts.optimize.cli import DatasetName, FeaturePreset
from scripts.optimize.history import UnifiedHistoryEntry

# Union type for all optimization results
UnifiedOptimizationResult = (
    XGBoostOptimizationResult
    | MLPOptimizationResult
    | LightGBMOptimizationResult
    | LSTMOptimizationResult
    | ClearGBMOptimizationResult
)


class RunResult(TypedDict):
    """Result of an optimization run with history context.

    Bundles the optimization result with timing and historical comparison
    data for display purposes.
    """

    backend: BackendName
    result: UnifiedOptimizationResult
    elapsed: float
    previous_best: UnifiedHistoryEntry | None
    all_time_best: UnifiedHistoryEntry | None
    is_new_best: bool


def get_project_root() -> Path:
    """Get project root directory (covenant-radar-api service root).

    Returns:
        Path: The absolute path to the service root directory.
    """
    return Path(__file__).parent.parent.parent


def _build_xgboost_config(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
) -> str:
    """Build XGBoost optimization config JSON.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds, or None for no limit.

    Returns:
        str: JSON configuration string for the optimizer.
    """
    config_dict: dict[str, str | int | None] = {
        "dataset": dataset,
        "n_trials": n_trials,
        "feature_preset": feature_preset,
        "device": device,
        "random_state": 42,
    }
    if timeout is not None:
        config_dict["timeout_seconds"] = timeout
    return dump_json_str(config_dict)


def _build_mlp_config(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
) -> str:
    """Build MLP optimization config JSON.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds, or None for no limit.

    Returns:
        str: JSON configuration string for the optimizer.
    """
    config_dict: dict[str, str | int | None] = {
        "dataset": dataset,
        "n_trials": n_trials,
        "feature_preset": feature_preset,
        "device": device,
        "random_state": 42,
        "precision": "fp32",
        "optimizer": "adamw",
        "n_epochs": 50,
        "early_stopping_patience": 10,
    }
    if timeout is not None:
        config_dict["timeout_seconds"] = timeout
    return dump_json_str(config_dict)


def _build_lightgbm_config(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
) -> str:
    """Build LightGBM optimization config JSON.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds, or None for no limit.

    Returns:
        str: JSON configuration string for the optimizer.
    """
    config_dict: dict[str, str | int | None] = {
        "dataset": dataset,
        "n_trials": n_trials,
        "feature_preset": feature_preset,
        "device": device,
        "random_state": 42,
        "early_stopping_rounds": 10,
    }
    if timeout is not None:
        config_dict["timeout_seconds"] = timeout
    return dump_json_str(config_dict)


def _build_lstm_config(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
) -> str:
    """Build LSTM optimization config JSON.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds, or None for no limit.

    Returns:
        str: JSON configuration string for the optimizer.
    """
    config_dict: dict[str, str | int | bool | None] = {
        "dataset": dataset,
        "n_trials": n_trials,
        "feature_preset": feature_preset,
        "device": device,
        "random_state": 42,
        "precision": "fp32",
        "n_epochs": 50,
        "early_stopping_patience": 10,
        "sequence_length": 5,
        "bidirectional": False,
    }
    if timeout is not None:
        config_dict["timeout_seconds"] = timeout
    return dump_json_str(config_dict)


def run_xgboost(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
    phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
    loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
) -> XGBoostOptimizationResult:
    """Run XGBoost hyperparameter optimization.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds, or None for no limit.
        progress_callback (XGBoostProgressCallbackProtocol | None): Optional
            callback invoked after each trial with progress info.
        phase_callback (XGBoostPhaseCallbackProtocol | None): Optional callback
            invoked when entering a new optimization phase.
        loading_progress_callback (XGBoostLoadingProgressCallbackProtocol | None):
            Optional callback for granular loading progress updates.

    Returns:
        XGBoostOptimizationResult: Optimization result with best hyperparameters
            and validation AUC.
    """
    project_root = get_project_root()
    external_dir = project_root / "data" / "external"
    output_dir = project_root / "models" / "xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_json = _build_xgboost_config(dataset, n_trials, feature_preset, device, timeout)
    return _hooks.xgboost_runner(
        config_json,
        external_dir,
        output_dir,
        progress_callback,
        phase_callback,
        loading_progress_callback,
    )


def run_mlp(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    progress_callback: _hooks.MLPTrialProgressCallbackProtocol | None = None,
    phase_callback: _hooks.MLPPhaseCallbackProtocol | None = None,
    loading_progress_callback: _hooks.MLPLoadingProgressCallbackProtocol | None = None,
) -> MLPOptimizationResult:
    """Run MLP hyperparameter optimization.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds, or None for no limit.
        progress_callback (MLPTrialProgressCallbackProtocol | None): Optional
            callback invoked after each trial with progress info.
        phase_callback (MLPPhaseCallbackProtocol | None): Optional callback
            invoked when entering a new optimization phase.
        loading_progress_callback (MLPLoadingProgressCallbackProtocol | None):
            Optional callback for granular loading progress updates.

    Returns:
        MLPOptimizationResult: Optimization result with best hyperparameters
            and validation AUC.
    """
    project_root = get_project_root()
    external_dir = project_root / "data" / "external"
    output_dir = project_root / "models" / "mlp"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_json = _build_mlp_config(dataset, n_trials, feature_preset, device, timeout)
    return _hooks.mlp_runner(
        config_json,
        external_dir,
        output_dir,
        progress_callback,
        phase_callback,
        loading_progress_callback,
    )


def run_lightgbm(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    progress_callback: _hooks.LightGBMTrialProgressCallbackProtocol | None = None,
    phase_callback: _hooks.LightGBMPhaseCallbackProtocol | None = None,
    loading_progress_callback: _hooks.LightGBMLoadingProgressCallbackProtocol | None = None,
) -> LightGBMOptimizationResult:
    """Run LightGBM hyperparameter optimization.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds, or None for no limit.
        progress_callback (LightGBMTrialProgressCallbackProtocol | None): Optional
            callback invoked after each trial with progress info.
        phase_callback (LightGBMPhaseCallbackProtocol | None): Optional callback
            invoked when entering a new optimization phase.
        loading_progress_callback (LightGBMLoadingProgressCallbackProtocol | None):
            Optional callback for granular loading progress updates.

    Returns:
        LightGBMOptimizationResult: Optimization result with best hyperparameters
            and validation AUC.
    """
    project_root = get_project_root()
    external_dir = project_root / "data" / "external"
    output_dir = project_root / "models" / "lightgbm"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_json = _build_lightgbm_config(dataset, n_trials, feature_preset, device, timeout)
    return _hooks.lightgbm_runner(
        config_json,
        external_dir,
        output_dir,
        progress_callback,
        phase_callback,
        loading_progress_callback,
    )


def run_lstm(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    progress_callback: _hooks.LSTMTrialProgressCallbackProtocol | None = None,
    phase_callback: _hooks.LSTMPhaseCallbackProtocol | None = None,
    loading_progress_callback: _hooks.LSTMLoadingProgressCallbackProtocol | None = None,
) -> LSTMOptimizationResult:
    """Run LSTM hyperparameter optimization.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds, or None for no limit.
        progress_callback (LSTMTrialProgressCallbackProtocol | None): Optional
            callback invoked after each trial with progress info.
        phase_callback (LSTMPhaseCallbackProtocol | None): Optional callback
            invoked when entering a new optimization phase.
        loading_progress_callback (LSTMLoadingProgressCallbackProtocol | None):
            Optional callback for granular loading progress updates.

    Returns:
        LSTMOptimizationResult: Optimization result with best hyperparameters
            and validation AUC.
    """
    project_root = get_project_root()
    external_dir = project_root / "data" / "external"
    output_dir = project_root / "models" / "lstm"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_json = _build_lstm_config(dataset, n_trials, feature_preset, device, timeout)
    return _hooks.lstm_runner(
        config_json,
        external_dir,
        output_dir,
        progress_callback,
        phase_callback,
        loading_progress_callback,
    )


def _build_cleargbm_config(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    timeout: int | None,
) -> str:
    """Build ClearGBM optimization config JSON.

    Note: ClearGBM is pure Python and does not use device setting.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        timeout (int | None): Optional timeout in seconds, or None for no limit.

    Returns:
        str: JSON configuration string for the optimizer.
    """
    config_dict: dict[str, str | int | None] = {
        "dataset": dataset,
        "n_trials": n_trials,
        "feature_preset": feature_preset,
        "random_state": 42,
        "early_stopping_rounds": 10,
    }
    if timeout is not None:
        config_dict["timeout_seconds"] = timeout
    return dump_json_str(config_dict)


def run_cleargbm(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    progress_callback: _hooks.ClearGBMTrialProgressCallbackProtocol | None = None,
    phase_callback: _hooks.ClearGBMPhaseCallbackProtocol | None = None,
    loading_progress_callback: _hooks.ClearGBMLoadingProgressCallbackProtocol | None = None,
) -> ClearGBMOptimizationResult:
    """Run ClearGBM hyperparameter optimization.

    Note: ClearGBM is pure Python and does not use device setting.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Ignored - ClearGBM is pure Python (CPU only).
        timeout (int | None): Optional timeout in seconds, or None for no limit.
        progress_callback (ClearGBMTrialProgressCallbackProtocol | None): Optional
            callback invoked after each trial with progress info.
        phase_callback (ClearGBMPhaseCallbackProtocol | None): Optional callback
            invoked when entering a new optimization phase.
        loading_progress_callback (ClearGBMLoadingProgressCallbackProtocol | None):
            Optional callback for granular loading progress updates.

    Returns:
        ClearGBMOptimizationResult: Optimization result with best hyperparameters
            and validation AUC.
    """
    _ = device  # ClearGBM is pure Python, no device selection
    project_root = get_project_root()
    external_dir = project_root / "data" / "external"
    output_dir = project_root / "models" / "cleargbm"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_json = _build_cleargbm_config(dataset, n_trials, feature_preset, timeout)
    return _hooks.cleargbm_runner(
        config_json,
        external_dir,
        output_dir,
        progress_callback,
        phase_callback,
        loading_progress_callback,
    )


__all__ = [
    "ClearGBMOptimizationResult",
    "LSTMOptimizationResult",
    "LightGBMOptimizationResult",
    "MLPOptimizationResult",
    "RunResult",
    "UnifiedOptimizationResult",
    "XGBoostOptimizationResult",
    "get_project_root",
    "run_cleargbm",
    "run_lightgbm",
    "run_lstm",
    "run_mlp",
    "run_xgboost",
]
