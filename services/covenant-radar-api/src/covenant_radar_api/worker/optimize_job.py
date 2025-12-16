"""Background job for hyperparameter optimization using Optuna TPE.

Runs Bayesian optimization on external bankruptcy datasets to find
optimal XGBoost hyperparameters. Results include best hyperparameters
and recommended TrainConfig for subsequent training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, TypedDict

import numpy as np
from covenant_ml.features import (
    FeaturePreset,
    engineer_features,
    get_feature_config_for_preset,
)
from covenant_ml.metrics import compute_auc
from covenant_ml.optimizer import (
    OptimizationConfig,
    OptimizationSummary,
    TrialResult,
    XGBoostSearchSpace,
    create_xgboost_optimizer,
)
from covenant_ml.trainer import stratified_split
from covenant_ml.types import TrainConfig
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    require_int,
    require_str,
)
from platform_core.logging import get_logger

from covenant_radar_api.seeding.real_data import (
    RawDataset,
    load_polish_raw,
    load_taiwan_raw,
    load_us_raw,
)

_log = get_logger(__name__)


class XGBClassifierProtocol(Protocol):
    """Protocol for XGBoost classifier interface."""

    def __init__(
        self,
        *,
        max_depth: int,
        n_estimators: int,
        learning_rate: float,
        reg_alpha: float,
        reg_lambda: float,
        subsample: float,
        colsample_bytree: float,
        random_state: int,
        scale_pos_weight: float,
        objective: str,
        eval_metric: str,
        n_jobs: int,
        tree_method: str,
        device: str,
    ) -> None: ...

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
        *,
        verbose: bool,
    ) -> None: ...

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


def _get_xgb_classifier() -> type[XGBClassifierProtocol]:
    """Get XGBClassifier class via dynamic import for strict typing."""
    xgb_module = __import__("xgboost")
    cls: type[XGBClassifierProtocol] = xgb_module.XGBClassifier
    return cls


class DMatrixProtocol(Protocol):
    """Protocol for XGBoost DMatrix."""

    def __init__(
        self,
        data: NDArray[np.float64],
        label: NDArray[np.int64] | None = ...,
    ) -> None: ...


class BoosterProtocol(Protocol):
    """Protocol for XGBoost Booster."""

    def predict(self, data: DMatrixProtocol) -> NDArray[np.float64]: ...


class XGBTrainFunc(Protocol):
    """Protocol for xgb.train function."""

    def __call__(
        self,
        params: dict[str, str | int | float],
        dtrain: DMatrixProtocol,
        num_boost_round: int = ...,
        *,
        verbose_eval: bool = ...,
    ) -> BoosterProtocol: ...


def _get_xgb_dmatrix_and_train() -> tuple[type[DMatrixProtocol], XGBTrainFunc]:
    """Get DMatrix class and train function via dynamic import."""
    xgb_module = __import__("xgboost")
    dmatrix_cls: type[DMatrixProtocol] = xgb_module.DMatrix
    train_fn: XGBTrainFunc = xgb_module.train
    return dmatrix_cls, train_fn


DatasetName = Literal["taiwan", "us", "polish"]
SpaceProfile = Literal["default", "categorical"]


def _optional_int(data: JSONObject, key: str, default: int) -> int:
    """Extract optional int from dict."""
    raw = data.get(key)
    if raw is None:
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    raise JSONTypeError(f"Field '{key}' must be a number")


def _parse_device(raw: JSONValue | None) -> Literal["cpu", "cuda", "auto"]:
    """Parse device setting, defaulting to 'auto'."""
    if raw is None:
        return "auto"
    if not isinstance(raw, str):
        raise JSONTypeError("device must be a string")
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        return "cuda"
    if raw == "auto":
        return "auto"
    raise ValueError("device must be one of: cpu, cuda, auto")


def _parse_space_profile(raw: JSONValue | None) -> SpaceProfile:
    """Parse space profile, defaulting to 'default'."""
    if raw is None:
        return "default"
    if not isinstance(raw, str):
        raise JSONTypeError("space_profile must be a string")
    if raw == "default":
        return "default"
    if raw == "categorical":
        return "categorical"
    raise JSONTypeError("space_profile must be one of: default, categorical")


def _parse_feature_preset(raw: JSONValue | None) -> FeaturePreset:
    """Parse feature preset, defaulting to 'none'."""
    if raw is None:
        return "none"
    if not isinstance(raw, str):
        raise JSONTypeError("feature_preset must be a string")
    if raw == "none":
        return "none"
    if raw == "log_only":
        return "log_only"
    if raw == "ratios_only":
        return "ratios_only"
    if raw == "full":
        return "full"
    raise JSONTypeError("feature_preset must be one of: none, log_only, ratios_only, full")


class OptimizeParseResult(TypedDict, total=True):
    """Parsed optimization request."""

    dataset: DatasetName
    n_trials: int
    timeout_seconds: int | None
    device: Literal["cpu", "cuda", "auto"]
    space_profile: SpaceProfile
    feature_preset: FeaturePreset
    random_state: int


def _parse_optimize_config(config_json: str) -> OptimizeParseResult:
    """Parse optimization config from JSON string.

    Returns:
        OptimizeParseResult with all optimization parameters.
    """
    raw = load_json_str(config_json)
    if not isinstance(raw, dict):
        raise JSONTypeError("config must be a JSON object")

    # Dataset selection (required)
    dataset = require_str(raw, "dataset")
    dataset_name: DatasetName
    if dataset == "taiwan":
        dataset_name = "taiwan"
    elif dataset == "us":
        dataset_name = "us"
    elif dataset == "polish":
        dataset_name = "polish"
    else:
        raise ValueError(f"dataset must be one of: taiwan, us, polish (got {dataset})")

    n_trials = require_int(raw, "n_trials")

    timeout_raw = raw.get("timeout_seconds")
    timeout_seconds: int | None = None
    if timeout_raw is not None:
        if not isinstance(timeout_raw, int):
            raise JSONTypeError("timeout_seconds must be an integer or null")
        timeout_seconds = timeout_raw

    device = _parse_device(raw.get("device"))
    space_profile = _parse_space_profile(raw.get("space_profile"))
    feature_preset = _parse_feature_preset(raw.get("feature_preset"))
    random_state = _optional_int(raw, "random_state", 42)

    return OptimizeParseResult(
        dataset=dataset_name,
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        device=device,
        space_profile=space_profile,
        feature_preset=feature_preset,
        random_state=random_state,
    )


def _load_dataset(dataset_name: DatasetName, external_dir: Path) -> RawDataset:
    """Load the specified dataset.

    Args:
        dataset_name: Which dataset to load ('taiwan', 'us', or 'polish')
        external_dir: Path to data/external directory

    Returns:
        RawDataset with feature matrix, labels, and column names

    Raises:
        FileNotFoundError: If dataset file doesn't exist
    """
    if dataset_name == "taiwan":
        data_path = external_dir / "taiwan_data" / "data.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Taiwan dataset not found at {data_path}")
        return load_taiwan_raw(data_path)
    if dataset_name == "us":
        data_path = external_dir / "us_data" / "american_bankruptcy.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"US dataset not found at {data_path}")
        return load_us_raw(data_path)
    # dataset_name == "polish"
    data_path = external_dir / "polish_data" / "1year.arff"
    if not data_path.exists():
        raise FileNotFoundError(f"Polish dataset not found at {data_path}")
    return load_polish_raw(data_path)


def _get_search_space(profile: SpaceProfile) -> XGBoostSearchSpace:
    """Get search space based on profile name."""
    from covenant_ml.optimizer import (
        make_xgboost_categorical_space,
        make_xgboost_default_space,
    )

    if profile == "default":
        return make_xgboost_default_space()
    return make_xgboost_categorical_space()


def _build_optimization_config(
    n_trials: int,
    timeout_seconds: int | None,
    random_state: int,
) -> OptimizationConfig:
    """Build optimization config with standard train/val/test splits."""
    from covenant_ml.optimizer import make_default_optimization_config

    return make_default_optimization_config(
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        random_state=random_state,
    )


class _XGBoostObjective:
    """XGBoost objective that trains on pre-split data and returns validation AUC.

    Uses DMatrix directly for full GPU pipeline - no scikit-learn wrapper overhead.
    """

    def __init__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        device: Literal["cpu", "cuda", "auto"],
        feature_preset: FeaturePreset,
    ) -> None:
        """Initialize with pre-split data and pre-created DMatrix objects.

        Args:
            x_features: Feature matrix
            y_labels: Binary labels
            feature_names: Original feature names
            device: Device to use for training
            feature_preset: Feature engineering preset to apply
        """
        # Apply feature engineering BEFORE splitting
        if feature_preset != "none":
            config = get_feature_config_for_preset(feature_preset)
            engineered = engineer_features(x_features, feature_names, config)
            x_engineered = engineered["x"]
            n_original = engineered["n_original"]
            n_ratios = engineered["n_ratios"]
            n_products = engineered["n_products"]
            n_log = engineered["n_log"]
            _log.info(
                "Applied feature engineering",
                extra={
                    "preset": feature_preset,
                    "n_original": n_original,
                    "n_ratios": n_ratios,
                    "n_products": n_products,
                    "n_log": n_log,
                    "total_features": int(x_engineered.shape[1]),
                },
            )
        else:
            x_engineered = x_features

        # Store actual feature count (after engineering)
        self._n_features = int(x_engineered.shape[1])

        # Pre-split data once (stratified)
        self._splits = stratified_split(
            x_engineered,
            y_labels,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )
        # Resolve device once
        self._device = ("cuda" if _cuda_available() else "cpu") if device == "auto" else device

        # Calculate scale_pos_weight from training data (once)
        n_pos = int(np.sum(self._splits.y_train))
        n_neg = len(self._splits.y_train) - n_pos
        self._scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # Pre-create DMatrix objects (device is set in params, not DMatrix)
        dmatrix_cls, train_fn = _get_xgb_dmatrix_and_train()
        self._train_dmatrix = dmatrix_cls(
            self._splits.x_train,
            label=self._splits.y_train,
        )
        self._val_dmatrix = dmatrix_cls(
            self._splits.x_val,
            label=self._splits.y_val,
        )
        self._y_val = self._splits.y_val  # Keep for AUC computation
        self._xgb_train = train_fn

    @property
    def n_features(self) -> int:
        """Return the actual feature count (after engineering)."""
        return self._n_features

    def __call__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        max_depth: int,
        n_estimators: int,
        learning_rate: float,
        reg_alpha: float,
        reg_lambda: float,
        subsample: float,
        colsample_bytree: float,
        random_state: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> float:
        """Train XGBoost using DMatrix directly and return validation AUC."""
        # Ignore passed data - use pre-computed DMatrix
        _ = x_features, y_labels, feature_names
        _ = train_ratio, val_ratio, test_ratio

        # XGBoost parameters for direct training
        params: dict[str, str | int | float] = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": self._scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "device": self._device,
            "seed": random_state,
        }

        # Train using xgb.train directly (full GPU pipeline)
        booster = self._xgb_train(
            params,
            self._train_dmatrix,
            num_boost_round=n_estimators,
            verbose_eval=False,
        )

        # Predict on validation set (already on GPU)
        y_pred_proba: NDArray[np.float64] = booster.predict(self._val_dmatrix)
        # Use our typed compute_auc instead of sklearn
        return compute_auc(self._y_val, y_pred_proba)


def _create_xgboost_objective(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    feature_names: list[str],
    device: Literal["cpu", "cuda", "auto"],
    feature_preset: FeaturePreset,
) -> _XGBoostObjective:
    """Create an objective function for XGBoost optimization.

    Applies feature engineering based on preset and pre-splits data for efficient
    trial evaluation. The returned objective tracks the engineered feature count
    via its n_features property.

    Args:
        x_features: Feature matrix
        y_labels: Binary labels
        feature_names: Original feature names
        device: Device to use for training
        feature_preset: Feature engineering preset to apply

    Returns:
        Objective callable with n_features property for engineered feature count
    """
    return _XGBoostObjective(x_features, y_labels, feature_names, device, feature_preset)


class XGBBuildInfoProtocol(Protocol):
    """Protocol for xgboost module's build_info function."""

    def __call__(self) -> dict[str, str]: ...


def _cuda_available() -> bool:
    """Check if CUDA is available for XGBoost."""
    xgb_module = __import__("xgboost")
    build_info_fn: XGBBuildInfoProtocol = xgb_module.build_info
    build_info: dict[str, str] = build_info_fn()
    use_cuda_value = build_info.get("USE_CUDA", "OFF")
    return use_cuda_value == "ON"


class OptimizationResult(TypedDict, total=True):
    """Result of a hyperparameter optimization run."""

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
    best_max_depth: int
    best_n_estimators: int
    best_learning_rate: float
    best_reg_alpha: float
    best_reg_lambda: float
    best_subsample: float
    best_colsample_bytree: float
    duration_seconds: float
    recommended_config: TrainConfig


def _generate_train_config(
    summary: OptimizationSummary,
    device: Literal["cpu", "cuda", "auto"],
) -> TrainConfig:
    """Generate a TrainConfig from optimization summary."""
    return TrainConfig(
        device=device,
        learning_rate=summary["best_learning_rate"],
        max_depth=summary["best_max_depth"],
        n_estimators=summary["best_n_estimators"],
        subsample=summary["best_subsample"],
        colsample_bytree=summary["best_colsample_bytree"],
        random_state=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        early_stopping_rounds=20,
        reg_alpha=summary["best_reg_alpha"],
        reg_lambda=summary["best_reg_lambda"],
    )


class TrialProgressInfo(TypedDict):
    """Information about trial progress for CLI display."""

    trial_number: int
    n_trials_total: int
    current_auc: float
    best_auc: float
    best_trial: int
    is_best: bool
    best_learning_rate: float
    best_max_depth: int
    best_n_estimators: int


class TrialProgressCallbackProtocol(Protocol):
    """Protocol for trial progress callback."""

    def __call__(self, info: TrialProgressInfo) -> None:
        """Called after each trial with progress info."""
        ...


def run_optimization(
    config_json: str,
    external_dir: Path,
    output_dir: Path,
    progress_callback: TrialProgressCallbackProtocol | None = None,
) -> OptimizationResult:
    """Run hyperparameter optimization on external dataset.

    Args:
        config_json: JSON config with dataset and optimization parameters
        external_dir: Path to data/external directory with datasets
        output_dir: Directory to save optimization results
        progress_callback: Optional callback for trial progress updates

    Returns:
        OptimizationResult with best hyperparameters and recommended config
    """
    parse_result = _parse_optimize_config(config_json)
    dataset_name = parse_result["dataset"]

    # Load raw dataset
    dataset = _load_dataset(dataset_name, external_dir)

    _log.info(
        "Starting hyperparameter optimization",
        extra={
            "dataset": dataset_name,
            "n_samples": dataset["n_samples"],
            "n_features": dataset["n_features"],
            "n_trials": parse_result["n_trials"],
            "space_profile": parse_result["space_profile"],
            "feature_preset": parse_result["feature_preset"],
            "device": parse_result["device"],
        },
    )

    # Build config and search space
    config = _build_optimization_config(
        n_trials=parse_result["n_trials"],
        timeout_seconds=parse_result["timeout_seconds"],
        random_state=parse_result["random_state"],
    )
    search_space = _get_search_space(parse_result["space_profile"])

    # Create objective function (applies feature engineering if preset != "none")
    objective = _create_xgboost_objective(
        dataset["x"],
        dataset["y"],
        dataset["feature_names"],
        parse_result["device"],
        parse_result["feature_preset"],
    )

    # Track progress
    best_auc = 0.0
    best_trial_num = 0
    best_learning_rate = 0.0
    best_max_depth = 0
    best_n_estimators = 0
    trials_seen = 0
    n_trials_total = parse_result["n_trials"]

    def trial_callback(result: TrialResult) -> None:
        nonlocal best_auc
        nonlocal best_trial_num
        nonlocal trials_seen
        nonlocal best_learning_rate
        nonlocal best_max_depth
        nonlocal best_n_estimators
        trials_seen += 1
        auc = result["value"]
        is_best = auc > best_auc
        if is_best:
            best_auc = auc
            best_trial_num = result["trial_number"]
            best_learning_rate = result["params_learning_rate"]
            best_max_depth = result["params_max_depth"]
            best_n_estimators = result["params_n_estimators"]
            _log.info(
                "New best trial",
                extra={
                    "trial": result["trial_number"],
                    "auc": f"{auc:.4f}",
                    "max_depth": result["params_max_depth"],
                    "learning_rate": f"{result['params_learning_rate']:.4f}",
                    "n_estimators": result["params_n_estimators"],
                },
            )

        # Call progress callback if provided
        if progress_callback is not None:
            progress_info: TrialProgressInfo = {
                "trial_number": result["trial_number"],
                "n_trials_total": n_trials_total,
                "current_auc": auc,
                "best_auc": best_auc,
                "best_trial": best_trial_num,
                "is_best": is_best,
                "best_learning_rate": best_learning_rate,
                "best_max_depth": best_max_depth,
                "best_n_estimators": best_n_estimators,
            }
            progress_callback(progress_info)

    # Run optimization
    optimizer = create_xgboost_optimizer()
    summary: OptimizationSummary = optimizer.optimize(
        x_features=dataset["x"],
        y_labels=dataset["y"],
        feature_names=dataset["feature_names"],
        search_space=search_space,
        config=config,
        objective=objective,
        trial_callback=trial_callback,
    )

    _log.info(
        "Optimization complete",
        extra={
            "dataset": dataset_name,
            "best_trial": summary["best_trial_number"],
            "best_auc": f"{summary['best_value']:.4f}",
            "n_trials_complete": summary["n_trials_complete"],
            "duration_seconds": f"{summary['total_duration_seconds']:.1f}",
        },
    )

    # Generate recommended config
    recommended_config = _generate_train_config(summary, parse_result["device"])

    # Save results to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{dataset_name}_optuna_result.json"
    config_path = output_dir / f"{dataset_name}_optimal_config.json"

    # Build serializable result (use objective.n_features for actual engineered count)
    result_dict: dict[str, JSONValue] = {
        "dataset": dataset_name,
        "n_samples": dataset["n_samples"],
        "n_features": objective.n_features,
        "best_trial": summary["best_trial_number"],
        "best_val_auc": summary["best_value"],
        "best_max_depth": summary["best_max_depth"],
        "best_n_estimators": summary["best_n_estimators"],
        "best_learning_rate": summary["best_learning_rate"],
        "best_reg_alpha": summary["best_reg_alpha"],
        "best_reg_lambda": summary["best_reg_lambda"],
        "best_subsample": summary["best_subsample"],
        "best_colsample_bytree": summary["best_colsample_bytree"],
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
        "subsample": recommended_config["subsample"],
        "colsample_bytree": recommended_config["colsample_bytree"],
        "random_state": recommended_config["random_state"],
        "train_ratio": recommended_config["train_ratio"],
        "val_ratio": recommended_config["val_ratio"],
        "test_ratio": recommended_config["test_ratio"],
        "early_stopping_rounds": recommended_config["early_stopping_rounds"],
        "reg_alpha": recommended_config["reg_alpha"],
        "reg_lambda": recommended_config["reg_lambda"],
    }

    with open(config_path, "w") as f:
        f.write(dump_json_str(config_dict))

    _log.info(
        "Saved optimization results",
        extra={
            "result_path": str(result_path),
            "config_path": str(config_path),
        },
    )

    return OptimizationResult(
        status="complete",
        dataset=dataset_name,
        n_samples=dataset["n_samples"],
        n_features=objective.n_features,
        feature_preset=parse_result["feature_preset"],
        n_trials_complete=summary["n_trials_complete"],
        n_trials_pruned=summary["n_trials_pruned"],
        n_trials_failed=summary["n_trials_failed"],
        best_trial_number=summary["best_trial_number"],
        best_val_auc=summary["best_value"],
        best_max_depth=summary["best_max_depth"],
        best_n_estimators=summary["best_n_estimators"],
        best_learning_rate=summary["best_learning_rate"],
        best_reg_alpha=summary["best_reg_alpha"],
        best_reg_lambda=summary["best_reg_lambda"],
        best_subsample=summary["best_subsample"],
        best_colsample_bytree=summary["best_colsample_bytree"],
        duration_seconds=summary["total_duration_seconds"],
        recommended_config=recommended_config,
    )


def process_optimize_job(config_json: str) -> dict[str, JSONValue]:
    """RQ job entry point for hyperparameter optimization.

    Args:
        config_json: JSON config with dataset and optimization parameters

    Returns:
        Optimization result with best hyperparameters
    """
    from covenant_radar_api.core.config import settings_from_env

    settings = settings_from_env()

    # Get directories from settings
    data_root = Path(settings["app"]["data_root"])
    external_dir = data_root / "external"
    output_dir = Path(settings["app"]["models_root"]) / "optuna"

    result = run_optimization(config_json, external_dir, output_dir)

    # Convert to JSON-serializable dict
    config_json_dict: dict[str, JSONValue] = {
        "device": result["recommended_config"]["device"],
        "learning_rate": result["recommended_config"]["learning_rate"],
        "max_depth": result["recommended_config"]["max_depth"],
        "n_estimators": result["recommended_config"]["n_estimators"],
        "subsample": result["recommended_config"]["subsample"],
        "colsample_bytree": result["recommended_config"]["colsample_bytree"],
        "random_state": result["recommended_config"]["random_state"],
        "train_ratio": result["recommended_config"]["train_ratio"],
        "val_ratio": result["recommended_config"]["val_ratio"],
        "test_ratio": result["recommended_config"]["test_ratio"],
        "early_stopping_rounds": result["recommended_config"]["early_stopping_rounds"],
        "reg_alpha": result["recommended_config"]["reg_alpha"],
        "reg_lambda": result["recommended_config"]["reg_lambda"],
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
        "best_learning_rate": result["best_learning_rate"],
        "best_reg_alpha": result["best_reg_alpha"],
        "best_reg_lambda": result["best_reg_lambda"],
        "best_subsample": result["best_subsample"],
        "best_colsample_bytree": result["best_colsample_bytree"],
        "duration_seconds": result["duration_seconds"],
        "recommended_config": config_json_dict,
    }


__all__ = [
    "OptimizationResult",
    "TrialProgressCallbackProtocol",
    "TrialProgressInfo",
    "process_optimize_job",
    "run_optimization",
]
