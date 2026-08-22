"""Regression model training with validation-based early stopping."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.metrics_regression import (
    compute_all_regression_metrics,
    format_regression_metrics_str,
)
from covenant_ml.trainer import regression_split
from covenant_ml.trainer_fit import (
    _resolve_device,
    _XGBRegressorModuleProto,
    extract_feature_importances,
    save_model,
)
from covenant_ml.types import TrainConfig
from covenant_ml.types_regression import (
    RegressionTrainOutcome,
    RegressionTrainProgress,
    XGBRegressorFactory,
    XGBRegressorModelProtocol,
)

_log = get_logger(__name__)


def _get_regression_predictions(
    model: XGBRegressorModelProtocol,
    x_features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Get regression predictions from a trained model.

    Args:
        model: Trained XGBoost regressor.
        x_features: Feature matrix, shape (n_samples, n_features).

    Returns:
        Predicted values, shape (n_samples,).
    """
    return model.predict(x_features)


def train_regression_model_with_validation(
    x_features: NDArray[np.float64],
    y_targets: NDArray[np.float64],
    config: TrainConfig,
    output_dir: Path,
    feature_names: list[str],
    progress_callback: Callable[[RegressionTrainProgress], None] | None = None,
) -> RegressionTrainOutcome:
    """Train XGBoost regressor with validation and early stopping.

    Parallel to train_model_with_validation for classification.
    Key differences:
    - objective='reg:squarederror', eval_metric='rmse'
    - No scale_pos_weight (regression has no class imbalance)
    - Early stopping on validation RMSE (lower is better)
    - Uses regression_split (random, not stratified)
    - Returns RegressionTrainOutcome with RegressionMetrics

    Args:
        x_features: Feature matrix (n_samples, n_features).
        y_targets: Continuous target values (n_samples,).
        config: Training configuration with hyperparameters.
        output_dir: Directory to save model artifacts.
        feature_names: List of feature names for importance reporting.
        progress_callback: Optional callback for progress updates.

    Returns:
        RegressionTrainOutcome with complete training results.
    """
    xgb = __import__("xgboost")
    xgb_module: _XGBRegressorModuleProto = xgb
    regressor_factory: XGBRegressorFactory = xgb_module.XGBRegressor
    resolved_device = _resolve_device(config["device"], xgb_module)
    n_jobs = max(1, int(os.cpu_count() or 1))

    splits = regression_split(
        x_features,
        y_targets,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        random_state=config["random_state"],
    )

    def _build_regressor(
        total_estimators: int,
    ) -> XGBRegressorModelProtocol:
        return regressor_factory(
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            n_estimators=total_estimators,
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            random_state=config["random_state"],
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=n_jobs,
            tree_method="hist",
            device=resolved_device,
            reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"],
        )

    n_estimators = config["n_estimators"]
    early_stopping_rounds = config["early_stopping_rounds"]

    # Track best model state (RMSE: lower is better)
    best_val_rmse = float("inf")
    best_round = 0
    rounds_no_improve = 0
    early_stopped = False

    model: XGBRegressorModelProtocol | None = None
    current_round = 0

    for current_round in range(1, n_estimators + 1):
        model = _build_regressor(current_round)
        model.fit(splits.x_train, splits.y_train, verbose=False)

        # Evaluate on train and validation sets
        train_preds = _get_regression_predictions(
            model,
            splits.x_train,
        )
        val_preds = _get_regression_predictions(
            model,
            splits.x_val,
        )

        train_metrics = compute_all_regression_metrics(
            splits.y_train,
            train_preds,
        )
        val_metrics = compute_all_regression_metrics(
            splits.y_val,
            val_preds,
        )

        # Report progress
        if progress_callback is not None:
            progress_callback(
                RegressionTrainProgress(
                    round=current_round,
                    total_rounds=n_estimators,
                    train_rmse=train_metrics["rmse"],
                    val_rmse=val_metrics["rmse"],
                )
            )

        # Check for improvement (RMSE — lower is better)
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_round = current_round
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1

        # Log progress every 50 rounds
        if current_round % 50 == 0:
            _log.debug(
                "Regression training progress",
                extra={
                    "round": current_round,
                    "total_rounds": n_estimators,
                    "train_rmse": train_metrics["rmse"],
                    "val_rmse": val_metrics["rmse"],
                    "best_val_rmse": best_val_rmse,
                    "best_round": best_round,
                    "rounds_no_improve": rounds_no_improve,
                },
            )

        # Early stopping check
        if rounds_no_improve >= early_stopping_rounds:
            early_stopped = True
            _log.info(
                "Regression early stopping triggered",
                extra={
                    "stopped_at_round": current_round,
                    "best_round": best_round,
                    "best_val_rmse": best_val_rmse,
                    "early_stopping_rounds": early_stopping_rounds,
                },
            )
            break

    # If early stopped, retrain with optimal number of estimators
    if early_stopped and best_round < current_round:
        _log.info(
            "Retraining regressor with optimal estimators",
            extra={"best_round": best_round},
        )
        model = _build_regressor(best_round)
        model.fit(splits.x_train, splits.y_train, verbose=False)
        actual_rounds = best_round
    else:
        actual_rounds = current_round

    if model is None:
        raise RuntimeError("Model not trained - n_estimators must be >= 1")

    # Final evaluation on all splits
    final_train_preds = _get_regression_predictions(
        model,
        splits.x_train,
    )
    final_val_preds = _get_regression_predictions(
        model,
        splits.x_val,
    )
    final_test_preds = _get_regression_predictions(
        model,
        splits.x_test,
    )

    final_train_metrics = compute_all_regression_metrics(
        splits.y_train,
        final_train_preds,
    )
    final_val_metrics = compute_all_regression_metrics(
        splits.y_val,
        final_val_preds,
    )
    final_test_metrics = compute_all_regression_metrics(
        splits.y_test,
        final_test_preds,
    )

    _log.info(
        "Regression training complete",
        extra={
            "total_rounds_trained": actual_rounds,
            "early_stopped": early_stopped,
            "best_round": best_round,
            "train_metrics": format_regression_metrics_str(
                final_train_metrics,
            ),
            "val_metrics": format_regression_metrics_str(
                final_val_metrics,
            ),
            "test_metrics": format_regression_metrics_str(
                final_test_metrics,
            ),
        },
    )

    # Save model
    model_id = str(uuid.uuid4())
    model_filename = f"covenant_reg_{model_id[:8]}.ubj"
    model_path = output_dir / model_filename

    save_model(model, str(model_path))

    _log.info(
        "Regression model saved",
        extra={"model_path": str(model_path)},
    )

    # Extract feature importances
    importances = extract_feature_importances(model, feature_names)

    _log.info(
        "Regression feature importances extracted",
        extra={
            "top_features": [
                {
                    "name": f["name"],
                    "importance": f"{f['importance']:.4f}",
                }
                for f in importances[:3]
            ],
        },
    )

    return RegressionTrainOutcome(
        model_path=str(model_path),
        model_id=model_id,
        samples_total=splits.n_total,
        samples_train=splits.n_train,
        samples_val=splits.n_val,
        samples_test=splits.n_test,
        train_metrics=final_train_metrics,
        val_metrics=final_val_metrics,
        test_metrics=final_test_metrics,
        best_val_rmse=best_val_rmse,
        best_round=best_round,
        total_rounds=actual_rounds,
        early_stopped=early_stopped,
        config=config,
        feature_importances=importances,
    )


__all__ = [
    "train_regression_model_with_validation",
]
