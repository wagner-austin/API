"""XGBoost wrapper for covenant breach risk prediction."""

from __future__ import annotations

from covenant_ml.metrics import (
    compute_accuracy,
    compute_all_metrics,
    compute_auc,
    compute_f1_score,
    compute_log_loss,
    compute_precision,
    compute_recall,
    format_metrics_str,
)
from covenant_ml.metrics_regression import (
    compute_all_regression_metrics,
    compute_mae,
    compute_mape,
    compute_mse,
    compute_r_squared,
    compute_rmse,
    format_regression_metrics_str,
)
from covenant_ml.trainer import (
    DataSplits,
    RegressionDataSplits,
    regression_split,
    stratified_split,
)
from covenant_ml.trainer_fit import (
    save_model,
    train_model,
    train_model_with_validation,
)
from covenant_ml.trainer_regression_fit import train_regression_model_with_validation
from covenant_ml.types import (
    EvalMetrics,
    FeatureImportance,
    Proba2DProtocol,
    TrainConfig,
    TrainOutcome,
    TrainProgress,
    XGBBoosterProtocol,
    XGBClassifierFactory,
    XGBClassifierLoader,
    XGBModelProtocol,
)
from covenant_ml.types_regression import (
    RegressionMetrics,
    RegressionTrainOutcome,
    RegressionTrainProgress,
    RegressorBackendName,
    RegressorTrainConfig,
    XGBRegressorFactory,
    XGBRegressorModelProtocol,
)

from .features import (
    EngineeredFeatures,
    FeatureEngineeringConfig,
    FeaturePreset,
    compute_log_transforms,
    compute_pairwise_products,
    compute_pairwise_ratios,
    default_feature_config,
    engineer_features,
    get_feature_config_for_preset,
)
from .predictor import load_model, predict_probabilities

__all__ = [
    "DataSplits",
    "EngineeredFeatures",
    "EvalMetrics",
    "FeatureEngineeringConfig",
    "FeatureImportance",
    "FeaturePreset",
    "Proba2DProtocol",
    "RegressionDataSplits",
    "RegressionMetrics",
    "RegressionTrainOutcome",
    "RegressionTrainProgress",
    "RegressorBackendName",
    "RegressorTrainConfig",
    "TrainConfig",
    "TrainOutcome",
    "TrainProgress",
    "XGBBoosterProtocol",
    "XGBClassifierFactory",
    "XGBClassifierLoader",
    "XGBModelProtocol",
    "XGBRegressorFactory",
    "XGBRegressorModelProtocol",
    "compute_accuracy",
    "compute_all_metrics",
    "compute_all_regression_metrics",
    "compute_auc",
    "compute_f1_score",
    "compute_log_loss",
    "compute_log_transforms",
    "compute_mae",
    "compute_mape",
    "compute_mse",
    "compute_pairwise_products",
    "compute_pairwise_ratios",
    "compute_precision",
    "compute_r_squared",
    "compute_recall",
    "compute_rmse",
    "default_feature_config",
    "engineer_features",
    "format_metrics_str",
    "format_regression_metrics_str",
    "get_feature_config_for_preset",
    "load_model",
    "predict_probabilities",
    "regression_split",
    "save_model",
    "stratified_split",
    "train_model",
    "train_model_with_validation",
    "train_regression_model_with_validation",
]
