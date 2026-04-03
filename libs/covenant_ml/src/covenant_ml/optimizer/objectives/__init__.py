"""Objective functions for hyperparameter optimization.

Each backend has its own objective function that:
1. Takes hyperparameters sampled by Optuna
2. Trains a model with those hyperparameters
3. Returns validation metric for Optuna to maximize

Classification objectives return AUC (higher is better).
Regression objectives return negative RMSE (higher = lower error = better).

PyTorch objectives (MLP, LSTM) live in covenant_nn.

Usage:
    from covenant_ml.optimizer.objectives import (
        XGBoostObjective,
        create_xgboost_objective,
        LightGBMObjective,
        create_lightgbm_objective,
        ClearGBMObjective,
        create_cleargbm_objective,
        LogRegObjective,
        create_logreg_objective,
        RandomForestObjective,
        create_random_forest_objective,
        XGBoostRegressorObjective,
        create_xgboost_regressor_objective,
        LightGBMRegressorObjective,
        create_lightgbm_regressor_objective,
    )
"""

from .cleargbm_objective import (
    ClearGBMObjective,
    create_cleargbm_objective,
)
from .lightgbm_objective import (
    LightGBMObjective,
    create_lightgbm_objective,
)
from .lightgbm_regressor_objective import (
    LightGBMRegressorObjective,
    create_lightgbm_regressor_objective,
)
from .logreg_objective import (
    LogRegObjective,
    create_logreg_objective,
)
from .random_forest_objective import (
    RandomForestObjective,
    create_random_forest_objective,
)
from .xgboost_objective import (
    XGBoostObjective,
    create_xgboost_objective,
)
from .xgboost_regressor_objective import (
    XGBoostRegressorObjective,
    create_xgboost_regressor_objective,
)

__all__ = [
    "ClearGBMObjective",
    "LightGBMObjective",
    "LightGBMRegressorObjective",
    "LogRegObjective",
    "RandomForestObjective",
    "XGBoostObjective",
    "XGBoostRegressorObjective",
    "create_cleargbm_objective",
    "create_lightgbm_objective",
    "create_lightgbm_regressor_objective",
    "create_logreg_objective",
    "create_random_forest_objective",
    "create_xgboost_objective",
    "create_xgboost_regressor_objective",
]
