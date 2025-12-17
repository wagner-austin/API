"""Objective functions for hyperparameter optimization.

Each backend has its own objective function that:
1. Takes hyperparameters sampled by Optuna
2. Trains a model with those hyperparameters
3. Returns validation AUC for Optuna to maximize

Usage:
    from covenant_ml.optimizer.objectives import (
        XGBoostObjective,
        create_xgboost_objective,
        MLPObjective,
        create_mlp_objective,
        LSTMObjective,
        create_lstm_objective,
        LightGBMObjective,
        create_lightgbm_objective,
    )

    # XGBoost objective
    xgb_objective = create_xgboost_objective(
        x_features=X,
        y_labels=y,
        feature_names=names,
        device="auto",
        feature_preset="none",
    )

    # MLP objective
    mlp_objective = create_mlp_objective(
        x_features=X,
        y_labels=y,
        feature_names=names,
        device="auto",
        precision="fp32",
        feature_preset="none",
        n_epochs=20,
        early_stopping_patience=5,
    )

    # LSTM objective
    lstm_objective = create_lstm_objective(
        x_features=X,
        y_labels=y,
        feature_names=names,
        device="auto",
        precision="fp32",
        feature_preset="none",
        n_epochs=20,
        early_stopping_patience=5,
        sequence_length=5,
        bidirectional=False,
    )

    # LightGBM objective
    lgb_objective = create_lightgbm_objective(
        x_features=X,
        y_labels=y,
        feature_names=names,
        device="auto",
        feature_preset="none",
        early_stopping_rounds=10,
    )

    # Pass to optimizer.optimize()
"""

from .lightgbm_objective import (
    LightGBMObjective,
    create_lightgbm_objective,
)
from .lstm_objective import (
    LSTMObjective,
    create_lstm_objective,
)
from .mlp_objective import (
    MLPObjective,
    create_mlp_objective,
)
from .xgboost_objective import (
    XGBoostObjective,
    create_xgboost_objective,
)

__all__ = [
    "LSTMObjective",
    "LightGBMObjective",
    "MLPObjective",
    "XGBoostObjective",
    "create_lightgbm_objective",
    "create_lstm_objective",
    "create_mlp_objective",
    "create_xgboost_objective",
]
