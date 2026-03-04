"""Objective functions for neural network hyperparameter optimization.

Each backend has its own objective function that:
1. Takes hyperparameters sampled by Optuna
2. Trains a model with those hyperparameters
3. Returns validation metric for Optuna to maximize

Classification objectives return AUC (higher is better).
Regression objectives return negative RMSE (higher = lower error = better).
"""

from .lstm_objective import (
    LSTMObjective,
    create_lstm_objective,
)
from .lstm_regressor_objective import (
    LSTMRegressorObjective,
    create_lstm_regressor_objective,
)
from .mlp_objective import (
    MLPObjective,
    create_mlp_objective,
)
from .mlp_regressor_objective import (
    MLPRegressorObjective,
    create_mlp_regressor_objective,
)

__all__ = [
    "LSTMObjective",
    "LSTMRegressorObjective",
    "MLPObjective",
    "MLPRegressorObjective",
    "create_lstm_objective",
    "create_lstm_regressor_objective",
    "create_mlp_objective",
    "create_mlp_regressor_objective",
]
