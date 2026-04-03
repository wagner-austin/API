"""Optuna-based hyperparameter optimizers.

Per-backend optimizer implementations using Optuna TPE.
Each backend has its own module with an optimizer class and factory function.

Strict typing only: no Any, no casts, no stubs.
"""

from ._hooks import set_optuna_module_hook, use_real_optuna
from .cleargbm import OptunaClearGBMOptimizer, create_cleargbm_optimizer
from .lightgbm import OptunaLightGBMOptimizer, create_lightgbm_optimizer
from .logreg import OptunaLogRegOptimizer, create_logreg_optimizer
from .lstm import OptunaLSTMOptimizer, create_lstm_optimizer
from .mlp import OptunaMLPOptimizer, create_mlp_optimizer
from .random_forest import OptunaRandomForestOptimizer, create_random_forest_optimizer
from .xgboost import OptunaXGBoostOptimizer, create_xgboost_optimizer

__all__ = [
    "OptunaClearGBMOptimizer",
    "OptunaLSTMOptimizer",
    "OptunaLightGBMOptimizer",
    "OptunaLogRegOptimizer",
    "OptunaMLPOptimizer",
    "OptunaRandomForestOptimizer",
    "OptunaXGBoostOptimizer",
    "create_cleargbm_optimizer",
    "create_lightgbm_optimizer",
    "create_logreg_optimizer",
    "create_lstm_optimizer",
    "create_mlp_optimizer",
    "create_random_forest_optimizer",
    "create_xgboost_optimizer",
    "set_optuna_module_hook",
    "use_real_optuna",
]
