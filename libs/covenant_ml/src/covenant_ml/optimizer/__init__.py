"""Hyperparameter optimization module for covenant_ml.

Provides Bayesian optimization using Optuna's TPE algorithm.
Includes predefined search spaces for XGBoost and optimization utilities.

Usage:
    from covenant_ml.optimizer import (
        create_xgboost_optimizer,
        make_xgboost_default_space,
        make_default_optimization_config,
    )

    optimizer = create_xgboost_optimizer()
    space = make_xgboost_default_space()
    config = make_default_optimization_config(n_trials=100)

    summary = optimizer.optimize(
        x_features=X,
        y_labels=y,
        feature_names=names,
        search_space=space,
        config=config,
        objective=my_objective_function,
    )
"""

from .optuna_backend import (
    OptunaXGBoostOptimizer,
    create_xgboost_optimizer,
    set_optuna_module_hook,
    use_real_optuna,
)
from .protocol import (
    TrialCallbackProtocol,
    XGBoostObjectiveCallable,
    XGBoostObjectiveProtocol,
    XGBoostOptimizerProtocol,
)
from .search_spaces import (
    make_default_optimization_config,
    make_xgboost_categorical_space,
    make_xgboost_default_space,
    make_xgboost_focused_space,
)
from .types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    FloatRangeSpec,
    IntRangeSpec,
    MLPSearchSpace,
    OptimizationConfig,
    OptimizationSummary,
    ParamSpec,
    TrialResult,
    TrialState,
    XGBoostSearchSpace,
)

__all__ = [
    "CategoricalFloatSpec",
    "CategoricalIntSpec",
    "FloatRangeSpec",
    "IntRangeSpec",
    "MLPSearchSpace",
    "OptimizationConfig",
    "OptimizationSummary",
    "OptunaXGBoostOptimizer",
    "ParamSpec",
    "TrialCallbackProtocol",
    "TrialResult",
    "TrialState",
    "XGBoostObjectiveCallable",
    "XGBoostObjectiveProtocol",
    "XGBoostOptimizerProtocol",
    "XGBoostSearchSpace",
    "create_xgboost_optimizer",
    "make_default_optimization_config",
    "make_xgboost_categorical_space",
    "make_xgboost_default_space",
    "make_xgboost_focused_space",
    "set_optuna_module_hook",
    "use_real_optuna",
]
