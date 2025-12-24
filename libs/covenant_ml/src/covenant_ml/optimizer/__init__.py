"""Hyperparameter optimization module for covenant_ml.

Provides pluggable optimization strategies with Bayesian (Optuna TPE),
random search, and grid search algorithms. Supports XGBoost, MLP, LSTM,
and LightGBM backends through a unified interface.

Key components:
- HyperparameterOptimizerProtocol: Unified protocol for all optimizers
- OptimizerStrategyRegistry: Registry for pluggable optimizer strategies
- OptunaTpeOptimizer: Bayesian optimization using TPE
- RandomSearchOptimizer: Random sampling from search space
- GridSearchOptimizer: Exhaustive grid search

Usage:
    from covenant_ml.optimizer import (
        default_optimizer_registry,
        make_xgboost_default_space,
        make_default_optimization_config,
    )

    # Get optimizer from registry
    registry = default_optimizer_registry()
    optimizer = registry.get("optuna_tpe")

    # Create search space and config
    space = make_xgboost_default_space()
    config = make_default_optimization_config(n_trials=100)

    # Run optimization
    summary = optimizer.optimize(
        x_features=X,
        y_labels=y,
        feature_names=names,
        search_space=space,
        config=config,
        objective=my_objective_function,
    )

    # Or use backend-specific optimizers directly
    from covenant_ml.optimizer import create_xgboost_optimizer, use_real_optuna
    use_real_optuna()
    optimizer = create_xgboost_optimizer()
"""

from .objectives import (
    LightGBMObjective,
    LSTMObjective,
    MLPObjective,
    XGBoostObjective,
    create_lightgbm_objective,
    create_lstm_objective,
    create_mlp_objective,
    create_xgboost_objective,
)
from .optuna_backend import (
    OptunaLightGBMOptimizer,
    OptunaLSTMOptimizer,
    OptunaMLPOptimizer,
    OptunaXGBoostOptimizer,
    create_lightgbm_optimizer,
    create_lstm_optimizer,
    create_mlp_optimizer,
    create_xgboost_optimizer,
    set_optuna_module_hook,
    use_real_optuna,
)
from .protocol import (
    LightGBMOptimizerProtocol,
    LSTMOptimizerProtocol,
    MLPOptimizerProtocol,
    ObjectiveProtocol,
    TrialCallbackProtocol,
    XGBoostOptimizerProtocol,
)
from .registry import (
    OptimizerStrategyRegistration,
    OptimizerStrategyRegistry,
    default_optimizer_registry,
)
from .search_spaces import (
    make_default_optimization_config,
    make_lightgbm_default_space,
    make_lightgbm_focused_space,
    make_lstm_default_space,
    make_lstm_focused_space,
    make_mlp_default_space,
    make_mlp_focused_space,
    make_xgboost_categorical_space,
    make_xgboost_default_space,
    make_xgboost_focused_space,
)
from .strategies import (
    GridSearchOptimizer,
    OptunaTpeOptimizer,
    RandomSearchOptimizer,
    create_grid_search_optimizer,
    create_optuna_tpe_optimizer,
    create_random_search_optimizer,
)
from .strategy_protocol import (
    HyperparameterOptimizerProtocol,
    OptimizerStrategyCapabilities,
    OptimizerStrategyFactory,
    OptimizerStrategyName,
)
from .types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    FloatRangeSpec,
    IntRangeSpec,
    LightGBMSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    OptimizationConfig,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SearchSpace,
    TrialResult,
    TrialState,
    XGBoostSearchSpace,
)

__all__ = [
    "CategoricalFloatSpec",
    "CategoricalIntSpec",
    "FloatRangeSpec",
    "GridSearchOptimizer",
    "HyperparameterOptimizerProtocol",
    "IntRangeSpec",
    "LSTMObjective",
    "LSTMOptimizerProtocol",
    "LSTMSearchSpace",
    "LightGBMObjective",
    "LightGBMOptimizerProtocol",
    "LightGBMSearchSpace",
    "MLPObjective",
    "MLPOptimizerProtocol",
    "MLPSearchSpace",
    "ObjectiveProtocol",
    "OptimizationConfig",
    "OptimizationSummary",
    "OptimizerStrategyCapabilities",
    "OptimizerStrategyFactory",
    "OptimizerStrategyName",
    "OptimizerStrategyRegistration",
    "OptimizerStrategyRegistry",
    "OptunaLSTMOptimizer",
    "OptunaLightGBMOptimizer",
    "OptunaMLPOptimizer",
    "OptunaTpeOptimizer",
    "OptunaXGBoostOptimizer",
    "RandomSearchOptimizer",
    "SampledFloatParams",
    "SampledIntParams",
    "SearchSpace",
    "TrialCallbackProtocol",
    "TrialResult",
    "TrialState",
    "XGBoostObjective",
    "XGBoostOptimizerProtocol",
    "XGBoostSearchSpace",
    "create_grid_search_optimizer",
    "create_lightgbm_objective",
    "create_lightgbm_optimizer",
    "create_lstm_objective",
    "create_lstm_optimizer",
    "create_mlp_objective",
    "create_mlp_optimizer",
    "create_optuna_tpe_optimizer",
    "create_random_search_optimizer",
    "create_xgboost_objective",
    "create_xgboost_optimizer",
    "default_optimizer_registry",
    "make_default_optimization_config",
    "make_lightgbm_default_space",
    "make_lightgbm_focused_space",
    "make_lstm_default_space",
    "make_lstm_focused_space",
    "make_mlp_default_space",
    "make_mlp_focused_space",
    "make_xgboost_categorical_space",
    "make_xgboost_default_space",
    "make_xgboost_focused_space",
    "set_optuna_module_hook",
    "use_real_optuna",
]
