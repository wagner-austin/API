"""Hyperparameter optimization module for covenant_ml.

Provides pluggable optimization strategies with Bayesian (Optuna TPE),
random search, and grid search algorithms. Supports XGBoost, LightGBM,
ClearGBM, LogReg, RandomForest, MLP, and LSTM search spaces through a unified interface.

PyTorch objectives (MLPObjective, LSTMObjective) live in covenant_nn.

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
    from covenant_ml.optimizer import create_xgboost_optimizer
    optimizer = create_xgboost_optimizer()

Nothing has to be wired at startup. Both optuna injection points -- the module
factories the backend optimizers use and the ones the TPE strategy uses -- are
bound to the real import, so an unwired one is not a state that exists. It used
to be: each had to be set separately, every entry point set only the first
because this docstring said to, and /ml/optimize raised "Optuna TPE hook not
set" for every backend.
"""

from .objectives import (
    ClearGBMObjective,
    LightGBMObjective,
    LightGBMRegressorObjective,
    LogRegObjective,
    RandomForestObjective,
    XGBoostObjective,
    XGBoostRegressorObjective,
    create_cleargbm_objective,
    create_lightgbm_objective,
    create_lightgbm_regressor_objective,
    create_logreg_objective,
    create_random_forest_objective,
    create_xgboost_objective,
    create_xgboost_regressor_objective,
)
from .optuna_backend import (
    OptunaClearGBMOptimizer,
    OptunaLightGBMOptimizer,
    OptunaLogRegOptimizer,
    OptunaLSTMOptimizer,
    OptunaMLPOptimizer,
    OptunaRandomForestOptimizer,
    OptunaXGBoostOptimizer,
    create_cleargbm_optimizer,
    create_lightgbm_optimizer,
    create_logreg_optimizer,
    create_lstm_optimizer,
    create_mlp_optimizer,
    create_random_forest_optimizer,
    create_xgboost_optimizer,
)
from .protocol import (
    ClearGBMOptimizerProtocol,
    LightGBMOptimizerProtocol,
    LogRegOptimizerProtocol,
    LSTMOptimizerProtocol,
    MLPOptimizerProtocol,
    ObjectiveProtocol,
    RandomForestOptimizerProtocol,
    TrialCallbackProtocol,
    XGBoostOptimizerProtocol,
)
from .registry import (
    OptimizerStrategyRegistration,
    OptimizerStrategyRegistry,
    default_optimizer_registry,
)
from .search_spaces import (
    make_cleargbm_default_space,
    make_cleargbm_focused_space,
    make_default_optimization_config,
    make_lightgbm_default_space,
    make_lightgbm_focused_space,
    make_logreg_default_space,
    make_logreg_focused_space,
    make_lstm_default_space,
    make_lstm_focused_space,
    make_mlp_default_space,
    make_mlp_focused_space,
    make_random_forest_default_space,
    make_random_forest_focused_space,
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
    ClearGBMSearchSpace,
    FloatRangeSpec,
    IntRangeSpec,
    LightGBMSearchSpace,
    LogRegSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    OptimizationConfig,
    OptimizationSummary,
    RandomForestSearchSpace,
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
    "ClearGBMObjective",
    "ClearGBMOptimizerProtocol",
    "ClearGBMSearchSpace",
    "FloatRangeSpec",
    "GridSearchOptimizer",
    "HyperparameterOptimizerProtocol",
    "IntRangeSpec",
    "LSTMOptimizerProtocol",
    "LSTMSearchSpace",
    "LightGBMObjective",
    "LightGBMOptimizerProtocol",
    "LightGBMRegressorObjective",
    "LightGBMSearchSpace",
    "LogRegObjective",
    "LogRegOptimizerProtocol",
    "LogRegSearchSpace",
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
    "OptunaClearGBMOptimizer",
    "OptunaLSTMOptimizer",
    "OptunaLightGBMOptimizer",
    "OptunaLogRegOptimizer",
    "OptunaMLPOptimizer",
    "OptunaRandomForestOptimizer",
    "OptunaTpeOptimizer",
    "OptunaXGBoostOptimizer",
    "RandomForestObjective",
    "RandomForestOptimizerProtocol",
    "RandomForestSearchSpace",
    "RandomSearchOptimizer",
    "SampledFloatParams",
    "SampledIntParams",
    "SearchSpace",
    "TrialCallbackProtocol",
    "TrialResult",
    "TrialState",
    "XGBoostObjective",
    "XGBoostOptimizerProtocol",
    "XGBoostRegressorObjective",
    "XGBoostSearchSpace",
    "create_cleargbm_objective",
    "create_cleargbm_optimizer",
    "create_grid_search_optimizer",
    "create_lightgbm_objective",
    "create_lightgbm_optimizer",
    "create_lightgbm_regressor_objective",
    "create_logreg_objective",
    "create_logreg_optimizer",
    "create_lstm_optimizer",
    "create_mlp_optimizer",
    "create_optuna_tpe_optimizer",
    "create_random_forest_objective",
    "create_random_forest_optimizer",
    "create_random_search_optimizer",
    "create_xgboost_objective",
    "create_xgboost_optimizer",
    "create_xgboost_regressor_objective",
    "default_optimizer_registry",
    "make_cleargbm_default_space",
    "make_cleargbm_focused_space",
    "make_default_optimization_config",
    "make_lightgbm_default_space",
    "make_lightgbm_focused_space",
    "make_logreg_default_space",
    "make_logreg_focused_space",
    "make_lstm_default_space",
    "make_lstm_focused_space",
    "make_mlp_default_space",
    "make_mlp_focused_space",
    "make_random_forest_default_space",
    "make_random_forest_focused_space",
    "make_xgboost_categorical_space",
    "make_xgboost_default_space",
    "make_xgboost_focused_space",
]
