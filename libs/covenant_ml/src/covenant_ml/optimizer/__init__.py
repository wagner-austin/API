"""Hyperparameter optimization module for covenant_ml.

Provides Bayesian optimization using Optuna's TPE algorithm.
Supports XGBoost, MLP, LSTM, and LightGBM backends.

Usage:
    from covenant_ml.optimizer import (
        create_xgboost_optimizer,
        make_xgboost_default_space,
        make_default_optimization_config,
        use_real_optuna,
    )

    # Set up Optuna hook at application startup
    use_real_optuna()

    # Create optimizer and search space
    optimizer = create_xgboost_optimizer()
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
    "OptunaLSTMOptimizer",
    "OptunaLightGBMOptimizer",
    "OptunaMLPOptimizer",
    "OptunaXGBoostOptimizer",
    "SampledFloatParams",
    "SampledIntParams",
    "SearchSpace",
    "TrialCallbackProtocol",
    "TrialResult",
    "TrialState",
    "XGBoostObjective",
    "XGBoostOptimizerProtocol",
    "XGBoostSearchSpace",
    "create_lightgbm_objective",
    "create_lightgbm_optimizer",
    "create_lstm_objective",
    "create_lstm_optimizer",
    "create_mlp_objective",
    "create_mlp_optimizer",
    "create_xgboost_objective",
    "create_xgboost_optimizer",
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
