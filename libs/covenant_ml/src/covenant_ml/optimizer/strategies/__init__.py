"""Hyperparameter optimization strategy implementations.

Provides pluggable optimizer implementations that satisfy HyperparameterOptimizerProtocol.
Each strategy can be registered in OptimizerStrategyRegistry and used interchangeably.

Strategies:
- OptunaTpeOptimizer: Bayesian optimization using Optuna's TPE algorithm
- RandomSearchOptimizer: Random sampling from search space
- GridSearchOptimizer: Exhaustive grid search for small spaces
"""

from .grid_search import (
    GridSearchOptimizer,
    GridTuple,
    create_grid_search_optimizer,
)
from .optuna_tpe import (
    OptunaTpeOptimizer,
    create_optuna_tpe_optimizer,
)
from .random_search import (
    RandomSearchOptimizer,
    create_random_search_optimizer,
)

__all__ = [
    "GridSearchOptimizer",
    "GridTuple",
    "OptunaTpeOptimizer",
    "RandomSearchOptimizer",
    "create_grid_search_optimizer",
    "create_optuna_tpe_optimizer",
    "create_random_search_optimizer",
]
