"""Predefined search spaces for hyperparameter optimization.

Each backend has its own module with default and focused search space factories.
The optimization config factory is in config.py.

Strict typing only: no Any, no casts, no stubs.
"""

from .cleargbm import (
    make_cleargbm_default_space,
    make_cleargbm_focused_space,
)
from .config import make_default_optimization_config
from .lightgbm import (
    make_lightgbm_default_space,
    make_lightgbm_focused_space,
)
from .logreg import (
    make_logreg_default_space,
    make_logreg_focused_space,
)
from .lstm import (
    make_lstm_default_space,
    make_lstm_focused_space,
)
from .mlp import (
    make_mlp_default_space,
    make_mlp_focused_space,
)
from .random_forest import (
    make_random_forest_default_space,
    make_random_forest_focused_space,
)
from .xgboost import (
    make_xgboost_categorical_space,
    make_xgboost_default_space,
    make_xgboost_focused_space,
)

__all__ = [
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
