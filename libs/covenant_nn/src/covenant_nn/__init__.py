"""PyTorch neural network backends (MLP, LSTM) for covenant_ml.

Provides MLP and LSTM classifier and regressor backends that implement
the covenant_ml ClassifierBackend and RegressorBackend protocols, plus
Optuna objective functions for hyperparameter optimization.
"""

from __future__ import annotations

from .backends.lstm import (
    LSTM_CAPABILITIES,
    LSTM_REGRESSOR_CAPABILITIES,
    create_lstm_backend,
    create_lstm_regressor_backend,
)
from .backends.mlp import (
    MLP_CAPABILITIES,
    MLP_REGRESSOR_CAPABILITIES,
    create_mlp_backend,
    create_mlp_regressor_backend,
)
from .objectives.lstm_objective import LSTMObjective, create_lstm_objective
from .objectives.lstm_regressor_objective import (
    LSTMRegressorObjective,
    create_lstm_regressor_objective,
)
from .objectives.mlp_objective import MLPObjective, create_mlp_objective
from .objectives.mlp_regressor_objective import (
    MLPRegressorObjective,
    create_mlp_regressor_objective,
)

__all__ = [
    "LSTM_CAPABILITIES",
    "LSTM_REGRESSOR_CAPABILITIES",
    "MLP_CAPABILITIES",
    "MLP_REGRESSOR_CAPABILITIES",
    "LSTMObjective",
    "LSTMRegressorObjective",
    "MLPObjective",
    "MLPRegressorObjective",
    "create_lstm_backend",
    "create_lstm_objective",
    "create_lstm_regressor_backend",
    "create_lstm_regressor_objective",
    "create_mlp_backend",
    "create_mlp_objective",
    "create_mlp_regressor_backend",
    "create_mlp_regressor_objective",
]
