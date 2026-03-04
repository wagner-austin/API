"""Namespace for LSTM backend.

Exports factory symbol to be used by the registry. Implementation is
provided by backend.py in this package.

Includes sequence building utilities for proper temporal modeling:
- build_sequences: Build temporal sequences from entity/year data
- reshape_flat_to_pseudo_sequences: Reshape flat data to pseudo-sequences
"""

from __future__ import annotations

from .backend import LSTM_CAPABILITIES, create_lstm_backend
from .regressor import LSTM_REGRESSOR_CAPABILITIES, create_lstm_regressor_backend
from .sequences import SequenceData, build_sequences, reshape_flat_to_pseudo_sequences

__all__ = [
    "LSTM_CAPABILITIES",
    "LSTM_REGRESSOR_CAPABILITIES",
    "SequenceData",
    "build_sequences",
    "create_lstm_backend",
    "create_lstm_regressor_backend",
    "reshape_flat_to_pseudo_sequences",
]
