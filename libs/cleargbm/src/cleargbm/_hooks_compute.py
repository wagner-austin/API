"""Computation backend hooks for cleargbm — re-export layer.

Re-exports all hook protocols, default implementations, and public delegator
functions from the focused sub-modules:

- ``_hooks_histogram`` — build_histogram, subtract_histogram
- ``_hooks_prediction`` — predict_tree
- ``_hooks_sigmoid`` — sigmoid, sigmoid_array
- ``_hooks_loss`` — binary_log_loss, gradients, hessians, initial_prediction
- ``_hooks_binning`` — precompute_feature_bins
- ``_hooks_ensemble`` — predict_raw_ensemble, predict_proba_from_raw

Mutable hook variables (``_*_backend``) must be set on their originating
sub-module, not on this re-export module.

This module is private (underscore prefix) - not for external use.
"""

from __future__ import annotations

from cleargbm._hooks_binning import (
    PrecomputeFeatureBinsBackend,
    precompute_feature_bins,
)
from cleargbm._hooks_ensemble import (
    PredictProbaBackend,
    PredictRawBackend,
    predict_proba_from_raw,
    predict_raw_ensemble,
)
from cleargbm._hooks_histogram import (
    BuildHistogramBackend,
    SubtractHistogramBackend,
    build_histogram,
    subtract_histogram,
)
from cleargbm._hooks_loss import (
    BinaryLogLossBackend,
    BinaryLogLossGradientsBackend,
    BinaryLogLossHessiansBackend,
    BinaryLogLossInitialPredictionBackend,
    binary_log_loss,
    binary_log_loss_gradients,
    binary_log_loss_hessians,
    binary_log_loss_initial_prediction,
)
from cleargbm._hooks_prediction import (
    PredictTreeBackend,
    predict_tree,
)
from cleargbm._hooks_sigmoid import (
    SigmoidArrayBackend,
    SigmoidBackend,
    sigmoid,
    sigmoid_array,
)

__all__ = [
    "BinaryLogLossBackend",
    "BinaryLogLossGradientsBackend",
    "BinaryLogLossHessiansBackend",
    "BinaryLogLossInitialPredictionBackend",
    "BuildHistogramBackend",
    "PrecomputeFeatureBinsBackend",
    "PredictProbaBackend",
    "PredictRawBackend",
    "PredictTreeBackend",
    "SigmoidArrayBackend",
    "SigmoidBackend",
    "SubtractHistogramBackend",
    "binary_log_loss",
    "binary_log_loss_gradients",
    "binary_log_loss_hessians",
    "binary_log_loss_initial_prediction",
    "build_histogram",
    "precompute_feature_bins",
    "predict_proba_from_raw",
    "predict_raw_ensemble",
    "predict_tree",
    "sigmoid",
    "sigmoid_array",
    "subtract_histogram",
]
