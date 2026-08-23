"""Type definitions for ClearGBM — re-export layer.

Re-exports all TypedDicts, NamedTuples, encode/decode functions, and
validation helpers from the focused sub-modules:

- ``_types_json`` — JSON aliases, JSONTypeError, validators, extractors
- ``_types_tree`` — BinEdges, FeatureBins, tree structures, SplitCandidate
- ``_types_model`` — GradientBoostingConfig, GradientBoostingModel,
  TrainingProgress
- ``_types_explain`` — FeatureContribution, PredictionExplanation, Rule
- ``_types_tuning`` — TimingResult, TuningReport
- ``_types_buffer`` — FloatBufferData, IntBufferData, HistogramBufferData

All names are immutable (TypedDicts, NamedTuples, functions), so re-export
via ``from module import name`` is safe and does not create binding drift.

This module is the public API — consumers import from ``cleargbm.types``.
"""

from __future__ import annotations

from cleargbm._types_buffer import (
    FloatBufferData,
    HistogramBufferData,
    IntBufferData,
    decode_float_buffer_data,
    decode_histogram_buffer_data,
    decode_int_buffer_data,
    encode_float_buffer_data,
    encode_histogram_buffer_data,
    encode_int_buffer_data,
)
from cleargbm._types_explain import (
    FeatureContribution,
    PredictionExplanation,
    Rule,
    decode_feature_contribution,
    decode_prediction_explanation,
    decode_rule,
    encode_feature_contribution,
    encode_prediction_explanation,
    encode_rule,
)
from cleargbm._types_json import (
    JSONDict,
    JSONTypeError,
    JSONValue,
    require_n_jobs,
    require_non_negative_float,
    require_non_negative_int,
    require_open_unit_float,
    require_positive_float,
    require_positive_int,
    require_unit_float,
)
from cleargbm._types_model import (
    GROWTH_STRATEGIES,
    OBJECTIVES,
    GradientBoostingConfig,
    GradientBoostingModel,
    GrowthStrategy,
    Objective,
    TrainingProgress,
    decode_gradient_boosting_config,
    decode_gradient_boosting_model,
    decode_training_progress,
    encode_gradient_boosting_config,
    encode_gradient_boosting_model,
    encode_training_progress,
    require_growth_strategy,
    require_leaf_budget,
    require_objective,
)
from cleargbm._types_tree import (
    BinEdges,
    DecisionTree,
    FeatureBins,
    SplitCandidate,
    SplitCondition,
    TreeNode,
    TreePredictionExplanation,
    decode_decision_tree,
    decode_split_condition,
    decode_tree_node,
    decode_tree_prediction_explanation,
    encode_decision_tree,
    encode_split_condition,
    encode_tree_node,
    encode_tree_prediction_explanation,
)
from cleargbm._types_tuning import (
    TimingResult,
    TuningReport,
    decode_timing_result,
    decode_tuning_report,
    encode_timing_result,
    encode_tuning_report,
)

__all__ = [
    "GROWTH_STRATEGIES",
    "OBJECTIVES",
    "BinEdges",
    "DecisionTree",
    "FeatureBins",
    "FeatureContribution",
    "FloatBufferData",
    "GradientBoostingConfig",
    "GradientBoostingModel",
    "GrowthStrategy",
    "HistogramBufferData",
    "IntBufferData",
    "JSONDict",
    "JSONTypeError",
    "JSONValue",
    "Objective",
    "PredictionExplanation",
    "Rule",
    "SplitCandidate",
    "SplitCondition",
    "TimingResult",
    "TrainingProgress",
    "TreeNode",
    "TreePredictionExplanation",
    "TuningReport",
    "decode_decision_tree",
    "decode_feature_contribution",
    "decode_float_buffer_data",
    "decode_gradient_boosting_config",
    "decode_gradient_boosting_model",
    "decode_histogram_buffer_data",
    "decode_int_buffer_data",
    "decode_prediction_explanation",
    "decode_rule",
    "decode_split_condition",
    "decode_timing_result",
    "decode_training_progress",
    "decode_tree_node",
    "decode_tree_prediction_explanation",
    "decode_tuning_report",
    "encode_decision_tree",
    "encode_feature_contribution",
    "encode_float_buffer_data",
    "encode_gradient_boosting_config",
    "encode_gradient_boosting_model",
    "encode_histogram_buffer_data",
    "encode_int_buffer_data",
    "encode_prediction_explanation",
    "encode_rule",
    "encode_split_condition",
    "encode_timing_result",
    "encode_training_progress",
    "encode_tree_node",
    "encode_tree_prediction_explanation",
    "encode_tuning_report",
    "require_growth_strategy",
    "require_leaf_budget",
    "require_n_jobs",
    "require_non_negative_float",
    "require_non_negative_int",
    "require_objective",
    "require_open_unit_float",
    "require_positive_float",
    "require_positive_int",
    "require_unit_float",
]
