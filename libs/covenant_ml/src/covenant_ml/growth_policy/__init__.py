"""Growth-policy experiment: what leaf-wise growth buys, measured before building it.

XGBoost implements both a depth-wise and a leaf-wise growth policy, so it can
serve as the instrument for a question about ClearGBM: what would adding
leaf-wise growth actually buy, measured with dataset, splits, seeds and
constraint semantics held constant, before any Rust is written.

This package is deliberately separate from
:mod:`covenant_ml.benchmarking`. That harness answers "how does ClearGBM
compare to LightGBM today" and emits a manifest whose schema published results
are pinned to. This one answers "does growth policy move the outcome at all",
runs a different protocol, and must not be folded into the manifest without
re-measuring -- a point the experiment write-up makes about its own confounds.
What the two do share is lifted rather than copied: the fitted-model wrappers,
the LightGBM vendor Protocol, the timing summariser and the clock Protocol all
come from the benchmarking package.

The layers are independent: :mod:`types` holds the record shapes and codecs,
:mod:`protocols` names every injected boundary, :mod:`vendors` is the only
module that resolves a vendor import, :mod:`datasets`, :mod:`model_shape`,
:mod:`summarize` and :mod:`reporting` are pure, :mod:`trainers` and
:mod:`metrics` are the concrete arms and scorer, :mod:`runner` owns the
measurement protocol, and :mod:`factory` is the only place naming a concrete
implementation.
"""

from __future__ import annotations

from .datasets import (
    BANKRUPTCY_FEATURE_COUNT,
    TRAIN_FRACTION,
    GroupedDataset,
    PlainDataset,
    company_disjoint_indices,
    describe_dataset,
    encode_column,
    load_bankruptcy,
    load_german_credit,
    load_taiwan_bankruptcy,
    sorted_group_codes,
)
from .factory import (
    DEFAULT_LEAF_BUDGETS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_REPEATS,
    DEFAULT_SEEDS,
    DEFAULT_WARMUPS,
    STRATIFIED_TEST_SIZE,
    make_anchor_trainers,
    make_arm_specs,
    make_experiment_config,
    make_group_split_factory,
    make_metrics,
    make_stratified_split_factory,
    make_xgb_trainers,
)
from .metrics import SklearnMetrics
from .protocols import (
    ArmSpec,
    ArmTrainerProto,
    MetricsProto,
    SplitFactoryProto,
    TrainedModelProto,
    TwoWaySplit,
)
from .reporting import render_dataset_line, render_report
from .runner import fit_repeatedly, measure_arm, run_experiment
from .summarize import summarize_arms
from .trainers import (
    ClearGbmAnchorTrainer,
    LgbAnchorTrainer,
    XgbArmTrainer,
    XgbTrainedModel,
)
from .types import (
    REPORT_SCHEMA_VERSION,
    ArmResult,
    ArmSummary,
    DatasetInfo,
    ExperimentConfig,
    GrowthPolicyReport,
    decode_growth_policy_report,
    encode_growth_policy_report,
)

__all__ = [
    "BANKRUPTCY_FEATURE_COUNT",
    "DEFAULT_LEAF_BUDGETS",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_REPEATS",
    "DEFAULT_SEEDS",
    "DEFAULT_WARMUPS",
    "REPORT_SCHEMA_VERSION",
    "STRATIFIED_TEST_SIZE",
    "TRAIN_FRACTION",
    "ArmResult",
    "ArmSpec",
    "ArmSummary",
    "ArmTrainerProto",
    "ClearGbmAnchorTrainer",
    "DatasetInfo",
    "ExperimentConfig",
    "GroupedDataset",
    "GrowthPolicyReport",
    "LgbAnchorTrainer",
    "MetricsProto",
    "PlainDataset",
    "SklearnMetrics",
    "SplitFactoryProto",
    "TrainedModelProto",
    "TwoWaySplit",
    "XgbArmTrainer",
    "XgbTrainedModel",
    "company_disjoint_indices",
    "decode_growth_policy_report",
    "describe_dataset",
    "encode_column",
    "encode_growth_policy_report",
    "fit_repeatedly",
    "load_bankruptcy",
    "load_german_credit",
    "load_taiwan_bankruptcy",
    "make_anchor_trainers",
    "make_arm_specs",
    "make_experiment_config",
    "make_group_split_factory",
    "make_metrics",
    "make_stratified_split_factory",
    "make_xgb_trainers",
    "measure_arm",
    "render_dataset_line",
    "render_report",
    "run_experiment",
    "sorted_group_codes",
    "summarize_arms",
]
