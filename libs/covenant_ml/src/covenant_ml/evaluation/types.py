"""Record types for paired model evaluation.

Strict typing only. No Any, casts, stubs, or type-checking-only imports.

WHY A PAIRED EVALUATION AND NOT ANOTHER SCOREBOARD. The benchmark manifests
report each arm's mean AUC per seed and compare arms by counting which won at
each seed. Over five seeds that is a sign test whose best attainable
two-sided p is 0.0625, so no outcome it can produce rejects at 0.05 (board
task 1e4ab572, recorded in ``docs/RESEARCH.md``). The instrument that can
reject scores BOTH arms on the SAME held-out rows and puts an interval on the
difference: the shared rows cancel the split's own luck, which is most of the
variance a per-arm mean carries.

The intervals rest on INDEPENDENT UNITS, and which units those are is a
property of the data, not of the statistic. Bankruptcy rows are company-years,
so one company's ten years are one unit, not ten; market rows are days, and
neighbouring days are not independent either. :class:`ResamplingUnit` names
the three cases and every interval records which one it assumed.

Every TypedDict here has an ``encode_*`` / ``decode_*`` pair in
:mod:`covenant_ml.evaluation.codec_records` or
:mod:`covenant_ml.evaluation.codec_manifests`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict

from platform_core.comparability import RunFingerprint

from covenant_ml.benchmarking.types import BenchmarkConfig, DatasetInfo, QualityMetrics

# Traceable error codes. Each refusal in this package carries exactly one.
ERR_LENGTH_MISMATCH = "CVML-EVAL-001"
ERR_SINGLE_CLASS = "CVML-EVAL-002"
ERR_TOO_FEW_UNITS = "CVML-EVAL-003"
ERR_INVALID_CONFIDENCE = "CVML-EVAL-004"
ERR_INVALID_RESAMPLES = "CVML-EVAL-005"
ERR_DEGENERATE_RESAMPLE = "CVML-EVAL-006"
ERR_PAYOFF_SHAPE = "CVML-EVAL-007"
ERR_INVALID_PRICE = "CVML-EVAL-008"
ERR_UNKNOWN_FEATURE = "CVML-EVAL-009"
ERR_INVALID_FRACTION = "CVML-EVAL-010"
ERR_INVALID_BLOCK = "CVML-EVAL-011"
ERR_EMPTY_ABLATION = "CVML-EVAL-012"
ERR_REFERENCE_ARM = "CVML-EVAL-013"
ERR_SCHEMA_VERSION = "CVML-EVAL-014"
ERR_INVALID_COST = "CVML-EVAL-015"
ERR_TOO_FEW_SEEDS = "CVML-EVAL-016"
ERR_INVALID_EMBARGO = "CVML-EVAL-017"
ERR_INVALID_ARGUMENT = "CVML-EVAL-018"

#: Schema version of both manifests. Bump on any field change so an old
#: document is refused loudly instead of decoded into wrong types.
EVALUATION_SCHEMA_VERSION = 1


class ResamplingUnit(StrEnum):
    """What one independent observation is.

    ``row``: every held-out row is independent (simulated paths).
    ``cluster``: rows sharing a group code are one observation (company-years).
    ``block``: rows are in time order and each run of ``block_length``
    consecutive rows is one observation (days of a market).
    """

    ROW = "row"
    CLUSTER = "cluster"
    BLOCK = "block"


class IntervalMethod(StrEnum):
    """How an interval was computed.

    ``delong``: DeLong's asymptotic variance of a paired AUC difference.
    ``unit_bootstrap``: percentile bootstrap that resamples whole units.
    ``paired_t``: Student's t on per-row differences.
    ``cluster_robust``: t on per-row differences with the variance summed
    within clusters.
    ``newey_west``: normal interval with a Bartlett-kernel long-run variance,
    which is the Diebold-Mariano test of two forecasts.
    ``across_seeds``: Student's t over one difference per seed, with the
    Nadeau-Bengio correction for overlapping test sets and a floor at the
    mean within-seed standard error (:func:`.intervals.pool_across_seeds`).
    """

    DELONG = "delong"
    UNIT_BOOTSTRAP = "unit_bootstrap"
    MOVING_BLOCK_BOOTSTRAP = "moving_block_bootstrap"
    PAIRED_T = "paired_t"
    CLUSTER_ROBUST = "cluster_robust"
    NEWEY_WEST = "newey_west"
    ACROSS_SEEDS = "across_seeds"


class ComparisonMetric(StrEnum):
    """The quantity a paired comparison differences.

    AUC is higher-is-better; the two losses are lower-is-better; utility is
    the realised payoff of the decisions a model's probabilities drove, and
    is higher-is-better.
    """

    AUC = "auc"
    LOG_LOSS = "log_loss"
    BRIER = "brier"
    UTILITY = "utility"


class IntervalEstimate(TypedDict, total=True):
    """A point estimate with its uncertainty, and how both were obtained.

    Args:
        method: How the interval was computed.
        estimate: The point estimate.
        std_error: Its standard error.
        ci_low: Lower confidence bound.
        ci_high: Upper confidence bound.
        confidence: Coverage of the interval, e.g. 0.95.
        p_value: Two-sided p-value against zero.
        n_units: Independent units the interval rests on. A row count where
            rows are clustered would overstate the evidence, so this is the
            cluster or block count when the unit is not a row.
    """

    method: IntervalMethod
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    confidence: float
    p_value: float
    n_units: int


class PairedComparison(TypedDict, total=True):
    """Two arms scored on the same held-out rows.

    Args:
        metric: What was differenced.
        arm: The arm under test.
        reference: The arm it is measured against.
        value_arm: The arm's value of the metric.
        value_reference: The reference's value of the metric.
        difference: ``value_arm - value_reference`` with its interval.
        n_rows: Held-out rows both arms were scored on.
    """

    metric: ComparisonMetric
    arm: str
    reference: str
    value_arm: float
    value_reference: float
    difference: IntervalEstimate
    n_rows: int


class PolicyOutcome(TypedDict, total=True):
    """What acting on one arm's probabilities was worth.

    Args:
        arm: The arm whose probabilities chose the actions.
        action_counts: Rows on which each action was chosen, in the order of
            the manifest's ``action_names``.
        value: Mean realised utility per row, with its interval.
    """

    arm: str
    action_counts: list[int]
    value: IntervalEstimate


class ResamplingSettings(TypedDict, total=True):
    """How every interval in a manifest treated dependence.

    Args:
        unit: What one independent observation is.
        block_length: Rows per block when ``unit`` is ``block``, else 1.
        n_resamples: Bootstrap resamples, where a bootstrap was used.
        resample_seed: Seed of the bootstrap draws.
        confidence: Coverage of every interval.
    """

    unit: ResamplingUnit
    block_length: int
    n_resamples: int
    resample_seed: int
    confidence: float


class ArmResult(TypedDict, total=True):
    """One arm's held-out quality at one seed.

    Args:
        arm: The arm's name.
        seed: Split and model seed.
        quality: Its held-out quality.
    """

    arm: str
    seed: int
    quality: QualityMetrics


class SeedComparisons(TypedDict, total=True):
    """Everything compared within one seed's split.

    Args:
        seed: Split and model seed.
        n_train: Training rows. With ``n_test`` it sets how much two seeds'
            held-out sets overlap, which the pooled interval corrects for.
        n_test: Held-out rows.
        n_test_units: Independent units among them.
        comparisons: Every arm against the reference, for every metric.
        policies: Each arm's decision value, the reference's included.
    """

    seed: int
    n_train: int
    n_test: int
    n_test_units: int
    comparisons: list[PairedComparison]
    policies: list[PolicyOutcome]


class PooledComparison(TypedDict, total=True):
    """One arm-versus-reference difference carried across seeds.

    Within one seed the interval covers which held-out units happened to be
    drawn; across seeds it also covers which companies trained the model.
    Only the second says whether the difference survives a different split.

    Args:
        metric: What was differenced.
        arm: The arm under test.
        reference: The arm it is measured against.
        per_seed: The within-seed point difference at each seed, in seed order.
        difference: Their mean, with a t interval over seeds.
    """

    metric: ComparisonMetric
    arm: str
    reference: str
    per_seed: list[float]
    difference: IntervalEstimate


class AlertCosts(TypedDict, total=True):
    """Payoffs of the two-action alerting decision, per held-out row.

    Args:
        true_alert_value: Gain from alerting on a row that turns positive.
        false_alert_cost: Cost of alerting on a row that stays negative.
        missed_cost: Cost of not alerting on a row that turns positive.
    """

    true_alert_value: float
    false_alert_cost: float
    missed_cost: float


class EvaluationManifest(TypedDict, total=True):
    """Machine-readable record of one paired evaluation.

    Args:
        schema_version: :data:`EVALUATION_SCHEMA_VERSION`.
        dataset: Identity of the input data.
        config: Hyperparameters every arm was held to.
        resampling: How the intervals treated dependence.
        alert_costs: The payoffs the policy values were scored with.
        action_names: The decision's actions, in payoff order.
        reference: The arm every other arm is compared against.
        seeds: Seeds evaluated, in order.
        results: Every arm's quality at every seed.
        seed_comparisons: Within-seed comparisons, one entry per seed.
        pooled: Across-seed comparisons, one per arm and metric.
        fingerprint: The configuration this ran under.
    """

    schema_version: int
    dataset: DatasetInfo
    config: BenchmarkConfig
    resampling: ResamplingSettings
    alert_costs: AlertCosts
    action_names: list[str]
    reference: str
    seeds: list[int]
    results: list[ArmResult]
    seed_comparisons: list[SeedComparisons]
    pooled: list[PooledComparison]
    fingerprint: RunFingerprint


class LearningCurvePoint(TypedDict, total=True):
    """One arm trained on a prefix of its training fold.

    Args:
        arm: The arm's name.
        seed: Split and model seed.
        train_fraction: Share of the training fold used.
        n_train: Training rows used.
        quality: Held-out quality, on the full held-out fold.
    """

    arm: str
    seed: int
    train_fraction: float
    n_train: int
    quality: QualityMetrics


class AblationGroup(TypedDict, total=True):
    """A named set of features removed together.

    Args:
        name: How the group is reported.
        features: Feature names removed.
    """

    name: str
    features: list[str]


class AblationManifest(TypedDict, total=True):
    """Machine-readable record of one feature ablation and learning curve.

    Args:
        schema_version: :data:`EVALUATION_SCHEMA_VERSION`.
        dataset: Identity of the input data.
        config: Hyperparameters the model was held to.
        model: The learner ablated.
        resampling: How the intervals treated dependence.
        groups: The ablated feature groups.
        seeds: Seeds evaluated, in order.
        results: The full model's and every ablated model's quality per seed.
        seed_comparisons: Each ablated model against the full one, per seed.
        pooled: Across-seed ablation effects, one per group and metric.
        learning_curve: The full model at each training fraction and seed.
        fingerprint: The configuration this ran under.
    """

    schema_version: int
    dataset: DatasetInfo
    config: BenchmarkConfig
    model: str
    resampling: ResamplingSettings
    groups: list[AblationGroup]
    seeds: list[int]
    results: list[ArmResult]
    seed_comparisons: list[SeedComparisons]
    pooled: list[PooledComparison]
    learning_curve: list[LearningCurvePoint]
    fingerprint: RunFingerprint


__all__ = [
    "ERR_DEGENERATE_RESAMPLE",
    "ERR_EMPTY_ABLATION",
    "ERR_INVALID_ARGUMENT",
    "ERR_INVALID_BLOCK",
    "ERR_INVALID_CONFIDENCE",
    "ERR_INVALID_COST",
    "ERR_INVALID_EMBARGO",
    "ERR_INVALID_FRACTION",
    "ERR_INVALID_PRICE",
    "ERR_INVALID_RESAMPLES",
    "ERR_LENGTH_MISMATCH",
    "ERR_PAYOFF_SHAPE",
    "ERR_REFERENCE_ARM",
    "ERR_SCHEMA_VERSION",
    "ERR_SINGLE_CLASS",
    "ERR_TOO_FEW_SEEDS",
    "ERR_TOO_FEW_UNITS",
    "ERR_UNKNOWN_FEATURE",
    "EVALUATION_SCHEMA_VERSION",
    "AblationGroup",
    "AblationManifest",
    "AlertCosts",
    "ArmResult",
    "ComparisonMetric",
    "EvaluationManifest",
    "IntervalEstimate",
    "IntervalMethod",
    "LearningCurvePoint",
    "PairedComparison",
    "PolicyOutcome",
    "PooledComparison",
    "ResamplingSettings",
    "ResamplingUnit",
    "SeedComparisons",
]
