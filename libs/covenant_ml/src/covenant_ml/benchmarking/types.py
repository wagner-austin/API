"""Type definitions for the ClearGBM-versus-LightGBM benchmark.

Strict typing only. No Any, casts, stubs, or type-checking-only imports.

Every TypedDict in this module has a matching ``encode_*`` function that
lowers it to :data:`JSONValue`, and a matching ``decode_*`` function that
raises it back out of untrusted JSON through ``_require_*`` validators.
Decoding never softens a malformed document: the first invalid field raises
:class:`ValueError` carrying a traceable error code and a human message.
"""

from __future__ import annotations

from typing import Literal, TypedDict, get_args

from platform_core.comparability import RunFingerprint

# Traceable error codes. Every decode failure carries exactly one of these so
# a malformed manifest can be triaged from a log line alone.
ERR_NOT_MAPPING = "CVML-BENCH-001"
ERR_NOT_STR = "CVML-BENCH-002"
ERR_NOT_FLOAT = "CVML-BENCH-003"
ERR_NOT_INT = "CVML-BENCH-004"
ERR_NOT_BOOL = "CVML-BENCH-005"
ERR_NOT_LIST = "CVML-BENCH-006"
ERR_UNKNOWN_MODEL = "CVML-BENCH-007"
ERR_UNKNOWN_ESTIMATOR = "CVML-BENCH-008"
ERR_SCHEMA_VERSION = "CVML-BENCH-009"
ERR_NO_TIMING_SAMPLES = "CVML-BENCH-010"
ERR_MISSING_COLUMN = "CVML-BENCH-011"
ERR_EMPTY_SPLIT = "CVML-BENCH-012"
ERR_LENGTH_MISMATCH = "CVML-BENCH-013"
ERR_NO_TREES = "CVML-BENCH-014"
ERR_INVALID_REPEATS = "CVML-BENCH-015"
ERR_NO_SEEDS = "CVML-BENCH-016"
ERR_NO_RESULTS = "CVML-BENCH-017"
ERR_TOO_FEW_TRAINERS = "CVML-BENCH-018"
ERR_DUPLICATE_TRAINER = "CVML-BENCH-019"
ERR_POWER_THROTTLING = "CVML-BENCH-020"

#: Schema version of :class:`BenchmarkManifest`. Bump on any field change so
#: an old manifest is rejected loudly instead of decoded into wrong types.
#: Bumped to 2 when the harness stopped being a two-model comparison:
#: ``ran_first: bool`` cannot describe an ordering over three or more arms, so
#: it became ``position: int``.
#:
#: Bumped to 3 on 2026-08-27 when the manifest grew ``fingerprint``. A version
#: 2 document has no environment block, and its decoder must refuse rather
#: than default one: an absent fingerprint would compare EQUAL to another
#: absent one, reporting two runs on two machines as one configuration, which
#: is precisely the defect the field exists to end.
MANIFEST_SCHEMA_VERSION = 3

#: The arms this benchmark can compare.
#:
#: A closed literal, not an open string: an arm name is what a manifest is read
#: by, so a typo must fail at the boundary rather than silently produce a
#: fourth, nameless series. Variant arms are spelled ``<model>@<variant>``.
BenchmarkModelName = Literal["cleargbm", "cleargbm@leaf_wise", "lightgbm", "xgboost"]

#: Every accepted :data:`BenchmarkModelName`, for validation and iteration.
BENCHMARK_MODEL_NAMES: tuple[BenchmarkModelName, ...] = get_args(BenchmarkModelName)

#: The statistic taken as each seed's canonical fit time.
#:
#: ``median`` is the only permitted value. The minimum is deliberately not
#: offered: the first fits after an idle period run with full turbo headroom,
#: a different power regime rather than noise, so a minimum reports a
#: cold-start outlier as though it were the steady state that sustained
#: training actually experiences.
TimingEstimator = Literal["median"]


class TimingSummary(TypedDict, total=True):
    """Fit-time statistics over the timed repeats of one model at one seed.

    Args:
        canonical_s: The value callers should compare, in seconds. Always the
            median of ``samples_s``.
        min_s: Fastest timed repeat, in seconds.
        median_s: Median timed repeat, in seconds.
        mean_s: Arithmetic mean of timed repeats, in seconds.
        max_s: Slowest timed repeat, in seconds.
        samples_s: Every timed repeat, in seconds, in execution order.
    """

    canonical_s: float
    min_s: float
    median_s: float
    mean_s: float
    max_s: float
    samples_s: list[float]


class QualityMetrics(TypedDict, total=True):
    """Predictive-quality metrics on the held-out split.

    Recorded alongside timing so a change that trades accuracy for speed is
    visible in the same record rather than discovered later.

    Args:
        auc_roc: Area under the ROC curve.
        auc_pr: Area under the precision-recall curve (average precision).
        log_loss: Binary cross-entropy against the true labels.
        brier: Brier score (mean squared error of the probabilities).
        mean_pred: Mean predicted positive-class probability.
        positive_rate: Observed positive rate of the evaluation split.
    """

    auc_roc: float
    auc_pr: float
    log_loss: float
    brier: float
    mean_pred: float
    positive_rate: float


class BenchmarkConfig(TypedDict, total=True):
    """Hyperparameters held identical across both models.

    Args:
        n_estimators: Boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Shrinkage applied to each tree's contribution.
        max_bins: Histogram bin count.
        min_data_in_leaf: Minimum samples required in a leaf.
        num_leaves: Leaf cap. Binds LightGBM's leaf-wise growth only;
            ClearGBM grows depth-wise and is bounded by ``max_depth``.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.
        n_jobs: Worker threads. One, so measurements are single-threaded.
        repeats: Timed fits per model per seed.
        warmups: Discarded fits before timing, which pull the data into cache
            and burn off the turbo window.
    """

    n_estimators: int
    max_depth: int
    learning_rate: float
    max_bins: int
    min_data_in_leaf: int
    num_leaves: int
    reg_alpha: float
    reg_lambda: float
    n_jobs: int
    repeats: int
    warmups: int


class DatasetInfo(TypedDict, total=True):
    """Identity of the benchmark input, so manifests are provably same-input.

    Args:
        sha256: SHA-256 of the source CSV.
        n_rows: Row count of the loaded frame.
        n_features: Feature-column count after dropping identifier columns.
    """

    sha256: str
    n_rows: int
    n_features: int


class SeedResult(TypedDict, total=True):
    """One model's outcome at one seed.

    Args:
        model: Which arm produced this record.
        seed: Split and model seed.
        position: Zero-based slot this arm occupied at this seed. The order
            rotates across seeds so no arm systematically occupies the
            cold-CPU slot. This replaced a ``ran_first`` boolean in schema 2:
            with three or more arms, "was it first" no longer describes where
            an arm ran, and averaging over an unrecorded position hides a
            systematic warm-up advantage rather than cancelling it.
        timing: Fit-time statistics.
        quality: Predictive-quality metrics.
        mean_leaves: Mean leaves per tree. The work-per-tree normalizer that
            makes a depth-wise model comparable to a leaf-wise one.
    """

    model: BenchmarkModelName
    seed: int
    position: int
    timing: TimingSummary
    quality: QualityMetrics
    mean_leaves: float


class BenchmarkManifest(TypedDict, total=True):
    """Complete machine-readable record of one benchmark invocation.

    Args:
        schema_version: Value of :data:`MANIFEST_SCHEMA_VERSION` at write time.
        estimator: Statistic used for each seed's canonical fit time.
        config: Hyperparameters shared by both models.
        dataset: Identity of the input data.
        seeds: Seeds measured, in execution order.
        results: Every per-model per-seed record.
        fingerprint: The configuration these numbers were produced under,
            from :func:`~covenant_ml.benchmarking.provenance.benchmark_fingerprint`.

            THIS IS THE MANIFEST THAT NEEDED IT MOST. Its headline is a
            TIMING claim against LightGBM, and of everything that moves a fit
            time the machine moves it most -- yet an inventory on 2026-08-27
            found 37 of this project's 41 published manifests carrying no
            environment at all, and no manifest ever carrying a CPU or a core
            count. A fit time from a 24-core cluster node and one from an
            8-core workstation were indistinguishable in the file.

            Taken as an argument rather than captured here. Building it reads
            installed metadata, and this module must remain importable in a
            process that has not pinned its thread count yet.
    """

    schema_version: int
    estimator: TimingEstimator
    config: BenchmarkConfig
    dataset: DatasetInfo
    seeds: list[int]
    results: list[SeedResult]
    fingerprint: RunFingerprint


__all__ = [
    "BENCHMARK_MODEL_NAMES",
    "ERR_DUPLICATE_TRAINER",
    "ERR_EMPTY_SPLIT",
    "ERR_INVALID_REPEATS",
    "ERR_LENGTH_MISMATCH",
    "ERR_MISSING_COLUMN",
    "ERR_NOT_BOOL",
    "ERR_NOT_FLOAT",
    "ERR_NOT_INT",
    "ERR_NOT_LIST",
    "ERR_NOT_MAPPING",
    "ERR_NOT_STR",
    "ERR_NO_RESULTS",
    "ERR_NO_SEEDS",
    "ERR_NO_TIMING_SAMPLES",
    "ERR_NO_TREES",
    "ERR_POWER_THROTTLING",
    "ERR_SCHEMA_VERSION",
    "ERR_TOO_FEW_TRAINERS",
    "ERR_UNKNOWN_ESTIMATOR",
    "ERR_UNKNOWN_MODEL",
    "MANIFEST_SCHEMA_VERSION",
    "BenchmarkConfig",
    "BenchmarkManifest",
    "BenchmarkModelName",
    "DatasetInfo",
    "QualityMetrics",
    "SeedResult",
    "TimingEstimator",
    "TimingSummary",
]
