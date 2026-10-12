"""What each benchmark family IS, as data.

Every ``benchmark_cleargbm_*`` entry point is one of these plus a config
constructor. The flags, the experiment name, the arm axes and the metric names
all live here; :mod:`covenant_ml.benchmarking.harness` reads them and owns
everything else. Nothing in this module runs a benchmark or writes a file.

THE EXPERIMENT NAMES ARE DELIBERATELY DISTINCT. They are the field
:mod:`platform_core.comparability` reads to decide whether two records may be
subtracted. Before 2026-09-09 there was one constant --
``"cleargbm-vs-lightgbm-fit-time"`` -- and the tempting repair for the five
families that emitted no record at all was to route them through it. That
would have made a ranking run and a regression run claim one experiment, and
the layer whose entire job is refusing incomparable subtractions would then
have licensed subtracting a mean NDCG from an R-squared. Five families
emitting nothing was an honest absence; five claiming a shared experiment
would have been a defect that reads as provenance.

THE ARM AXES ARE NOT ALWAYS THE MODEL. ``goss`` measures each model under two
sampling modes and ``quantized`` under two histogram modes, so an observation
named for the model alone would pair a model's GOSS number with its own
full-sampling number in any contrast that read two records side by side.
"""

from __future__ import annotations

from covenant_ml.benchmarking.harness import (
    ArgumentKind,
    BenchmarkArgument,
    BenchmarkDeclaration,
)

#: Flags shared by every family that generates a synthetic corpus.
#:
#: Spelled out per declaration rather than spliced in by a helper: a family
#: that silently inherited a flag it does not honour is the defect class this
#: package's own history is full of, and a reader should be able to see a
#: family's whole argument surface in one place.
_TREES = BenchmarkArgument(flag="--trees", kind=ArgumentKind.INT, help="Boosting rounds per arm.")
_MAX_DEPTH = BenchmarkArgument(
    flag="--max-depth", kind=ArgumentKind.INT, help="Maximum tree depth."
)
_LEARNING_RATE = BenchmarkArgument(
    flag="--learning-rate", kind=ArgumentKind.FLOAT, help="Shrinkage per round."
)
_MAX_BINS = BenchmarkArgument(
    flag="--max-bins", kind=ArgumentKind.INT, help="Histogram bins per feature."
)
_MIN_SAMPLES_LEAF = BenchmarkArgument(
    flag="--min-samples-leaf", kind=ArgumentKind.INT, help="Minimum samples per leaf."
)
_SAMPLES = BenchmarkArgument(
    flag="--samples", kind=ArgumentKind.INT, help="Rows in the synthetic corpus."
)
_FEATURES = BenchmarkArgument(
    flag="--features", kind=ArgumentKind.INT, help="Features in the synthetic corpus."
)
_CSV = BenchmarkArgument(flag="--csv", kind=ArgumentKind.PATH, help="Bankruptcy corpus CSV.")
_NUM_LEAVES = BenchmarkArgument(
    flag="--num-leaves", kind=ArgumentKind.INT, help="Leaf budget for leaf-wise growth."
)


GOSS = BenchmarkDeclaration(
    experiment="cleargbm-vs-lightgbm-goss-quality",
    description="Measure GOSS quality cost for ClearGBM and LightGBM side by side.",
    arguments=(
        _SAMPLES,
        _FEATURES,
        _TREES,
        _MAX_DEPTH,
        _LEARNING_RATE,
        _MAX_BINS,
        _MIN_SAMPLES_LEAF,
        BenchmarkArgument(
            flag="--top-rate", kind=ArgumentKind.FLOAT, help="GOSS large-gradient retain rate."
        ),
        BenchmarkArgument(
            flag="--other-rate", kind=ArgumentKind.FLOAT, help="GOSS small-gradient sample rate."
        ),
    ),
    arm_axes=("model", "sampling"),
    quality_metrics=("auc", "log_loss"),
    result_metrics=(),
)

QUANTIZED = BenchmarkDeclaration(
    experiment="cleargbm-vs-lightgbm-quantized-quality",
    description="Measure quantized-histogram quality and fit time for both engines.",
    arguments=(
        _SAMPLES,
        _FEATURES,
        _TREES,
        _MAX_DEPTH,
        _LEARNING_RATE,
        _MAX_BINS,
        _MIN_SAMPLES_LEAF,
        BenchmarkArgument(
            flag="--quant-bins", kind=ArgumentKind.INT, help="Gradient quantization bins."
        ),
    ),
    arm_axes=("model", "histogram"),
    quality_metrics=("auc", "log_loss"),
    result_metrics=("fit_seconds",),
)

MULTICLASS = BenchmarkDeclaration(
    experiment="cleargbm-vs-lightgbm-multiclass-quality",
    description="Measure multiclass quality for ClearGBM and LightGBM side by side.",
    arguments=(
        _SAMPLES,
        _FEATURES,
        BenchmarkArgument(
            flag="--classes", kind=ArgumentKind.INT, help="Classes in the synthetic corpus."
        ),
        _TREES,
        _MAX_DEPTH,
        _LEARNING_RATE,
        _MAX_BINS,
        _MIN_SAMPLES_LEAF,
    ),
    arm_axes=("model",),
    quality_metrics=("log_loss", "accuracy"),
    result_metrics=(),
)

RANKING = BenchmarkDeclaration(
    experiment="cleargbm-vs-lightgbm-ranking-quality",
    description="Measure LambdaRank NDCG for ClearGBM and LightGBM side by side.",
    arguments=(
        BenchmarkArgument(
            flag="--queries", kind=ArgumentKind.INT, help="Queries in the synthetic corpus."
        ),
        BenchmarkArgument(flag="--docs", kind=ArgumentKind.INT, help="Documents per query."),
        _FEATURES,
        _TREES,
        _MAX_DEPTH,
        _LEARNING_RATE,
        _MAX_BINS,
        _MIN_SAMPLES_LEAF,
        BenchmarkArgument(
            flag="--truncation", kind=ArgumentKind.INT, help="NDCG truncation level."
        ),
    ),
    arm_axes=("model",),
    quality_metrics=("mean_ndcg",),
    result_metrics=(),
)

REGRESSION = BenchmarkDeclaration(
    experiment="cleargbm-vs-lightgbm-regression-quality",
    description="Measure regression quality and fit time on an external corpus.",
    arguments=(
        BenchmarkArgument(
            flag="--dataset", kind=ArgumentKind.STRING, help="External dataset name."
        ),
        BenchmarkArgument(
            flag="--external-dir", kind=ArgumentKind.PATH, help="Directory holding it."
        ),
        _TREES,
        _MAX_DEPTH,
        _NUM_LEAVES,
        _LEARNING_RATE,
        _MAX_BINS,
        _MIN_SAMPLES_LEAF,
        BenchmarkArgument(
            flag="--early-stopping", kind=ArgumentKind.INT, help="Early-stopping rounds."
        ),
    ),
    arm_axes=("model",),
    quality_metrics=("rmse", "mae", "r_squared"),
    result_metrics=("fit_seconds",),
)


VS_LIGHTGBM = BenchmarkDeclaration(
    experiment="cleargbm-vs-lightgbm-fit-time",
    description="Benchmark ClearGBM fit time against LightGBM on a real corpus.",
    arguments=(
        _CSV,
        _TREES,
        _MAX_DEPTH,
        _MAX_BINS,
        _NUM_LEAVES,
        BenchmarkArgument(
            flag="--repeats", kind=ArgumentKind.INT, help="Timed repeats per arm per seed."
        ),
        BenchmarkArgument(
            flag="--warmups", kind=ArgumentKind.INT, help="Untimed warmup fits before timing."
        ),
        BenchmarkArgument(
            flag="--variants", kind=ArgumentKind.FLAG, help="Include the growth-policy variants."
        ),
    ),
    # DELIBERATELY EMPTY, and this is the honest way to say what this family is.
    # The other five report per-arm means of per-seed quality. This one's
    # headline is a fit-time RATIO BETWEEN ARMS -- raw, per-leaf and normalized
    # -- which is not the mean of anything a result row carries, so there is no
    # arm axis or metric name that would produce it. Its observations are built
    # by `benchmark_observations` and handed to the harness as `derived`.
    # Declaring metrics it does not have would make the declaration lie in
    # order to look uniform.
    arm_axes=(),
    quality_metrics=(),
    result_metrics=(),
)


#: Flags shared by the two paired-evaluation families: how every interval
#: treats dependence between held-out rows.
_UNIT = BenchmarkArgument(
    flag="--unit",
    kind=ArgumentKind.STRING,
    help="Independent unit of the held-out rows: row, cluster (company) or block.",
)
_BLOCK_LENGTH = BenchmarkArgument(
    flag="--block-length", kind=ArgumentKind.INT, help="Rows per block for --unit block, else 1."
)
_N_RESAMPLES = BenchmarkArgument(
    flag="--n-resamples", kind=ArgumentKind.INT, help="Bootstrap resamples off the row unit."
)
_RESAMPLE_SEED = BenchmarkArgument(
    flag="--resample-seed", kind=ArgumentKind.INT, help="Seed of the bootstrap draws."
)
_CONFIDENCE = BenchmarkArgument(
    flag="--confidence", kind=ArgumentKind.FLOAT, help="Coverage of every interval, e.g. 0.95."
)

PAIRED_EVALUATION = BenchmarkDeclaration(
    experiment="covenant-paired-evaluation",
    description=(
        "Compare every arm with a reference on the same held-out rows: AUC, log loss, "
        "Brier and the value of its alerting decisions, with intervals per seed and pooled."
    ),
    arguments=(
        _CSV,
        _TREES,
        _MAX_DEPTH,
        _MAX_BINS,
        _NUM_LEAVES,
        BenchmarkArgument(
            flag="--reference", kind=ArgumentKind.STRING, help="Arm every other arm is compared to."
        ),
        _UNIT,
        _BLOCK_LENGTH,
        _N_RESAMPLES,
        _RESAMPLE_SEED,
        _CONFIDENCE,
        BenchmarkArgument(
            flag="--true-alert-value",
            kind=ArgumentKind.FLOAT,
            help="Gain from alerting on a row that turns positive.",
        ),
        BenchmarkArgument(
            flag="--false-alert-cost",
            kind=ArgumentKind.FLOAT,
            help="Cost of alerting on a row that stays negative.",
        ),
        BenchmarkArgument(
            flag="--missed-cost",
            kind=ArgumentKind.FLOAT,
            help="Cost of not alerting on a row that turns positive.",
        ),
    ),
    arm_axes=("arm",),
    quality_metrics=("auc_roc", "log_loss", "brier"),
    result_metrics=(),
)

FEATURE_ABLATION = BenchmarkDeclaration(
    experiment="covenant-feature-ablation",
    description=(
        "Retrain one learner without each feature group and compare it with the full model "
        "on the same held-out rows; trace the full model's learning curve."
    ),
    arguments=(
        _CSV,
        _TREES,
        _MAX_DEPTH,
        _MAX_BINS,
        _NUM_LEAVES,
        BenchmarkArgument(flag="--model", kind=ArgumentKind.STRING, help="Learner to ablate."),
        BenchmarkArgument(
            flag="--groups",
            kind=ArgumentKind.STRING,
            help="Groups to remove: 'X1;liquidity=X3,X4' (';' between groups).",
        ),
        BenchmarkArgument(
            flag="--fractions",
            kind=ArgumentKind.STRING,
            help="Learning-curve training shares, ascending: '0.1,0.25,0.5,1.0'.",
        ),
        _UNIT,
        _BLOCK_LENGTH,
        _N_RESAMPLES,
        _RESAMPLE_SEED,
        _CONFIDENCE,
    ),
    arm_axes=("arm",),
    quality_metrics=("auc_roc", "log_loss", "brier"),
    result_metrics=(),
)


__all__ = [
    "FEATURE_ABLATION",
    "GOSS",
    "MULTICLASS",
    "PAIRED_EVALUATION",
    "QUANTIZED",
    "RANKING",
    "REGRESSION",
    "VS_LIGHTGBM",
]
