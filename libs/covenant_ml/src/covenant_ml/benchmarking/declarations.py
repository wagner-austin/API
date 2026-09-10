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
        BenchmarkArgument(
            flag="--num-leaves", kind=ArgumentKind.INT, help="Leaf budget for leaf-wise growth."
        ),
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
        BenchmarkArgument(flag="--csv", kind=ArgumentKind.PATH, help="Bankruptcy corpus CSV."),
        _TREES,
        _MAX_DEPTH,
        _MAX_BINS,
        BenchmarkArgument(
            flag="--num-leaves", kind=ArgumentKind.INT, help="Leaf budget for leaf-wise growth."
        ),
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


__all__ = [
    "GOSS",
    "MULTICLASS",
    "QUANTIZED",
    "RANKING",
    "REGRESSION",
    "VS_LIGHTGBM",
]
