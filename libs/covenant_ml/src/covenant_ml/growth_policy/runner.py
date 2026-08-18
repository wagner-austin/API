"""The measurement protocol: one arm, one seed, and the loop over both.

The protocol is stated here rather than written out per script, because an
experiment left unstated is an experiment performed slightly differently every
time it is run. Three properties are deliberate:

* **Every arm is measured in the same run.** A figure carried forward from an
  earlier session cannot separate a code change from a machine-state change.
* **The canonical fit time is the median of the timed repeats**, taken through
  :func:`covenant_ml.benchmarking.timing.summarize_timings` rather than a
  second implementation. The first fits after an idle period run with full
  turbo headroom -- a different power regime rather than noise -- so a minimum
  reports a cold-start outlier in place of the steady state.
* **Quality is scored from the last timed fit**, not from an extra untimed one,
  so the model reported on is a model that was actually measured.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..benchmarking.timing import summarize_timings
from . import _test_hooks
from .protocols import (
    ArmTrainerProto,
    MetricsProto,
    SplitFactoryProto,
    TrainedModelProto,
    TwoWaySplit,
)
from .summarize import summarize_arms
from .types import (
    ERR_INVALID_REPEATS,
    ERR_NO_ARMS,
    ERR_NO_SEEDS,
    REPORT_SCHEMA_VERSION,
    ArmResult,
    DatasetInfo,
    ExperimentConfig,
    GrowthPolicyReport,
)


def fit_repeatedly(
    trainer: ArmTrainerProto,
    split: TwoWaySplit,
    seed: int,
    repeats: int,
    warmups: int,
) -> tuple[TrainedModelProto, float]:
    """Fit one arm repeatedly and report the canonical fit time.

    Warmup fits run first and are discarded. Each timed fit is bracketed by
    the clock hook, so tests drive this from a fixed sequence rather than from
    real elapsed time.

    Args:
        trainer: The arm to fit.
        split: The partition to fit on.
        seed: Seed for the model's internal randomness.
        repeats: Timed fits to perform. Must be at least one.
        warmups: Discarded fits before timing begins.

    Returns:
        The model from the last timed fit, and the median timed duration in
        seconds.

    Raises:
        ValueError: If ``repeats`` is less than one, which would leave no
            timing to summarise and no model to score.
    """
    if repeats < 1:
        raise ValueError(f"[{ERR_INVALID_REPEATS}] repeats must be at least 1, got {repeats}")
    for _ in range(warmups):
        trainer.fit(split, seed)
    samples: list[float] = []
    fitted: list[TrainedModelProto] = []
    for _ in range(repeats):
        started = _test_hooks.monotonic_clock()
        model = trainer.fit(split, seed)
        samples.append(_test_hooks.monotonic_clock() - started)
        fitted.append(model)
    summary = summarize_timings(samples)
    return fitted[-1], summary["canonical_s"]


def measure_arm(
    trainer: ArmTrainerProto,
    split: TwoWaySplit,
    seed: int,
    metrics: MetricsProto,
    config: ExperimentConfig,
) -> ArmResult:
    """Measure one arm at one seed.

    Args:
        trainer: The arm to measure.
        split: The partition to fit and score on.
        seed: Seed for both the partition and the model's randomness.
        metrics: Scorer for the held-out fold.
        config: Hyperparameters shared across arms, read for repeat counts.

    Returns:
        The arm's result at this seed.
    """
    model, fit_seconds = fit_repeatedly(trainer, split, seed, config["repeats"], config["warmups"])
    positive_proba = model.predict_positive_proba(split.x_test)
    return {
        "arm": trainer.arm_name,
        "seed": seed,
        "fit_seconds": fit_seconds,
        "auc_roc": metrics.auc_roc(split.y_test, positive_proba),
        "auc_pr": metrics.auc_pr(split.y_test, positive_proba),
        "log_loss": metrics.log_loss(split.y_test, positive_proba),
        "mean_leaves": model.mean_leaves(),
    }


def run_experiment(
    trainers: Sequence[ArmTrainerProto],
    split_factory: SplitFactoryProto,
    seeds: Sequence[int],
    metrics: MetricsProto,
    config: ExperimentConfig,
    dataset: DatasetInfo,
) -> GrowthPolicyReport:
    """Measure every arm at every seed and assemble the report.

    Every arm at one seed sees the identical partition, which is what makes the
    contrast between arms attributable to the arm rather than to the split.

    Args:
        trainers: The arms to measure, in report order.
        split_factory: Builds the partition for a seed.
        seeds: Seeds to measure at, in execution order.
        metrics: Scorer for the held-out fold.
        config: Hyperparameters shared across arms.
        dataset: Description of the dataset being measured.

    Returns:
        The complete report, with per-seed results and per-arm summaries.

    Raises:
        ValueError: If no arms or no seeds were supplied, either of which
            produces a report that states nothing.
    """
    if len(trainers) == 0:
        raise ValueError(f"[{ERR_NO_ARMS}] At least one arm is required")
    if len(seeds) == 0:
        raise ValueError(f"[{ERR_NO_SEEDS}] At least one seed is required")
    results: list[ArmResult] = []
    for seed in seeds:
        split = split_factory(seed)
        for trainer in trainers:
            results.append(measure_arm(trainer, split, seed, metrics, config))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "config": config,
        "dataset": dataset,
        "seeds": list(seeds),
        "results": results,
        "summaries": summarize_arms(results),
    }


__all__ = ["fit_repeatedly", "measure_arm", "run_experiment"]
