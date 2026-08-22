"""Reduction of a benchmark manifest to a human-readable report.

Pure functions over a finished manifest, so the report's arithmetic is
testable without running a benchmark.

The leaf-normalized ratio is the headline number. A raw wall-clock ratio
between a depth-wise and a leaf-wise learner conflates two different things:
being slower per unit of work, and doing more work per tree. Dividing by the
ratio of mean leaves separates them.
"""

from __future__ import annotations

import statistics
from typing import NamedTuple

from .types import ERR_NO_RESULTS, BenchmarkManifest, BenchmarkModelName, SeedResult


class ModelSummary(NamedTuple):
    """Aggregate of one model's records across seeds.

    Args:
        model: Which model this summarises.
        mean_fit_s: Mean of the per-seed canonical fit times.
        stdev_fit_s: Sample standard deviation of those times, or 0.0 when a
            single seed was measured.
        mean_leaves: Mean leaves per tree across seeds.
        mean_auc_roc: Mean ROC-AUC across seeds.
        mean_auc_pr: Mean average precision across seeds.
    """

    model: BenchmarkModelName
    mean_fit_s: float
    stdev_fit_s: float
    mean_leaves: float
    mean_auc_roc: float
    mean_auc_pr: float


class GapSummary(NamedTuple):
    """Comparison of the two models.

    Args:
        cleargbm: ClearGBM's aggregate.
        lightgbm: LightGBM's aggregate.
        raw_ratio: ClearGBM's mean fit time divided by LightGBM's.
        leaf_ratio: ClearGBM's mean leaves divided by LightGBM's.
        normalized_ratio: ``raw_ratio`` divided by ``leaf_ratio`` -- the
            per-leaf cost comparison, which is what "is it slower" actually
            means once tree size is held constant.
    """

    cleargbm: ModelSummary
    lightgbm: ModelSummary
    raw_ratio: float
    leaf_ratio: float
    normalized_ratio: float


def _stdev_or_zero(values: list[float]) -> float:
    """Sample standard deviation, defined as zero for a single observation.

    Args:
        values: Observations.

    Returns:
        Sample standard deviation, or 0.0 when fewer than two observations.
    """
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def summarize_model(results: list[SeedResult], model: BenchmarkModelName) -> ModelSummary:
    """Aggregate one model's per-seed records.

    Args:
        results: Every record in the manifest.
        model: Which model to aggregate.

    Returns:
        The model's aggregate across seeds.

    Raises:
        ValueError: If the manifest holds no record for this model.
    """
    selected = [result for result in results if result["model"] == model]
    if len(selected) == 0:
        raise ValueError(f"[{ERR_NO_RESULTS}] Manifest holds no results for model '{model}'")

    fit_times = [result["timing"]["canonical_s"] for result in selected]
    return ModelSummary(
        model=model,
        mean_fit_s=statistics.fmean(fit_times),
        stdev_fit_s=_stdev_or_zero(fit_times),
        mean_leaves=statistics.fmean([result["mean_leaves"] for result in selected]),
        mean_auc_roc=statistics.fmean([result["quality"]["auc_roc"] for result in selected]),
        mean_auc_pr=statistics.fmean([result["quality"]["auc_pr"] for result in selected]),
    )


def summarize_gap(manifest: BenchmarkManifest) -> GapSummary:
    """Compare the two models across every seed in a manifest.

    Args:
        manifest: A finished benchmark manifest.

    Returns:
        Both aggregates plus the raw, leaf and normalized ratios.

    Raises:
        ValueError: If the manifest holds no record for either model.
    """
    cleargbm = summarize_model(manifest["results"], "cleargbm")
    lightgbm = summarize_model(manifest["results"], "lightgbm")
    raw_ratio = cleargbm.mean_fit_s / lightgbm.mean_fit_s
    leaf_ratio = cleargbm.mean_leaves / lightgbm.mean_leaves
    return GapSummary(
        cleargbm=cleargbm,
        lightgbm=lightgbm,
        raw_ratio=raw_ratio,
        leaf_ratio=leaf_ratio,
        normalized_ratio=raw_ratio / leaf_ratio,
    )


def summarize_every_model(manifest: BenchmarkManifest) -> list[ModelSummary]:
    """Aggregate every arm the manifest actually contains.

    Driven off the records rather than a fixed pair, so an arm added to a run
    cannot appear in the per-seed detail and then vanish from the summary --
    which would read as though it had not been measured.

    Args:
        manifest: A finished benchmark manifest.

    Returns:
        One aggregate per arm, in first-appearance order.

    Raises:
        ValueError: If the manifest holds no records at all.
    """
    results = manifest["results"]
    if len(results) == 0:
        raise ValueError(f"[{ERR_NO_RESULTS}] Manifest holds no results")

    ordered: list[BenchmarkModelName] = []
    for result in results:
        if result["model"] not in ordered:
            ordered.append(result["model"])
    return [summarize_model(results, model) for model in ordered]


def render_seed_line(result: SeedResult) -> str:
    """Render one per-model per-seed record.

    The full min/median/mean/max spread is always shown, so a reader can see
    whether a difference between two runs is resolvable or is noise.

    Args:
        result: The record to render.

    Returns:
        A single line without a trailing newline.
    """
    timing = result["timing"]
    return (
        f"  {result['model']:<9} seed={result['seed']} "
        f"fit={timing['canonical_s']:.4f}s "
        f"(over {len(timing['samples_s'])}: "
        f"{timing['min_s']:.4f}/{timing['median_s']:.4f}/"
        f"{timing['mean_s']:.4f}/{timing['max_s']:.4f}) "
        f"leaves={result['mean_leaves']:.2f}"
    )


def render_report(manifest: BenchmarkManifest) -> str:
    """Render a finished manifest as a report.

    Args:
        manifest: A finished benchmark manifest.

    Returns:
        The report, newline-terminated.

    Raises:
        ValueError: If the manifest holds no record for either model.
    """
    gap = summarize_gap(manifest)
    config = manifest["config"]
    lines: list[str] = [
        f"dataset  rows={manifest['dataset']['n_rows']} "
        f"features={manifest['dataset']['n_features']} "
        f"sha256={manifest['dataset']['sha256'][:16]}",
        f"config   trees={config['n_estimators']} depth={config['max_depth']} "
        f"bins={config['max_bins']} num_leaves={config['num_leaves']} "
        f"repeats={config['repeats']} warmups={config['warmups']}",
        f"estimator {manifest['estimator']} of repeats, per seed",
        "",
    ]
    lines.extend(render_seed_line(result) for result in manifest["results"])
    lines.append("")

    # Every arm in the manifest, not just the two the ratios compare.
    for summary in summarize_every_model(manifest):
        lines.append(
            f"{summary.model:<18} fit={summary.mean_fit_s:.4f}s "
            f"+/- {summary.stdev_fit_s:.4f}s  "
            f"leaves={summary.mean_leaves:.2f}  "
            f"auc_roc={summary.mean_auc_roc:.4f}  auc_pr={summary.mean_auc_pr:.4f}"
        )

    lines.extend(
        [
            "",
            f"raw ratio         {gap.raw_ratio:.3f}x  (cleargbm / lightgbm wall clock)",
            f"leaf ratio        {gap.leaf_ratio:.3f}x  (cleargbm / lightgbm tree size)",
            f"per-leaf ratio    {gap.normalized_ratio:.3f}x  (cost at equal tree size)",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "GapSummary",
    "ModelSummary",
    "render_report",
    "render_seed_line",
    "summarize_every_model",
    "summarize_gap",
    "summarize_model",
]
