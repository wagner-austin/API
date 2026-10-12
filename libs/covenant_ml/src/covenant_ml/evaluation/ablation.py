"""What a feature group contributes, measured by taking it away.

Feature importance (SHAP, permutation) says how much a FITTED model leans on
a feature. An ablation asks a different question: retrained without the
feature, is the model any worse? A feature can carry high importance and no
ablation effect when another feature carries the same information; the
model simply leans on the substitute. The ablation is the one that answers
"do we need to collect this".

Each seed fits the full model and one model per group on the same partition,
and compares each reduced model with the full one on the same held-out rows
(arm ``without:<group>``, reference ``full``). A difference near zero with a
narrow interval is a measured "not needed", which a single score cannot say.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from platform_core.comparability import RunFingerprint

from ..benchmarking.protocols import DataSplit, TrainerProto
from ..benchmarking.quality import compute_quality
from ..benchmarking.types import BenchmarkConfig, DatasetInfo
from .learning_curve import learning_curve_seed, require_fractions
from .paired import compare_probabilities
from .pooling import pool_seed_comparisons
from .resampling import build_resampling
from .runner import require_seeds
from .splits import GroupedSplit, GroupedSplitFactoryProto
from .types import (
    ERR_EMPTY_ABLATION,
    ERR_UNKNOWN_FEATURE,
    EVALUATION_SCHEMA_VERSION,
    AblationGroup,
    AblationManifest,
    ArmResult,
    LearningCurvePoint,
    PairedComparison,
    ResamplingSettings,
    SeedComparisons,
)

#: Arm name of the model fitted on every feature.
FULL_ARM = "full"


def ablated_arm(group: AblationGroup) -> str:
    """Name the arm fitted without a group.

    Args:
        group: The ablated group.

    Returns:
        ``"without:<name>"``.
    """
    return f"without:{group['name']}"


def keep_mask(feature_names: Sequence[str], group: AblationGroup) -> NDArray[np.bool_]:
    """Mark the feature columns that survive removing a group.

    Args:
        feature_names: Every feature column's name, in column order.
        group: The features to remove.

    Returns:
        True for each column kept.

    Raises:
        ValueError: If the group names a feature that does not exist, names
            none, or would remove every feature.
    """
    unknown = sorted(set(group["features"]) - set(feature_names))
    if unknown:
        raise ValueError(
            f"[{ERR_UNKNOWN_FEATURE}] group '{group['name']}' names unknown features {unknown}"
        )
    removed = set(group["features"])
    kept: list[bool] = [name not in removed for name in feature_names]
    n_kept = sum(kept)
    if not removed or n_kept == 0:
        raise ValueError(
            f"[{ERR_EMPTY_ABLATION}] group '{group['name']}' must remove at least one feature "
            f"and keep at least one; it removes {len(removed)} of {len(feature_names)}"
        )
    mask: NDArray[np.bool_] = np.asarray(kept, dtype=np.bool_)
    return mask


def drop_columns(split: DataSplit, keep: NDArray[np.bool_]) -> DataSplit:
    """The same partition with only the kept feature columns.

    Args:
        split: The full partition.
        keep: True for each column kept.

    Returns:
        The reduced partition; rows and labels are unchanged.
    """
    return DataSplit(
        x_train=split.x_train[:, keep],
        y_train=split.y_train,
        x_val=split.x_val[:, keep],
        y_val=split.y_val,
        x_test=split.x_test[:, keep],
        y_test=split.y_test,
    )


def ablate_seed(
    trainer: TrainerProto,
    grouped: GroupedSplit,
    seed: int,
    feature_names: Sequence[str],
    groups: Sequence[AblationGroup],
    settings: ResamplingSettings,
) -> tuple[list[ArmResult], SeedComparisons]:
    """Fit the full model and each reduced model on one partition, and compare.

    Args:
        trainer: The learner ablated.
        grouped: The partition and its held-out unit codes.
        seed: Model seed; also offsets the bootstrap seed.
        feature_names: Every feature column's name, in column order.
        groups: The groups to remove, one model each.
        settings: How intervals treat dependence.

    Returns:
        Every arm's quality, and each reduced model's comparison with the
        full one.

    Raises:
        ValueError: If a group is invalid, or the held-out fold cannot
            support an interval.
    """
    masks = [keep_mask(feature_names, group) for group in groups]
    split = grouped.split
    resampling = build_resampling(
        settings["unit"], len(split.y_test), grouped.test_groups, settings["block_length"]
    )
    full = trainer.fit(split, seed).predict_positive_proba(split.x_test)
    results: list[ArmResult] = [
        {"arm": FULL_ARM, "seed": seed, "quality": compute_quality(split.y_test, full)}
    ]
    comparisons: list[PairedComparison] = []
    for group, keep in zip(groups, masks, strict=True):
        reduced_split = drop_columns(split, keep)
        reduced = trainer.fit(reduced_split, seed).predict_positive_proba(reduced_split.x_test)
        arm = ablated_arm(group)
        results.append(
            {"arm": arm, "seed": seed, "quality": compute_quality(split.y_test, reduced)}
        )
        comparisons.extend(
            compare_probabilities(
                split.y_test,
                reduced,
                full,
                arm,
                FULL_ARM,
                resampling,
                settings["n_resamples"],
                settings["resample_seed"] + seed,
                settings["confidence"],
            )
        )
    return results, {
        "seed": seed,
        "n_train": len(split.y_train),
        "n_test": len(split.y_test),
        "n_test_units": resampling.n_units,
        "comparisons": comparisons,
        "policies": [],
    }


def run_feature_ablation(
    trainer: TrainerProto,
    build_split: GroupedSplitFactoryProto,
    seeds: Sequence[int],
    feature_names: Sequence[str],
    groups: Sequence[AblationGroup],
    fractions: Sequence[float],
    config: BenchmarkConfig,
    dataset: DatasetInfo,
    settings: ResamplingSettings,
    fingerprint: RunFingerprint,
) -> AblationManifest:
    """Ablate every group and trace the full model's learning curve, per seed.

    Args:
        trainer: The learner ablated.
        build_split: Produces the grouped partition for a seed.
        seeds: Seeds to evaluate, at least two.
        feature_names: Every feature column's name, in column order.
        groups: The groups to remove, distinctly named.
        fractions: Training shares for the learning curve.
        config: Hyperparameters the learner was built with.
        dataset: Identity of the input data.
        settings: How intervals treat dependence.
        fingerprint: The configuration this runs under.

    Returns:
        The complete manifest.

    Raises:
        ValueError: If the groups, seeds or fractions are invalid, or a
            held-out fold cannot support an interval.
    """
    names = [group["name"] for group in groups]
    if not names or len(set(names)) != len(names):
        raise ValueError(
            f"[{ERR_EMPTY_ABLATION}] at least one distinctly named group is needed, got {names}"
        )
    require_seeds(seeds)
    require_fractions(fractions)
    results: list[ArmResult] = []
    seed_comparisons: list[SeedComparisons] = []
    curve: list[LearningCurvePoint] = []
    for seed in seeds:
        grouped = build_split(seed)
        seed_results, comparisons = ablate_seed(
            trainer, grouped, seed, feature_names, groups, settings
        )
        results.extend(seed_results)
        seed_comparisons.append(comparisons)
        curve.extend(learning_curve_seed(FULL_ARM, trainer, grouped.split, seed, fractions))
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "dataset": dataset,
        "config": config,
        "model": trainer.model_name.value,
        "resampling": settings,
        "groups": list(groups),
        "seeds": list(seeds),
        "results": results,
        "seed_comparisons": seed_comparisons,
        "pooled": pool_seed_comparisons(seed_comparisons, settings["confidence"]),
        "learning_curve": curve,
        "fingerprint": fingerprint,
    }


__all__ = [
    "FULL_ARM",
    "ablate_seed",
    "ablated_arm",
    "drop_columns",
    "keep_mask",
    "run_feature_ablation",
]
