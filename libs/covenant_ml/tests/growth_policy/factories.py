"""Shared builders for the growth-policy tests.

Every helper here produces real data or a real record. Nothing in this module
stands in for the code under test: the datasets are genuinely small rather than
fake, and the records are the shapes the codecs actually accept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.growth_policy.datasets import BANKRUPTCY_FEATURE_COUNT
from covenant_ml.growth_policy.types import (
    REPORT_SCHEMA_VERSION,
    ArmResult,
    DatasetInfo,
    ExperimentConfig,
    GrowthPolicyReport,
)

from .numeric import as_float_list

#: Rows per company in the synthetic bankruptcy CSV, mirroring the real
#: dataset's company-year shape so a group-disjoint split has something to do.
ROWS_PER_COMPANY = 4


def make_separable_dataset(
    row_count: int = 120,
    feature_count: int = 4,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Build a small, genuinely learnable binary dataset.

    The label depends on the first feature, so a fitted model scores above
    chance and a quality metric has a real signal to report rather than noise.

    Args:
        row_count: Number of rows.
        feature_count: Number of feature columns.
        seed: Seed for the generator.

    Returns:
        Features and labels.
    """
    rng = np.random.default_rng(seed)
    features: NDArray[np.float64] = rng.normal(size=(row_count, feature_count))
    logits: NDArray[np.float64] = features[:, 0] * 2.0
    labels: NDArray[np.int64] = (logits > 0.0).astype(np.int64)
    return features, labels


def write_bankruptcy_csv(
    path: Path,
    company_count: int = 12,
    seed: int = 0,
) -> None:
    """Write a synthetic CSV in the American-bankruptcy layout.

    Args:
        path: File to write.
        company_count: Distinct companies, each contributing several rows.
        seed: Seed for the generator.
    """
    rng = np.random.default_rng(seed)
    header = ["company_name", "status_label"]
    header.extend(f"X{index}" for index in range(1, BANKRUPTCY_FEATURE_COUNT + 1))
    lines = [",".join(header)]
    for company in range(company_count):
        status = "alive" if company % 2 == 0 else "failed"
        for _ in range(ROWS_PER_COMPANY):
            draw: NDArray[np.float64] = rng.normal(size=BANKRUPTCY_FEATURE_COUNT)
            values = [f"{value:.6f}" for value in as_float_list(draw)]
            lines.append(",".join([f"C{company:03d}", status, *values]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_taiwan_csv(path: Path, row_count: int = 40, feature_count: int = 5) -> None:
    """Write a synthetic CSV in the Taiwan-bankruptcy layout.

    Args:
        path: File to write.
        row_count: Number of data rows.
        feature_count: Number of feature columns after the label column.
    """
    header = ["Bankrupt?"] + [f"f{index}" for index in range(feature_count)]
    lines = [",".join(header)]
    for row in range(row_count):
        label = row % 2
        values = [f"{float(row + column):.3f}" for column in range(feature_count)]
        lines.append(",".join([str(label), *values]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_german_data(path: Path, row_count: int = 40) -> None:
    """Write a synthetic file in the German-credit layout.

    Mixes a categorical first column with numeric columns, so the loader's
    ordinal encoding and its numeric passthrough are both exercised.

    Args:
        path: File to write.
        row_count: Number of rows.
    """
    lines: list[str] = []
    for row in range(row_count):
        category = f"A{11 + (row % 3)}"
        numeric_one = row % 7
        numeric_two = row * 2
        label = "2" if row % 2 == 0 else "1"
        lines.append(f"{category} {numeric_one} {numeric_two} {label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_config(repeats: int = 1, warmups: int = 0) -> ExperimentConfig:
    """Build a configuration small enough for a test to fit quickly.

    Args:
        repeats: Timed fits per arm per seed.
        warmups: Discarded fits before timing begins.

    Returns:
        The configuration.
    """
    return {
        "n_estimators": 3,
        "learning_rate": 0.3,
        "max_bins": 16,
        "min_leaf": 2,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
        "repeats": repeats,
        "warmups": warmups,
    }


def make_arm_result(arm: str = "arm-a", seed: int = 42, scale: float = 1.0) -> ArmResult:
    """Build one arm-seed result.

    Args:
        arm: Arm name.
        seed: Seed the result was measured at.
        scale: Multiplier applied to every numeric field, so two results can
            differ in a way an averaging test can predict exactly.

    Returns:
        The result record.
    """
    return {
        "arm": arm,
        "seed": seed,
        "fit_seconds": 1.0 * scale,
        "auc_roc": 0.5 * scale,
        "auc_pr": 0.25 * scale,
        "log_loss": 0.125 * scale,
        "mean_leaves": 4.0 * scale,
    }


def make_dataset_info(name: str = "synthetic") -> DatasetInfo:
    """Build a dataset description.

    Args:
        name: Dataset name.

    Returns:
        The description record.
    """
    return {
        "name": name,
        "row_count": 100,
        "feature_count": 4,
        "positive_rate": 0.25,
    }


def make_report(results: list[ArmResult], summaries_source: list[ArmResult]) -> GrowthPolicyReport:
    """Build a report from explicit results.

    Args:
        results: Per-seed results to record.
        summaries_source: Results to summarise, so a test can control both
            halves independently.

    Returns:
        The report record.

    Raises:
        ValueError: If the summariser rejects ``summaries_source``.
    """
    from covenant_ml.growth_policy.summarize import summarize_arms

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "config": make_config(),
        "dataset": make_dataset_info(),
        "seeds": [42],
        "results": results,
        "summaries": summarize_arms(summaries_source),
    }


__all__ = [
    "ROWS_PER_COMPANY",
    "make_arm_result",
    "make_config",
    "make_dataset_info",
    "make_report",
    "make_separable_dataset",
    "write_bankruptcy_csv",
    "write_german_data",
    "write_taiwan_csv",
]
