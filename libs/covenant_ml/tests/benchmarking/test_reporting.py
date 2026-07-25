"""Tests for reducing a manifest to a report."""

from __future__ import annotations

import pytest

from covenant_ml.benchmarking.reporting import (
    render_report,
    render_seed_line,
    summarize_gap,
    summarize_model,
)
from covenant_ml.benchmarking.types import (
    ERR_NO_RESULTS,
    MANIFEST_SCHEMA_VERSION,
    BenchmarkManifest,
    BenchmarkModelName,
    SeedResult,
)


def make_result(
    model: BenchmarkModelName,
    seed: int,
    canonical_s: float,
    mean_leaves: float,
) -> SeedResult:
    """Build a per-seed record.

    Args:
        model: Which model.
        seed: Seed value.
        canonical_s: Canonical fit time.
        mean_leaves: Mean leaves per tree.

    Returns:
        The record.
    """
    return {
        "model": model,
        "seed": seed,
        "ran_first": True,
        "timing": {
            "canonical_s": canonical_s,
            "min_s": canonical_s - 0.1,
            "median_s": canonical_s,
            "mean_s": canonical_s,
            "max_s": canonical_s + 0.1,
            "samples_s": [canonical_s],
        },
        "quality": {
            "auc_roc": 0.68,
            "auc_pr": 0.14,
            "log_loss": 0.23,
            "brier": 0.06,
            "mean_pred": 0.065,
            "positive_rate": 0.066,
        },
        "mean_leaves": mean_leaves,
    }


def make_manifest(results: list[SeedResult]) -> BenchmarkManifest:
    """Wrap records in a manifest.

    Args:
        results: Records to include.

    Returns:
        The manifest.
    """
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "estimator": "median",
        "config": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "max_bins": 64,
            "min_data_in_leaf": 20,
            "num_leaves": 31,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "repeats": 5,
            "warmups": 2,
        },
        "dataset": {"sha256": "c" * 64, "n_rows": 78682, "n_features": 18},
        "seeds": [42],
        "results": results,
    }


def test_model_summary_averages_across_seeds() -> None:
    results = [
        make_result("cleargbm", 42, 1.0, 50.0),
        make_result("cleargbm", 43, 3.0, 60.0),
    ]
    summary = summarize_model(results, "cleargbm")
    assert summary.mean_fit_s == 2.0
    assert summary.mean_leaves == 55.0


def test_single_seed_has_zero_standard_deviation() -> None:
    summary = summarize_model([make_result("cleargbm", 42, 1.0, 50.0)], "cleargbm")
    assert summary.stdev_fit_s == 0.0


def test_multiple_seeds_report_a_standard_deviation() -> None:
    results = [
        make_result("cleargbm", 42, 1.0, 50.0),
        make_result("cleargbm", 43, 3.0, 50.0),
    ]
    summary = summarize_model(results, "cleargbm")
    assert summary.stdev_fit_s > 0.0


def test_missing_model_raises() -> None:
    with pytest.raises(ValueError, match=ERR_NO_RESULTS):
        summarize_model([make_result("cleargbm", 42, 1.0, 50.0)], "lightgbm")


def test_normalized_ratio_divides_out_the_tree_size_difference() -> None:
    """Twice the work at twice the time is parity per unit of work."""
    manifest = make_manifest(
        [
            make_result("cleargbm", 42, 2.0, 60.0),
            make_result("lightgbm", 42, 1.0, 30.0),
        ]
    )
    gap = summarize_gap(manifest)
    assert gap.raw_ratio == 2.0
    assert gap.leaf_ratio == 2.0
    assert gap.normalized_ratio == 1.0


def test_normalized_ratio_exposes_a_real_per_leaf_deficit() -> None:
    manifest = make_manifest(
        [
            make_result("cleargbm", 42, 3.0, 60.0),
            make_result("lightgbm", 42, 1.0, 30.0),
        ]
    )
    gap = summarize_gap(manifest)
    assert gap.raw_ratio == 3.0
    assert gap.normalized_ratio == pytest.approx(1.5)


def test_seed_line_shows_the_full_spread() -> None:
    line = render_seed_line(make_result("cleargbm", 42, 1.5, 57.9))
    assert "cleargbm" in line
    assert "seed=42" in line
    assert "leaves=57.90" in line
    # The spread is always visible so noise is distinguishable from signal.
    assert "over 1:" in line


def test_report_contains_both_models_and_all_three_ratios() -> None:
    manifest = make_manifest(
        [
            make_result("cleargbm", 42, 2.0, 60.0),
            make_result("lightgbm", 42, 1.0, 30.0),
        ]
    )
    report = render_report(manifest)
    assert "cleargbm" in report
    assert "lightgbm" in report
    assert "raw ratio" in report
    assert "leaf ratio" in report
    assert "per-leaf ratio" in report
    assert report.endswith("\n")


def test_report_records_the_dataset_and_config() -> None:
    manifest = make_manifest(
        [
            make_result("cleargbm", 42, 2.0, 60.0),
            make_result("lightgbm", 42, 1.0, 30.0),
        ]
    )
    report = render_report(manifest)
    assert "rows=78682" in report
    assert "trees=200" in report
    assert "median" in report
