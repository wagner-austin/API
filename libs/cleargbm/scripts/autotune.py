"""Autotune script for ClearGBM.

Runs grid search over {n_jobs, max_bins} to find optimal configuration
for the user's specific data and hardware.

Usage:
    poetry run python -m scripts.autotune --help
    poetry run python -m scripts.autotune --samples 1000 --features 20
"""

from __future__ import annotations

import sys
import time

import numpy as np
from numpy.typing import NDArray

from cleargbm.ensemble import train_gradient_boosting
from cleargbm.types import (
    GradientBoostingConfig,
    TimingResult,
    TuningReport,
)


def _write(msg: str) -> None:
    """Write message to stdout (avoids bare print guard rule).

    Args:
        msg: Message to write.
    """
    sys.stdout.write(msg)
    sys.stdout.flush()


def generate_sample_data(
    n_samples: int,
    n_features: int,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate synthetic classification data for tuning.

    Uses a simple linear decision boundary with noise.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y) where X is feature matrix and y is labels.
    """
    rng = np.random.default_rng(seed)

    # Generate random features
    x: NDArray[np.float64] = rng.random((n_samples, n_features), dtype=np.float64)

    # Decision: sum of first half > sum of second half + noise
    half = n_features // 2
    sum_first_half: NDArray[np.float64] = np.sum(x[:, :half], axis=1)
    sum_second_half: NDArray[np.float64] = np.sum(x[:, half:], axis=1)
    score: NDArray[np.float64] = sum_first_half - sum_second_half
    noise: NDArray[np.float64] = rng.random(n_samples, dtype=np.float64) - 0.5
    y: NDArray[np.int64] = (score + noise * 0.5 > 0).astype(np.int64)

    return x, y


def make_config(
    n_estimators: int = 5,
    max_depth: int = 4,
    max_bins: int = 64,
    n_jobs: int = 1,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> GradientBoostingConfig:
    """Create tuning configuration.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        max_bins: Number of histogram bins.
        n_jobs: Number of parallel workers.
        learning_rate: Learning rate.
        random_state: Random seed.

    Returns:
        GradientBoostingConfig with specified values.
    """
    return GradientBoostingConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        colsample_bytree=None,
        categorical_features=None,
        n_classes=None,
        lambdarank_truncation_level=None,
        goss_top_rate=None,
        goss_other_rate=None,
        quantized_gradient_bins=None,
        min_data_in_bin=None,
        max_bins=max_bins,
        subsample=1.0,
        random_state=random_state,
        monotonic_constraints=None,
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_jobs=n_jobs,
        early_stopping_rounds=None,
        growth_strategy="depth_wise",
        num_leaves=None,
        objective="binary_log_loss",
        scale_pos_weight=1.0,
    )


def time_config(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    feature_names: tuple[str, ...],
    config: GradientBoostingConfig,
) -> TimingResult:
    """Time a single configuration.

    Args:
        x: Feature matrix.
        y: Labels.
        feature_names: Feature names.
        config: Configuration to time.

    Returns:
        TimingResult with timing information.
    """
    # Warm-up run
    _ = train_gradient_boosting(
        x_train=x,
        y_train=y,
        x_val=None,
        y_val=None,
        config=config,
        feature_names=feature_names,
    )

    # Timed run
    start = time.perf_counter()
    _ = train_gradient_boosting(
        x_train=x,
        y_train=y,
        x_val=None,
        y_val=None,
        config=config,
        feature_names=feature_names,
    )
    elapsed = time.perf_counter() - start

    n_estimators = config["n_estimators"]
    trees_per_second = n_estimators / elapsed if elapsed > 0 else 0.0

    return TimingResult(
        n_jobs=config["n_jobs"],
        max_bins=config["max_bins"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        elapsed_seconds=elapsed,
        trees_per_second=trees_per_second,
    )


def run_autotune(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    n_estimators: int = 5,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    n_jobs_grid: tuple[int, ...] = (1, 2, 4),
    max_bins_grid: tuple[int, ...] = (32, 64, 128),
    verbose: bool = True,
) -> TuningReport:
    """Run autotune grid search.

    Args:
        x: Feature matrix (n_samples, n_features).
        y: Labels (n_samples,).
        n_estimators: Number of trees per config.
        max_depth: Tree depth for tuning.
        learning_rate: Learning rate for tuning.
        n_jobs_grid: Grid of n_jobs values to test.
        max_bins_grid: Grid of max_bins values to test.
        verbose: Print progress.

    Returns:
        TuningReport with recommendations.
    """
    start_time = time.perf_counter()
    n_samples: int = x.shape[0]
    n_features: int = x.shape[1] if x.size > 0 else 0
    feature_names = tuple(f"f{i}" for i in range(n_features))

    if verbose:
        _write(f"Autotuning on {n_samples} samples, {n_features} features\n")
        _write(f"Grid: n_jobs={n_jobs_grid}, max_bins={max_bins_grid}\n\n")

    timing_results: list[TimingResult] = []
    best_result: TimingResult | None = None
    sequential_time: float = 0.0

    for n_jobs in n_jobs_grid:
        for max_bins in max_bins_grid:
            config = make_config(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_bins=max_bins,
                n_jobs=n_jobs,
                learning_rate=learning_rate,
            )

            if verbose:
                _write(f"  n_jobs={n_jobs:2d}, max_bins={max_bins:3d}... ")

            result = time_config(x, y, feature_names, config)
            timing_results.append(result)

            if verbose:
                _write(f"{result['elapsed_seconds']:.2f}s\n")

            # Track sequential time for speedup calculation
            if n_jobs == 1 and max_bins == 64:
                sequential_time = result["elapsed_seconds"]

            # Track best result (lowest elapsed time)
            if best_result is None or result["elapsed_seconds"] < best_result["elapsed_seconds"]:
                best_result = result

    # Ensure we have a best result
    if best_result is None:
        raise ValueError("No timing results collected")

    # Calculate speedup vs sequential
    parallel_speedup = 1.0
    if sequential_time > 0 and best_result["elapsed_seconds"] > 0:
        parallel_speedup = sequential_time / best_result["elapsed_seconds"]

    # Build best config
    best_config = make_config(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_bins=best_result["max_bins"],
        n_jobs=best_result["n_jobs"],
        learning_rate=learning_rate,
    )

    total_time = time.perf_counter() - start_time

    return TuningReport(
        best_config=best_config,
        timing_results=tuple(timing_results),
        sample_size=n_samples,
        n_features=n_features,
        recommended_n_jobs=best_result["n_jobs"],
        recommended_max_bins=best_result["max_bins"],
        parallel_speedup=parallel_speedup,
        total_tune_time_seconds=total_time,
    )


def format_report(report: TuningReport) -> str:
    """Format TuningReport as readable text.

    Args:
        report: Report to format.

    Returns:
        Formatted string.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("ClearGBM Autotune Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Dataset: {report['sample_size']} samples, {report['n_features']} features")
    lines.append(f"Tune time: {report['total_tune_time_seconds']:.1f}s")
    lines.append("")
    lines.append("Recommendations:")
    lines.append(f"  n_jobs: {report['recommended_n_jobs']}")
    lines.append(f"  max_bins: {report['recommended_max_bins']}")
    lines.append(f"  Speedup vs sequential: {report['parallel_speedup']:.2f}x")
    lines.append("")
    lines.append("Timing Results:")
    lines.append("-" * 50)
    lines.append(f"{'n_jobs':>8} {'max_bins':>10} {'time (s)':>12} {'trees/s':>10}")
    lines.append("-" * 50)

    for r in report["timing_results"]:
        lines.append(
            f"{r['n_jobs']:>8} {r['max_bins']:>10} {r['elapsed_seconds']:>12.2f} "
            f"{r['trees_per_second']:>10.1f}"
        )

    lines.append("-" * 50)
    lines.append("")
    return "\n".join(lines)


def main(args: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        args: Command line arguments.

    Returns:
        Exit code.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ClearGBM Autotune")
    parser.add_argument(
        "--samples", type=int, default=2000, help="Number of samples (default: 2000)"
    )
    parser.add_argument("--features", type=int, default=20, help="Number of features (default: 20)")
    parser.add_argument("--trees", type=int, default=5, help="Trees per config (default: 5)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    parsed = parser.parse_args(args)

    # Extract typed values
    n_samples: int = parsed.samples
    n_features: int = parsed.features
    n_trees: int = parsed.trees
    quiet: bool = parsed.quiet

    if not quiet:
        _write("Generating sample data...\n")

    x, y = generate_sample_data(n_samples, n_features)

    report = run_autotune(
        x=x,
        y=y,
        n_estimators=n_trees,
        verbose=not quiet,
    )

    _write("\n")
    _write(format_report(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
