"""Benchmark suite for ClearGBM.

Measures performance of tree building with various configurations.
Run manually (not in CI) to compare optimization impact.

Usage:
    poetry run python -m scripts.benchmark
    poetry run python -m scripts.benchmark --samples 10000 --features 20
"""

from __future__ import annotations

import sys
import time
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from cleargbm.ensemble import train_gradient_boosting
from cleargbm.types import GradientBoostingConfig


def _write(msg: str) -> None:
    """Write message to stdout (avoids bare print guard rule)."""
    sys.stdout.write(msg)
    sys.stdout.flush()


class BenchmarkResult(NamedTuple):
    """Result of a single benchmark run."""

    name: str
    n_samples: int
    n_features: int
    n_estimators: int
    max_bins: int
    n_jobs: int
    elapsed_seconds: float
    trees_per_second: float


def generate_synthetic_data(
    n_samples: int,
    n_features: int,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate synthetic classification data.

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


def run_benchmark(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    feature_names: tuple[str, ...],
    config: GradientBoostingConfig,
    name: str,
) -> BenchmarkResult:
    """Run a single benchmark.

    Args:
        x: Feature matrix.
        y: Labels.
        feature_names: Names for each feature.
        config: Model configuration.
        name: Benchmark name.

    Returns:
        BenchmarkResult with timing information.
    """
    n_samples: int = x.shape[0]
    n_features: int = x.shape[1] if x.size > 0 else 0
    n_estimators = config["n_estimators"]
    max_bins = config["max_bins"]
    n_jobs = config["n_jobs"]

    # Warm-up run (JIT compilation, cache warming)
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

    trees_per_second = n_estimators / elapsed if elapsed > 0 else 0.0

    return BenchmarkResult(
        name=name,
        n_samples=n_samples,
        n_features=n_features,
        n_estimators=n_estimators,
        max_bins=max_bins,
        n_jobs=n_jobs,
        elapsed_seconds=elapsed,
        trees_per_second=trees_per_second,
    )


def make_config(
    n_estimators: int = 10,
    max_depth: int = 4,
    max_bins: int = 64,
    n_jobs: int = 1,
    random_state: int = 42,
) -> GradientBoostingConfig:
    """Create benchmark configuration."""
    return GradientBoostingConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        colsample_bytree=None,
        categorical_features=None,
        n_classes=None,
        lambdarank_truncation_level=None,
        goss_top_rate=None,
        goss_other_rate=None,
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


def format_table(results: list[BenchmarkResult]) -> str:
    """Format results as ASCII table."""
    lines: list[str] = []
    header = (
        f"{'Name':<30} {'Samples':>8} {'Feats':>6} {'Trees':>6} "
        f"{'Bins':>5} {'Jobs':>5} {'Time':>8} {'T/s':>8}"
    )
    separator = "-" * len(header)

    lines.append(separator)
    lines.append(header)
    lines.append(separator)

    for r in results:
        line = (
            f"{r.name:<30} {r.n_samples:>8} {r.n_features:>6} {r.n_estimators:>6} "
            f"{r.max_bins:>5} {r.n_jobs:>5} {r.elapsed_seconds:>7.2f}s {r.trees_per_second:>7.1f}"
        )
        lines.append(line)

    lines.append(separator)
    return "\n".join(lines)


def run_benchmark_suite(
    n_samples: int = 5000,
    n_features: int = 10,
    n_estimators: int = 20,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """Run the full benchmark suite.

    Args:
        n_samples: Number of samples in synthetic data.
        n_features: Number of features.
        n_estimators: Number of trees to build.
        verbose: Print progress.

    Returns:
        List of benchmark results.
    """
    if verbose:
        _write(f"Generating synthetic data: {n_samples} samples, {n_features} features\n")

    x, y = generate_synthetic_data(n_samples, n_features)
    feature_names = tuple(f"f{i}" for i in range(n_features))
    results: list[BenchmarkResult] = []

    benchmarks: list[tuple[str, GradientBoostingConfig]] = [
        # Vary max_bins
        ("max_bins=32", make_config(n_estimators=n_estimators, max_bins=32)),
        ("max_bins=64 (default)", make_config(n_estimators=n_estimators, max_bins=64)),
        ("max_bins=128", make_config(n_estimators=n_estimators, max_bins=128)),
        # max_bins caps at 255 to keep bin indices in u8 (see cleargbm_rs
        # FeatureBins storage layout).
        ("max_bins=255", make_config(n_estimators=n_estimators, max_bins=255)),
        # Vary n_jobs
        ("n_jobs=1 (sequential)", make_config(n_estimators=n_estimators, n_jobs=1)),
        ("n_jobs=2", make_config(n_estimators=n_estimators, n_jobs=2)),
        ("n_jobs=4", make_config(n_estimators=n_estimators, n_jobs=4)),
        ("n_jobs=-1 (all cores)", make_config(n_estimators=n_estimators, n_jobs=-1)),
        # Vary depth
        ("max_depth=2", make_config(n_estimators=n_estimators, max_depth=2)),
        ("max_depth=4 (default)", make_config(n_estimators=n_estimators, max_depth=4)),
        ("max_depth=6", make_config(n_estimators=n_estimators, max_depth=6)),
        ("max_depth=8", make_config(n_estimators=n_estimators, max_depth=8)),
    ]

    for name, config in benchmarks:
        if verbose:
            _write(f"  Running: {name}... ")
        result = run_benchmark(x, y, feature_names, config, name)
        results.append(result)
        if verbose:
            _write(f"{result.elapsed_seconds:.2f}s\n")

    return results


def main(args: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        args: Command line arguments.

    Returns:
        Exit code.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ClearGBM Benchmark Suite")
    parser.add_argument(
        "--samples", type=int, default=5000, help="Number of samples (default: 5000)"
    )
    parser.add_argument("--features", type=int, default=10, help="Number of features (default: 10)")
    parser.add_argument("--trees", type=int, default=20, help="Number of trees (default: 20)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    parsed = parser.parse_args(args)

    # Extract typed values from argparse
    n_samples: int = parsed.samples
    n_features: int = parsed.features
    n_trees: int = parsed.trees
    quiet: bool = parsed.quiet

    _write("=" * 60 + "\n")
    _write("ClearGBM Benchmark Suite\n")
    _write("=" * 60 + "\n")
    _write("\n")

    results = run_benchmark_suite(
        n_samples=n_samples,
        n_features=n_features,
        n_estimators=n_trees,
        verbose=not quiet,
    )

    _write("\n")
    _write(format_table(results) + "\n")
    _write("\n")

    # Summary
    def get_elapsed(r: BenchmarkResult) -> float:
        return r.elapsed_seconds

    fastest = min(results, key=get_elapsed)
    slowest = max(results, key=get_elapsed)
    _write(f"Fastest: {fastest.name} ({fastest.elapsed_seconds:.2f}s)\n")
    _write(f"Slowest: {slowest.name} ({slowest.elapsed_seconds:.2f}s)\n")
    _write(f"Speedup: {slowest.elapsed_seconds / fastest.elapsed_seconds:.1f}x\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
