"""Shared inputs for the evaluation tests.

Datasets are generated, not mocked: a learnable signal in the first feature,
an irrelevant second and third feature, and rows grouped into companies, so
the real learners the runner drives have something to find and the cluster
unit has something to cluster.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from platform_core.comparability import NO_VALUE, RunFingerprint
from platform_core.determinism_env import SINGLE_THREAD
from platform_core.determinism_record import determinism_record
from platform_core.testing import sample_run_fingerprint

from covenant_ml.benchmarking.factory import make_benchmark_config
from covenant_ml.benchmarking.types import BenchmarkConfig, DatasetInfo
from covenant_ml.evaluation.types import AlertCosts, ResamplingSettings, ResamplingUnit

FINGERPRINT: RunFingerprint = sample_run_fingerprint(
    image_digest="sha256:" + "ab" * 32,
    gpu_model=NO_VALUE,
    driver_version=NO_VALUE,
    determinism=determinism_record("cpu", {"OMP_NUM_THREADS": SINGLE_THREAD}),
)

DATASET: DatasetInfo = {"sha256": "cd" * 32, "n_rows": 480, "n_features": 3}

COSTS: AlertCosts = {"true_alert_value": 10.0, "false_alert_cost": 1.0, "missed_cost": 5.0}

FEATURE_NAMES: tuple[str, ...] = ("signal", "noise_a", "noise_b")


def floats(values: list[float]) -> NDArray[np.float64]:
    """Build a float array from a typed list.

    Args:
        values: The values.

    Returns:
        A float64 array.
    """
    array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    return array


def ints(values: list[int]) -> NDArray[np.int64]:
    """Build an int array from a typed list.

    Args:
        values: The values.

    Returns:
        An int64 array.
    """
    array: NDArray[np.int64] = np.asarray(values, dtype=np.int64)
    return array


def at(array: NDArray[np.float64], index: int) -> float:
    """Read one element of a float array as a plain float.

    Args:
        array: The array.
        index: Flat index.

    Returns:
        The element.
    """
    return float(array.flat[index].item())


def grouped_dataset(
    n_companies: int = 120,
    rows_per_company: int = 4,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Generate company-grouped rows with a learnable first feature.

    Args:
        n_companies: Distinct companies.
        rows_per_company: Rows each company contributes.

    Returns:
        Features (n_rows, 3), labels and company codes.
    """
    n_rows = n_companies * rows_per_company
    rng = np.random.default_rng(11)
    features: NDArray[np.float64] = rng.normal(size=(n_rows, 3))
    noise: NDArray[np.float64] = rng.normal(scale=0.7, size=n_rows)
    signal: NDArray[np.float64] = features[:, 0] + noise
    labels: NDArray[np.int64] = (signal > 0.6).astype(np.int64)
    codes: NDArray[np.int64] = np.repeat(np.arange(n_companies, dtype=np.int64), rows_per_company)
    return features, labels, codes


def tiny_config() -> BenchmarkConfig:
    """A configuration small enough for every learner to fit in milliseconds.

    Returns:
        Five depth-2 trees over 16 bins.
    """
    return make_benchmark_config(
        n_estimators=5, max_depth=2, max_bins=16, num_leaves=4, repeats=1, warmups=0
    )


def settings(
    unit: ResamplingUnit = ResamplingUnit.CLUSTER,
    block_length: int = 1,
    n_resamples: int = 40,
) -> ResamplingSettings:
    """Resampling settings for a test run.

    Args:
        unit: Independent unit.
        block_length: Rows per block for ``block``.
        n_resamples: Bootstrap resamples.

    Returns:
        The settings at 95 percent coverage.
    """
    return {
        "unit": unit,
        "block_length": block_length,
        "n_resamples": n_resamples,
        "resample_seed": 3,
        "confidence": 0.95,
    }


__all__ = [
    "COSTS",
    "DATASET",
    "FEATURE_NAMES",
    "FINGERPRINT",
    "at",
    "floats",
    "grouped_dataset",
    "ints",
    "settings",
    "tiny_config",
]
