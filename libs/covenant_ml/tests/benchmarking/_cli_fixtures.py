"""Inputs shared by the paired-evaluation entry-point tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from platform_core.determinism_cpu import CPU_STACK
from platform_core.determinism_env import BLAS_THREAD_ENV_VARS, SINGLE_THREAD
from platform_core.determinism_record import DeterminismRecord, determinism_record


def stand_in_pin() -> DeterminismRecord:
    """Report the posture a production run would have, without pinning.

    The real pin refuses once numpy is loaded, and this is a numpy suite;
    whether it refuses is asserted by running each script as ``__main__``.

    Returns:
        The single-thread CPU posture.
    """
    return determinism_record(CPU_STACK, dict.fromkeys(BLAS_THREAD_ENV_VARS, SINGLE_THREAD))


def write_corpus(directory: Path, n_companies: int = 120) -> Path:
    """Write a bankruptcy-shaped CSV with a learnable first feature.

    Args:
        directory: Destination directory.
        n_companies: Distinct companies, four years each.

    Returns:
        Path to the written CSV.
    """
    rng = np.random.default_rng(21)
    lines = ["company_name,status_label,year,X1,X2,X3"]
    for company in range(n_companies):
        values: NDArray[np.float64] = rng.normal(size=(4, 3))
        for year in range(4):
            x1 = float(values.flat[year * 3].item())
            x2 = float(values.flat[year * 3 + 1].item())
            x3 = float(values.flat[year * 3 + 2].item())
            label = "failed" if x1 + 0.5 * x2 > 0.7 else "alive"
            lines.append(f"C_{company},{label},{2000 + year},{x1:.5f},{x2:.5f},{x3:.5f}")
    path = directory / "corpus.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def model_flags() -> list[str]:
    """The model-shape flags both families take, at a size that fits instantly.

    Returns:
        Command-line arguments.
    """
    return ["--trees", "4", "--max-depth", "2", "--max-bins", "16", "--num-leaves", "4"]


def interval_flags(unit: str) -> list[str]:
    """The resampling flags both families take.

    Args:
        unit: The ``--unit`` value.

    Returns:
        Command-line arguments.
    """
    return [
        "--unit",
        unit,
        "--block-length",
        "1",
        "--n-resamples",
        "30",
        "--resample-seed",
        "3",
        "--confidence",
        "0.9",
    ]


__all__ = ["interval_flags", "model_flags", "stand_in_pin", "write_corpus"]
