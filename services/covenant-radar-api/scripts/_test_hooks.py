"""Test hooks for CLI scripts.

Production code uses real implementations; tests can override these module-level
symbols to inject fakes without conditionals in core logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from covenant_radar_api.worker.optimize_job import (
    OptimizationResult,
    TrialProgressCallbackProtocol,
    TrialProgressInfo,
    run_optimization,
)


class OptimizationRunnerProtocol(Protocol):
    """Protocol for optimization runner function."""

    def __call__(
        self,
        config_json: str,
        external_dir: Path,
        output_dir: Path,
        progress_callback: TrialProgressCallbackProtocol | None = None,
    ) -> OptimizationResult:
        """Run hyperparameter optimization.

        Args:
            config_json: JSON configuration string
            external_dir: Directory with external datasets
            output_dir: Directory for output files
            progress_callback: Optional callback for trial progress updates

        Returns:
            Optimization result with best parameters
        """
        ...


# Default to real implementation - run_optimization signature matches the protocol
optimization_runner: OptimizationRunnerProtocol = run_optimization


__all__ = [
    "OptimizationRunnerProtocol",
    "TrialProgressCallbackProtocol",
    "TrialProgressInfo",
    "optimization_runner",
]
