"""Calibration TypedDicts - shared types without _test_hooks imports.

This module contains TypedDicts used by calibration code. It's separate
to avoid circular imports between _test_hooks.py and calibration modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict


class Candidate(TypedDict):
    """Candidate configuration for calibration."""

    intra_threads: int
    interop_threads: int | None
    num_workers: int
    batch_size: int


class CalibrationResult(TypedDict):
    """Result from measuring a candidate configuration."""

    intra_threads: int
    interop_threads: int | None
    num_workers: int
    batch_size: int
    samples_per_sec: float
    p95_ms: float


class BudgetConfig(TypedDict):
    """Budget configuration for calibration stages."""

    start_pct_max: float
    abort_pct: float
    timeout_s: float
    max_failures: int


class OrchestratorConfig(TypedDict):
    """Configuration for the calibration orchestrator."""

    stage_a_budget: BudgetConfig
    stage_b_budget: BudgetConfig
    checkpoint_path: Path


class CandidateError(TypedDict):
    """Error information from a candidate run."""

    kind: str  # "timeout" | "oom" | "runtime"
    message: str
    exit_code: int | None


class CandidateOutcome(TypedDict):
    """Outcome from running a calibration candidate."""

    ok: bool
    res: CalibrationResult | None
    error: CandidateError | None


__all__ = [
    "BudgetConfig",
    "CalibrationResult",
    "Candidate",
    "CandidateError",
    "CandidateOutcome",
    "OrchestratorConfig",
]
