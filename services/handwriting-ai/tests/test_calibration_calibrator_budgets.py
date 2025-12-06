from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pytest
from PIL import Image

from handwriting_ai.monitoring import (
    CgroupMemoryBreakdown,
    CgroupMemoryUsage,
    MemorySnapshot,
    ProcessMemory,
)
from handwriting_ai.training.calibration.calibrator import calibrate_input_pipeline as _cal
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.ds_spec import AugmentSpec, InlineSpec, PreprocessSpec
from handwriting_ai.training.calibration.measure import CalibrationResult
from handwriting_ai.training.calibration.orchestrator import OrchestratorConfig
from handwriting_ai.training.calibration.runner import CandidateRunner
from handwriting_ai.training.dataset import PreprocessDataset
from handwriting_ai.training.resources import ResourceLimits


class _OrchestratorProtocol(Protocol):
    def __init__(self, *, runner: CandidateRunner, config: OrchestratorConfig) -> None: ...
    def run_stage_a(
        self, ds: PreprocessDataset, cands: list[Candidate], samples: int
    ) -> list[CalibrationResult]: ...
    def run_stage_b(
        self, ds: PreprocessDataset, shortlist: list[CalibrationResult], samples: int
    ) -> list[CalibrationResult]: ...


class _FakeMNIST:
    def __init__(self, n: int = 8) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), 0


def _make_snapshot(limit_bytes: int) -> MemorySnapshot:
    """Create a MemorySnapshot TypedDict for testing."""
    main: ProcessMemory = {"pid": 1, "rss_bytes": 100 * 1024 * 1024}
    cgroup_usage: CgroupMemoryUsage = {
        "usage_bytes": 500 * 1024 * 1024,
        "limit_bytes": limit_bytes,
        "percent": 50.0,
    }
    cgroup_breakdown: CgroupMemoryBreakdown = {
        "anon_bytes": 400 * 1024 * 1024,
        "file_bytes": 50 * 1024 * 1024,
        "kernel_bytes": 30 * 1024 * 1024,
        "slab_bytes": 20 * 1024 * 1024,
    }
    return {
        "main_process": main,
        "workers": (),
        "cgroup_usage": cgroup_usage,
        "cgroup_breakdown": cgroup_breakdown,
    }


def test_calibrator_low_mem_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Monkeypatch orchestrator to capture budgets and avoid subprocess work
    captured: list[OrchestratorConfig] = []

    class _Orch:
        def __init__(self, *, runner: CandidateRunner, config: OrchestratorConfig) -> None:
            _ = runner
            captured.append(config)

        def run_stage_a(
            self, ds: PreprocessDataset, cands: list[Candidate], samples: int
        ) -> list[CalibrationResult]:
            # return one result per candidate
            return [
                {
                    "intra_threads": int(c["intra_threads"]),
                    "interop_threads": c["interop_threads"],
                    "num_workers": int(c["num_workers"]),
                    "batch_size": int(c["batch_size"]),
                    "samples_per_sec": 1.0,
                    "p95_ms": 1.0,
                }
                for c in cands
            ]

        def run_stage_b(
            self, ds: PreprocessDataset, shortlist: list[CalibrationResult], samples: int
        ) -> list[CalibrationResult]:
            return shortlist

    _orch_typed: type[_OrchestratorProtocol] = _Orch

    def _snap_fn() -> MemorySnapshot:
        return _make_snapshot(1024 * 1024 * 1024)  # 1GB

    monkeypatch.setattr("handwriting_ai.training.calibration.calibrator.Orchestrator", _orch_typed)
    monkeypatch.setattr(
        "handwriting_ai.training.calibration.calibrator.get_memory_snapshot", _snap_fn
    )
    aug: AugmentSpec = {
        "augment": False,
        "aug_rotate": 0.0,
        "aug_translate": 0.0,
        "noise_prob": 0.0,
        "noise_salt_vs_pepper": 0.5,
        "dots_prob": 0.0,
        "dots_count": 0,
        "dots_size_px": 1,
        "blur_sigma": 0.0,
        "morph": "none",
    }
    inline: InlineSpec = {"n": 8, "sleep_s": 0.0, "fail": False}
    base: PreprocessSpec = {
        "base_kind": "inline",
        "mnist": None,
        "inline": inline,
        "augment": aug,
    }
    limits: ResourceLimits = {
        "cpu_cores": 2,
        "memory_bytes": 1024 * 1024 * 1024,
        "optimal_threads": 1,
        "optimal_workers": 0,
        "max_batch_size": None,
    }
    _cal(
        base,
        limits=limits,
        requested_batch_size=4,
        samples=1,
        cache_path=Path("/tmp/calib.json"),
        ttl_seconds=0,
        force=True,
    )
    cfg = captured[0]
    assert cfg["stage_a_budget"]["start_pct_max"] == 80.0
    assert cfg["stage_b_budget"]["abort_pct"] == 88.0


def test_calibrator_high_mem_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[OrchestratorConfig] = []

    class _Orch:
        def __init__(self, *, runner: CandidateRunner, config: OrchestratorConfig) -> None:
            _ = runner
            captured.append(config)

        def run_stage_a(
            self, ds: PreprocessDataset, cands: list[Candidate], samples: int
        ) -> list[CalibrationResult]:
            out: list[CalibrationResult] = []
            for c in cands:
                out.append(
                    {
                        "intra_threads": int(c["intra_threads"]),
                        "interop_threads": c["interop_threads"],
                        "num_workers": int(c["num_workers"]),
                        "batch_size": int(c["batch_size"]),
                        "samples_per_sec": 1.0,
                        "p95_ms": 1.0,
                    }
                )
            return out

        def run_stage_b(
            self, ds: PreprocessDataset, shortlist: list[CalibrationResult], samples: int
        ) -> list[CalibrationResult]:
            return shortlist

    _orch_typed: type[_OrchestratorProtocol] = _Orch

    def _snap_fn() -> MemorySnapshot:
        return _make_snapshot(4 * 1024 * 1024 * 1024)  # 4GB

    monkeypatch.setattr("handwriting_ai.training.calibration.calibrator.Orchestrator", _orch_typed)
    monkeypatch.setattr(
        "handwriting_ai.training.calibration.calibrator.get_memory_snapshot", _snap_fn
    )
    aug: AugmentSpec = {
        "augment": False,
        "aug_rotate": 0.0,
        "aug_translate": 0.0,
        "noise_prob": 0.0,
        "noise_salt_vs_pepper": 0.5,
        "dots_prob": 0.0,
        "dots_count": 0,
        "dots_size_px": 1,
        "blur_sigma": 0.0,
        "morph": "none",
    }
    inline: InlineSpec = {"n": 8, "sleep_s": 0.0, "fail": False}
    base: PreprocessSpec = {
        "base_kind": "inline",
        "mnist": None,
        "inline": inline,
        "augment": aug,
    }
    limits: ResourceLimits = {
        "cpu_cores": 4,
        "memory_bytes": 4 * 1024 * 1024 * 1024,
        "optimal_threads": 2,
        "optimal_workers": 1,
        "max_batch_size": None,
    }
    _cal(
        base,
        limits=limits,
        requested_batch_size=4,
        samples=1,
        cache_path=Path("/tmp/calib.json"),
        ttl_seconds=0,
        force=True,
    )
    cfg = captured[0]
    assert cfg["stage_a_budget"]["start_pct_max"] == 85.0
    assert cfg["stage_b_budget"]["abort_pct"] == 92.0
