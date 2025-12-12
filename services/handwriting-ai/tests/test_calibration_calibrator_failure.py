from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import (
    CandidateRunnerProtocol,
    OrchestratorProtocol,
    PreprocessDatasetProtocol,
)
from handwriting_ai.training.calibration._types import (
    CalibrationResultDict,
    CandidateDict,
    OrchestratorConfigDict,
)
from handwriting_ai.training.calibration.calibrator import (
    CalibrationError,
)
from handwriting_ai.training.calibration.calibrator import (
    calibrate_input_pipeline as _cal,
)
from handwriting_ai.training.calibration.ds_spec import AugmentSpec, InlineSpec, PreprocessSpec
from handwriting_ai.training.resources import ResourceLimits

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


class _FakeMNIST:
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), 0


class _OrchEmpty:
    def __init__(self, *, runner: CandidateRunnerProtocol, config: OrchestratorConfigDict) -> None:
        _ = (runner, config)

    def run_stage_a(
        self,
        ds: PreprocessDatasetProtocol | PreprocessSpec,
        cands: list[CandidateDict],
        samples: int,
    ) -> list[CalibrationResultDict]:
        _ = (ds, cands, samples)
        return []

    def run_stage_b(
        self,
        ds: PreprocessDatasetProtocol | PreprocessSpec,
        shortlist: list[CalibrationResultDict],
        samples: int,
    ) -> list[CalibrationResultDict]:
        _ = (ds, shortlist, samples)
        return []


class _OrchEmptyB:
    def __init__(self, *, runner: CandidateRunnerProtocol, config: OrchestratorConfigDict) -> None:
        _ = (runner, config)

    def run_stage_a(
        self,
        ds: PreprocessDatasetProtocol | PreprocessSpec,
        cands: list[CandidateDict],
        samples: int,
    ) -> list[CalibrationResultDict]:
        _ = (ds, cands, samples)
        return [
            {
                "intra_threads": 1,
                "interop_threads": None,
                "num_workers": 0,
                "batch_size": 2,
                "samples_per_sec": 1.0,
                "p95_ms": 1.0,
            }
        ]

    def run_stage_b(
        self,
        ds: PreprocessDatasetProtocol | PreprocessSpec,
        shortlist: list[CalibrationResultDict],
        samples: int,
    ) -> list[CalibrationResultDict]:
        _ = (ds, shortlist, samples)
        return []


def test_calibrator_raises_on_empty_stage_a(tmp_path: Path) -> None:
    def _orch_factory(
        *, runner: CandidateRunnerProtocol, config: OrchestratorConfigDict
    ) -> OrchestratorProtocol:
        return _OrchEmpty(runner=runner, config=config)

    _test_hooks.orchestrator_factory = _orch_factory

    aug = AugmentSpec(
        augment=False,
        aug_rotate=0.0,
        aug_translate=0.0,
        noise_prob=0.0,
        noise_salt_vs_pepper=0.5,
        dots_prob=0.0,
        dots_count=0,
        dots_size_px=1,
        blur_sigma=0.0,
        morph="none",
    )
    base = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=4, sleep_s=0.0, fail=False),
        augment=aug,
    )
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=1024 * 1024 * 1024,
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=None,
    )
    with pytest.raises(CalibrationError):
        _cal(
            base,
            limits=limits,
            requested_batch_size=4,
            samples=1,
            cache_path=tmp_path / "calibration.json",
            ttl_seconds=0,
            force=True,
        )


def test_calibrator_raises_on_empty_stage_b(tmp_path: Path) -> None:
    def _orch_factory(
        *, runner: CandidateRunnerProtocol, config: OrchestratorConfigDict
    ) -> OrchestratorProtocol:
        return _OrchEmptyB(runner=runner, config=config)

    _test_hooks.orchestrator_factory = _orch_factory

    aug = AugmentSpec(
        augment=False,
        aug_rotate=0.0,
        aug_translate=0.0,
        noise_prob=0.0,
        noise_salt_vs_pepper=0.5,
        dots_prob=0.0,
        dots_count=0,
        dots_size_px=1,
        blur_sigma=0.0,
        morph="none",
    )
    base = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=4, sleep_s=0.0, fail=False),
        augment=aug,
    )
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=1024 * 1024 * 1024,
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=None,
    )
    with pytest.raises(CalibrationError):
        _cal(
            base,
            limits=limits,
            requested_batch_size=4,
            samples=1,
            cache_path=tmp_path / "calibration.json",
            ttl_seconds=0,
            force=True,
        )
