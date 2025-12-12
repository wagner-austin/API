from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import (
    CalibrationRunnerResultDict,
    PreprocessDatasetProtocol,
)
from handwriting_ai.training.calibration._types import CandidateDict
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.ds_spec import AugmentSpec, InlineSpec, PreprocessSpec
from handwriting_ai.training.calibration.runner import (
    _child_entry,
    _mnist_find_raw_dir,
    _mnist_read_images_labels,
)
from handwriting_ai.training.dataset import AugmentConfig, PreprocessDataset


class _Base:
    """Minimal base dataset for testing."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), int(idx)


_CFG: AugmentConfig = {
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
    "morph_kernel_px": 1,
    "batch_size": 1,
}


def test_child_entry_emits_logs_and_writes_result(tmp_path: Path) -> None:
    """Test that _child_entry emits logs and writes result file."""

    def _fake_setup_logging(
        *,
        level: str,
        format_mode: str,
        service_name: str,
        instance_id: str | None,
        extra_fields: list[str] | None,
    ) -> None:
        pass

    def _fake_build_dataset_from_spec(spec: PreprocessSpec) -> PreprocessDataset:
        return PreprocessDataset(_Base(), _CFG)

    def _fake_measure_candidate_internal(
        ds: PreprocessDatasetProtocol,
        cand: CandidateDict,
        samples: int,
        on_improvement: Callable[[CalibrationRunnerResultDict], None] | None,
        *,
        enable_headroom: bool,
    ) -> CalibrationRunnerResultDict:
        result: CalibrationRunnerResultDict = {
            "intra_threads": cand["intra_threads"],
            "interop_threads": cand["interop_threads"],
            "num_workers": cand["num_workers"],
            "batch_size": cand["batch_size"],
            "samples_per_sec": 1.0,
            "p95_ms": 1.0,
        }
        if on_improvement is not None:
            on_improvement(result)
        return result

    # Set hooks
    _test_hooks.runner_setup_logging = _fake_setup_logging
    _test_hooks.build_dataset_from_spec = _fake_build_dataset_from_spec
    _test_hooks.measure_candidate_internal = _fake_measure_candidate_internal

    log_q: mp.Queue[logging.LogRecord] = mp.Queue()

    spec: PreprocessSpec = {
        "base_kind": "inline",
        "mnist": None,
        "inline": InlineSpec(n=1, sleep_s=0.0, fail=False),
        "augment": AugmentSpec(
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
        ),
    }
    cand: Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 1,
    }

    out_path = tmp_path / "res.txt"
    _child_entry(
        out_path=out_path.as_posix(),
        spec=spec,
        cand=cand,
        samples=1,
        abort_pct=90.0,
        log_q=log_q,
    )
    records: list[logging.LogRecord] = []
    while not log_q.empty():
        records.append(log_q.get())
    assert any("calibration_child_started" in r.getMessage() for r in records)
    assert out_path.exists()


def test_mnist_find_raw_dir_prefers_raw(tmp_path: Path) -> None:
    """Test that _mnist_find_raw_dir prefers MNIST/raw subdirectory."""
    root = tmp_path / "mnist"
    raw = root / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    assert _mnist_find_raw_dir(root) == raw


def test_mnist_read_images_labels_missing_files_raises(tmp_path: Path) -> None:
    """Test that _mnist_read_images_labels raises when files are missing."""
    with pytest.raises(RuntimeError):
        _ = _mnist_read_images_labels(tmp_path, train=True)


def test_child_entry_handler_without_flush(tmp_path: Path) -> None:
    """Test that _child_entry handles log handlers without flush method (line 190->189)."""
    from platform_core.logging import stdlib_logging

    def _fake_setup_logging(
        *,
        level: str,
        format_mode: str,
        service_name: str,
        instance_id: str | None,
        extra_fields: list[str] | None,
    ) -> None:
        # Inject a handler without flush attribute using the test hook
        # Use "handwriting_ai" logger since that's what _child_entry uses at line 119
        log = stdlib_logging.getLogger("handwriting_ai")
        _test_hooks.inject_no_flush_handler(log)

    def _fake_build_dataset_from_spec(spec: PreprocessSpec) -> PreprocessDataset:
        return PreprocessDataset(_Base(), _CFG)

    def _fake_measure_candidate_internal(
        ds: PreprocessDatasetProtocol,
        cand: CandidateDict,
        samples: int,
        on_improvement: Callable[[CalibrationRunnerResultDict], None] | None,
        *,
        enable_headroom: bool,
    ) -> CalibrationRunnerResultDict:
        result: CalibrationRunnerResultDict = {
            "intra_threads": cand["intra_threads"],
            "interop_threads": cand["interop_threads"],
            "num_workers": cand["num_workers"],
            "batch_size": cand["batch_size"],
            "samples_per_sec": 1.0,
            "p95_ms": 1.0,
        }
        return result

    # Set hooks
    _test_hooks.runner_setup_logging = _fake_setup_logging
    _test_hooks.build_dataset_from_spec = _fake_build_dataset_from_spec
    _test_hooks.measure_candidate_internal = _fake_measure_candidate_internal

    log_q: mp.Queue[logging.LogRecord] = mp.Queue()

    spec: PreprocessSpec = {
        "base_kind": "inline",
        "mnist": None,
        "inline": InlineSpec(n=1, sleep_s=0.0, fail=False),
        "augment": AugmentSpec(
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
        ),
    }
    cand: Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 1,
    }

    out_path = tmp_path / "res.txt"
    # This should complete without error even with handler lacking flush
    _child_entry(
        out_path=out_path.as_posix(),
        spec=spec,
        cand=cand,
        samples=1,
        abort_pct=90.0,
        log_q=log_q,
    )
    assert out_path.exists()
