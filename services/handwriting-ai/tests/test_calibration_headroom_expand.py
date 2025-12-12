from __future__ import annotations

import torch
from PIL import Image
from torch.nn import Module as TorchModule
from torch.optim.optimizer import Optimizer as TorchOptimizer

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import (
    BatchIterableProtocol,
    DataLoaderConfigProtocol,
    PreprocessDatasetProtocol,
)
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.measure import _measure_candidate
from handwriting_ai.training.dataset import AugmentConfig, PreprocessDataset
from handwriting_ai.training.safety import (
    MemoryGuardConfig,
    reset_memory_guard,
    set_memory_guard_config,
)


class _StubBatchIterator:
    """Stub iterator that yields nothing."""

    def __iter__(self) -> _StubBatchIterator:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise StopIteration


class _StubLoader:
    """Stub loader implementing BatchIterableProtocol for test stubs."""

    def __iter__(self) -> _StubBatchIterator:
        return _StubBatchIterator()


class _Base:
    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        # Loader is not used by the stubbed _measure_training
        return Image.new("L", (28, 28), 0), int(idx % 10)


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


def _fake_loader(
    ds: PreprocessDatasetProtocol, cfg: DataLoaderConfigProtocol
) -> BatchIterableProtocol:
    _ = (ds, cfg)
    return _StubLoader()


def test_headroom_expansion_raises_upper_bound() -> None:
    # Arrange: dataset and candidate with a small initial cap
    ds = PreprocessDataset(_Base(), _CFG)
    cand: Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 2,
    }

    # Stub loader via hook
    _test_hooks.safe_loader = _fake_loader

    # Stub measurement: succeed while bs < 4 with low peak (10%), then fail at >= 4
    calls: dict[str, list[int]] = {"seen": []}

    def _stub_measure_training(
        ds_len: int,
        loader: BatchIterableProtocol,
        k: int,
        *,
        device: torch.device,
        batch_size_hint: int,
        model: TorchModule,
        opt: TorchOptimizer,
    ) -> tuple[float, float, float, bool]:
        _ = (ds_len, loader, k, device, model, opt)
        calls["seen"].append(int(batch_size_hint))
        if batch_size_hint >= 4:
            return 0.0, 0.0, 75.0, True  # exceeded near guard, no headroom
        return 100.0, 1.0, 10.0, False  # low peak -> triggers expansion at cap

    _test_hooks.measure_training = _stub_measure_training

    # Enable memory guard with a positive threshold so headroom expansion
    # logic is active for this test.
    set_memory_guard_config(
        MemoryGuardConfig(enabled=True, threshold_percent=92.0, required_consecutive=1)
    )
    reset_memory_guard()
    try:
        # Act
        res = _measure_candidate(ds, cand, samples=2)
    finally:
        # Restore default disabled guard to avoid cross-test effects
        set_memory_guard_config(
            MemoryGuardConfig(enabled=False, threshold_percent=0.0, required_consecutive=0)
        )
        reset_memory_guard()

    # Assert: best batch exceeds the initial cap (2) because we expanded window
    assert res["batch_size"] >= 3
    # Ensure we attempted batch sizes beyond the initial upper bound
    assert any(bs > 2 for bs in calls["seen"])  # proves expansion occurred


def test_headroom_expansion_requires_guard_enabled() -> None:
    # Arrange: dataset and candidate with a small initial cap
    ds = PreprocessDataset(_Base(), _CFG)
    cand: Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 2,
    }

    _test_hooks.safe_loader = _fake_loader

    calls: dict[str, list[int]] = {"seen": []}

    def _stub_measure_training(
        ds_len: int,
        loader: BatchIterableProtocol,
        k: int,
        *,
        device: torch.device,
        batch_size_hint: int,
        model: TorchModule,
        opt: TorchOptimizer,
    ) -> tuple[float, float, float, bool]:
        _ = (ds_len, loader, k, device, model, opt)
        calls["seen"].append(int(batch_size_hint))
        # Always succeed with a low peak usage to exercise headroom logic.
        return 100.0, 1.0, 10.0, False

    _test_hooks.measure_training = _stub_measure_training

    # Disable guard so headroom expansion should be inactive even with low peak usage.
    set_memory_guard_config(
        MemoryGuardConfig(enabled=False, threshold_percent=92.0, required_consecutive=1)
    )
    reset_memory_guard()
    try:
        res = _measure_candidate(ds, cand, samples=2)
    finally:
        set_memory_guard_config(
            MemoryGuardConfig(enabled=False, threshold_percent=0.0, required_consecutive=0)
        )
        reset_memory_guard()

    # With guard disabled we should not expand beyond the initial cap.
    assert res["batch_size"] <= cand["batch_size"]
    assert max(calls["seen"]) <= cand["batch_size"]
