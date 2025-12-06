from __future__ import annotations

from typing import Protocol

import pytest
import torch
from PIL import Image
from torch.nn import Module
from torch.optim.optimizer import Optimizer

import handwriting_ai.training.calibration.measure as meas
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.dataset import AugmentConfig, DataLoaderConfig, PreprocessDataset
from handwriting_ai.training.safety import (
    MemoryGuardConfig,
    reset_memory_guard,
    set_memory_guard_config,
)

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


class _BatchIteratorProto(Protocol):
    """Protocol for batch iterator."""

    def __iter__(self) -> _BatchIteratorProto: ...
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]: ...


class _BatchIterableProto(Protocol):
    """Protocol for batch iterable matching measure._BatchIterable."""

    def __iter__(self) -> _BatchIteratorProto: ...


class _StubBatchIterator:
    """Stub iterator that yields nothing."""

    def __iter__(self) -> _StubBatchIterator:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise StopIteration


class _StubLoader:
    """Stub loader implementing _BatchIterableProto for test stubs."""

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


def test_headroom_expansion_raises_upper_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: dataset and candidate with a small initial cap
    ds = PreprocessDataset(_Base(), _CFG)
    cand: Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 2,
    }

    # Stub loader to avoid real DataLoader construction
    def _fake_loader(ds: PreprocessDataset, cfg: DataLoaderConfig) -> _StubLoader:
        _ = (ds, cfg)
        return _StubLoader()

    monkeypatch.setattr(meas, "_safe_loader", _fake_loader, raising=True)

    # Stub measurement: succeed while bs < 4 with low peak (10%), then fail at >= 4
    calls: dict[str, list[int]] = {"seen": []}

    def _stub_measure_training(
        ds_len: int,
        _loader: _BatchIterableProto,
        k: int,
        *,
        device: torch.device,
        batch_size_hint: int,
        model: Module,
        opt: Optimizer,
    ) -> tuple[float, float, float, bool]:
        _ = (ds_len, _loader, k, device, model, opt)
        calls["seen"].append(int(batch_size_hint))
        if batch_size_hint >= 4:
            return 0.0, 0.0, 75.0, True  # exceeded near guard, no headroom
        return 100.0, 1.0, 10.0, False  # low peak -> triggers expansion at cap

    monkeypatch.setattr(meas, "_measure_training", _stub_measure_training, raising=True)

    # Enable memory guard with a positive threshold so headroom expansion
    # logic is active for this test.
    set_memory_guard_config(
        MemoryGuardConfig(enabled=True, threshold_percent=92.0, required_consecutive=1)
    )
    reset_memory_guard()
    try:
        # Act
        res = meas._measure_candidate(ds, cand, samples=2)
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


def test_headroom_expansion_requires_guard_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: dataset and candidate with a small initial cap
    ds = PreprocessDataset(_Base(), _CFG)
    cand: Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 2,
    }

    def _fake_loader(ds: PreprocessDataset, cfg: DataLoaderConfig) -> _StubLoader:
        _ = (ds, cfg)
        return _StubLoader()

    monkeypatch.setattr(meas, "_safe_loader", _fake_loader, raising=True)

    calls: dict[str, list[int]] = {"seen": []}

    def _stub_measure_training(
        ds_len: int,
        _loader: _BatchIterableProto,
        k: int,
        *,
        device: torch.device,
        batch_size_hint: int,
        model: Module,
        opt: Optimizer,
    ) -> tuple[float, float, float, bool]:
        _ = (ds_len, _loader, k, device, model, opt)
        calls["seen"].append(int(batch_size_hint))
        # Always succeed with a low peak usage to exercise headroom logic.
        return 100.0, 1.0, 10.0, False

    monkeypatch.setattr(meas, "_measure_training", _stub_measure_training, raising=True)

    # Disable guard so headroom expansion should be inactive even with low peak usage.
    set_memory_guard_config(
        MemoryGuardConfig(enabled=False, threshold_percent=92.0, required_consecutive=1)
    )
    reset_memory_guard()
    try:
        res = meas._measure_candidate(ds, cand, samples=2)
    finally:
        set_memory_guard_config(
            MemoryGuardConfig(enabled=False, threshold_percent=0.0, required_consecutive=0)
        )
        reset_memory_guard()

    # With guard disabled we should not expand beyond the initial cap.
    assert res["batch_size"] <= cand["batch_size"]
    assert max(calls["seen"]) <= cand["batch_size"]
