from __future__ import annotations

from multiprocessing.context import BaseContext
from typing import Protocol

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

import handwriting_ai.training.calibration.measure as meas
from handwriting_ai.training.dataset import AugmentConfig, DataLoaderConfig, PreprocessDataset

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


class _DataLoaderArgsProto(Protocol):
    """Protocol for captured DataLoader kwargs."""

    multiprocessing_context: BaseContext | None


class _StubBatchIterator:
    """Stub iterator for test DataLoader."""

    def __iter__(self) -> _StubBatchIterator:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise StopIteration


class _StubDataLoader:
    """Stub DataLoader for tests."""

    def __init__(
        self,
        dataset: Dataset[tuple[torch.Tensor, int]],
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        prefetch_factor: int | None,
        persistent_workers: bool,
        multiprocessing_context: BaseContext | None,
    ) -> None:
        self._ctx = multiprocessing_context

    def __iter__(self) -> _StubBatchIterator:
        return _StubBatchIterator()


class _Base:
    def __len__(self) -> int:
        return 4

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


def test_safe_loader_sets_context_for_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub DataLoader to capture multiprocessing_context
    captured: dict[str, BaseContext | None] = {}

    def _stub_loader(
        dataset: Dataset[tuple[torch.Tensor, int]],
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        prefetch_factor: int | None,
        persistent_workers: bool,
        multiprocessing_context: BaseContext | None,
    ) -> _StubDataLoader:
        captured["ctx"] = multiprocessing_context
        return _StubDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        )

    monkeypatch.setattr(meas, "DataLoader", _stub_loader, raising=True)

    ds = PreprocessDataset(_Base(), _CFG)
    # No workers -> ctx is None
    meas._safe_loader(
        ds,
        DataLoaderConfig(
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
        ),
    )
    assert captured["ctx"] is None
    # With workers -> ctx is not None
    meas._safe_loader(
        ds,
        DataLoaderConfig(
            batch_size=1,
            num_workers=1,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
        ),
    )
    if captured["ctx"] is None:
        raise AssertionError("expected context")


class _FakeContext:
    """Fake multiprocessing context for tests."""

    def __init__(self, method: str | None) -> None:
        self.method = method


def test_resolve_worker_context_prefers_forkserver(monkeypatch: pytest.MonkeyPatch) -> None:
    """POSIX branch prefers forkserver over spawn when available."""

    class _FakeMP:
        def __init__(self) -> None:
            self.methods = ["forkserver", "spawn"]
            self.seen: list[str | None] = []

        def get_all_start_methods(self) -> list[str]:
            return list(self.methods)

        def get_context(self, method: str | None) -> _FakeContext:
            self.seen.append(method)
            return _FakeContext(method)

    class _FakeOS:
        def __init__(self) -> None:
            self.name = "posix"

    fake_mp = _FakeMP()
    fake_os = _FakeOS()
    monkeypatch.setattr("handwriting_ai.training.calibration.measure._os", fake_os, raising=True)
    monkeypatch.setattr("handwriting_ai.training.calibration.measure._mp", fake_mp, raising=True)

    ctx = meas._resolve_worker_context(1)
    if ctx is None:
        raise AssertionError("expected ctx")
    assert fake_mp.seen == ["forkserver"]


def test_resolve_worker_context_no_supported_method(monkeypatch: pytest.MonkeyPatch) -> None:
    """POSIX branch returns None when no forkserver/spawn methods are available."""

    class _FakeMP:
        def __init__(self) -> None:
            self.seen: list[str | None] = []

        def get_all_start_methods(self) -> list[str]:
            return []

        def get_context(self, method: str | None) -> _FakeContext:
            self.seen.append(method)
            return _FakeContext(method)

    class _FakeOS:
        def __init__(self) -> None:
            self.name = "posix"

    fake_mp = _FakeMP()
    fake_os = _FakeOS()
    monkeypatch.setattr("handwriting_ai.training.calibration.measure._os", fake_os, raising=True)
    monkeypatch.setattr("handwriting_ai.training.calibration.measure._mp", fake_mp, raising=True)

    ctx = meas._resolve_worker_context(1)
    assert ctx is None
    assert fake_mp.seen == []
