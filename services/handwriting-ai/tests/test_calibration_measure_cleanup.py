from __future__ import annotations

from collections.abc import Generator
from typing import Protocol

import pytest
import torch as _t
from PIL import Image

import handwriting_ai.training.calibration.measure as meas
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.dataset import PreprocessDataset
from handwriting_ai.training.train_config import default_train_config


def _one_batch_loader() -> Generator[tuple[_t.Tensor, _t.Tensor], None, None]:
    x = _t.zeros((1, 1, 28, 28), dtype=_t.float32)
    y = _t.zeros((1,), dtype=_t.long)
    yield x, y


class _Base:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), 0


class _FakeGc(Protocol):
    def collect(self) -> int: ...


class _FakeGcImpl:
    def __init__(self, calls: dict[str, int]) -> None:
        self._calls = calls

    def collect(self) -> int:
        self._calls["n"] += 1
        return 0


def test_measure_candidate_calls_gc_collect(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    fake_gc: _FakeGc = _FakeGcImpl(calls)
    monkeypatch.setattr(meas, "_gc", fake_gc, raising=True)

    ds = PreprocessDataset(_Base(), default_train_config(batch_size=1))
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=1)

    res = meas._measure_candidate(ds, cand, samples=1)
    assert res["batch_size"] >= 1
    assert calls["n"] >= 1
