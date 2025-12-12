from __future__ import annotations

from PIL import Image

from handwriting_ai import _test_hooks
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.measure import _measure_candidate
from handwriting_ai.training.dataset import PreprocessDataset
from handwriting_ai.training.train_config import default_train_config


class _Base:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), 0


def test_measure_candidate_calls_gc_collect() -> None:
    calls = {"n": 0}

    def _fake_gc_collect() -> int:
        calls["n"] += 1
        return 0

    _test_hooks.gc_collect = _fake_gc_collect

    ds = PreprocessDataset(_Base(), default_train_config(batch_size=1))
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=1)

    res = _measure_candidate(ds, cand, samples=1)
    assert res["batch_size"] >= 1
    assert calls["n"] >= 1
