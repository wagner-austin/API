from __future__ import annotations

import random as _random
from pathlib import Path

import pytest

from model_trainer.core.services.data import corpus as corpus_mod


class _StubRandom:
    def __init__(self: _StubRandom, _seed: int) -> None:
        self._n = 0

    def randint(self: _StubRandom, a: int, b: int) -> int:
        # First call -> choose a (<= k), second call -> choose b (> k), then alternate
        self._n += 1
        return a if self._n % 2 == 1 else b


def test_sample_lines_exercises_both_replace_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Create file with several lines
    fp = tmp_path / "data.txt"
    fp.write_text("\n".join(["l1", "l2", "l3", "l4"]), encoding="utf-8")

    # Monkeypatch the Random class used by sample_lines to our stub
    monkeypatch.setattr(_random, "Random", _StubRandom, raising=True)

    out = corpus_mod.sample_lines([str(fp)], 1, seed=123)
    # Reservoir size 1 is preserved, exact value not asserted
    assert len(out) == 1
