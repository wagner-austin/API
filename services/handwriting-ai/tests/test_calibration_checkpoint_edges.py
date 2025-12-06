from __future__ import annotations

from pathlib import Path

import pytest

from handwriting_ai.training.calibration.checkpoint import (
    CalibrationCheckpoint,
    CalibrationStage,
    read_checkpoint,
    write_checkpoint,
)
from handwriting_ai.training.calibration.measure import CalibrationResult


def test_read_checkpoint_invalid_root_raises(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        _ = read_checkpoint(p)


def test_write_checkpoint_and_read_back(tmp_path: Path) -> None:
    res: CalibrationResult = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 1,
        "samples_per_sec": 1.0,
        "p95_ms": 2.0,
    }
    ckpt: CalibrationCheckpoint = {
        "stage": CalibrationStage.B,
        "index": 1,
        "results": [res],
        "shortlist": None,
        "seed": None,
    }
    p = tmp_path / "ckpt.json"
    write_checkpoint(p, ckpt)
    loaded = read_checkpoint(p)
    if loaded is None:
        raise AssertionError("expected checkpoint")
    assert loaded["stage"] == CalibrationStage.B
    assert loaded["results"][0]["intra_threads"] == 1
