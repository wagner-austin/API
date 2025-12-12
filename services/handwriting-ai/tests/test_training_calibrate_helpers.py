from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from PIL import Image
from platform_core.json_utils import JSONTypeError

import handwriting_ai.training.calibrate as cal
from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import (
    BatchIterableProtocol,
    DataLoaderConfigProtocol,
    PreprocessDatasetProtocol,
)
from handwriting_ai.training.dataset import (
    AugmentConfig,
    MNISTLike,
    PreprocessDataset,
)
from handwriting_ai.training.resources import ResourceLimits


def test_as_obj_dict_and_number_parsers() -> None:
    assert cal._decode_obj_dict(123) is None
    # Test with valid string-keyed dict (JSON only allows string keys)
    d = cal._decode_obj_dict({"a": 7, "b": "x"})
    assert d == {"a": 7, "b": "x"}

    assert cal._decode_int({}, "k", 5) == 5
    assert cal._decode_int({"k": True}, "k", 5) == 5
    assert cal._decode_int({"k": "7"}, "k", 5) == 7
    # Should raise JSONTypeError for non-numeric string
    with pytest.raises(JSONTypeError):
        cal._decode_int({"k": "bad"}, "k", 5)

    assert cal._decode_float({}, "k", 1.5) == 1.5
    assert cal._decode_float({"k": False}, "k", 1.5) == 1.5
    assert cal._decode_float({"k": "2.5"}, "k", 1.5) == 2.5
    # Should raise JSONTypeError for non-numeric string
    with pytest.raises(JSONTypeError):
        cal._decode_float({"k": "not"}, "k", 1.5)


def test_read_cache_decode_and_missing_raises_after_logging(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text("not-json", encoding="utf-8")
    # Should raise on invalid JSON
    with pytest.raises((OSError, ValueError, TypeError)):
        cal._read_cache(p)

    p2 = tmp_path / "c2.json"
    p2.write_text("{}", encoding="utf-8")
    # Empty dict should raise (missing signature)
    with pytest.raises((OSError, ValueError, TypeError)):
        cal._read_cache(p2)

    # Non-dict JSON should raise
    p3 = tmp_path / "c3.json"
    p3.write_text("[]", encoding="utf-8")
    with pytest.raises((OSError, ValueError, TypeError)):
        cal._read_cache(p3)


def test_valid_cache_mismatch_and_expire() -> None:
    sig: cal.CalibrationSignature = {
        "cpu_cores": 2,
        "mem_bytes": None,
        "os": "x",
        "py": "3",
        "torch": "t",
    }
    res: cal.CalibrationResult = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 1,
        "samples_per_sec": 1.0,
        "p95_ms": 1.0,
    }
    now = 1000.0

    # Use now_ts hook to control time
    _test_hooks.now_ts = lambda: now

    # Mismatch sig
    sig_mismatch: cal.CalibrationSignature = {
        "cpu_cores": 1,
        "mem_bytes": None,
        "os": "x",
        "py": "3",
        "torch": "t",
    }
    assert cal._valid_cache(sig, (sig_mismatch, res, now), 10) is None
    # Expired
    assert cal._valid_cache(sig, (sig, res, now - 3600.0), 10) is None


def test_candidate_workers_zero_cores() -> None:
    limits: ResourceLimits = {
        "cpu_cores": 0,
        "memory_bytes": None,
        "optimal_threads": 1,
        "optimal_workers": 0,
        "max_batch_size": None,
    }
    assert cal._candidate_workers(limits) == [0]


def test_measure_candidate_backoff_raises_after_retries() -> None:
    # First call to _safe_loader raises, subsequent calls also raise to exhaust backoff

    calls = {"n": 0}

    class _Base(MNISTLike):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
            img = Image.new("L", (28, 28), 0)
            return img, idx % 10

    _cfg: AugmentConfig = {
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

    ds = PreprocessDataset(_Base(), _cfg)

    def _safe(
        ds: PreprocessDatasetProtocol, cfg: DataLoaderConfigProtocol
    ) -> BatchIterableProtocol:
        _ = (ds, cfg)  # unused
        calls["n"] += 1
        raise RuntimeError("no memory")

    # Use safe_loader hook
    _test_hooks.safe_loader = _safe

    cand: cal.Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 4,
    }
    # Should raise after exhausting backoff attempts
    with pytest.raises(RuntimeError, match="no memory"):
        cal._measure_candidate(ds, cand, samples=2)


def test_measure_loader_paths() -> None:
    # Single short batch with zero samples triggers fallback branch and p95 fallback
    def _loader_one() -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        x = torch.zeros((0, 1, 28, 28), dtype=torch.float32)
        y = torch.zeros((0,), dtype=torch.int64)
        yield x, y

    sps, p95 = cal._measure_loader(1, _loader_one(), 1, batch_size_hint=4)
    assert p95 >= 0.0 and sps >= 0.0

    # Two batches go through quantiles path
    def _loader_two() -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        y = torch.ones((1,), dtype=torch.int64)
        for _ in range(2):
            yield x, y

    sps2, p952 = cal._measure_loader(4, _loader_two(), 2, batch_size_hint=2)
    assert p952 >= 0.0 and sps2 >= 0.0


def test_candidate_workers_is_generic() -> None:
    # 1 core -> [0]
    limits1: ResourceLimits = {
        "cpu_cores": 1,
        "memory_bytes": None,
        "optimal_threads": 1,
        "optimal_workers": 0,
        "max_batch_size": None,
    }
    # 2 cores -> includes 1
    limits2: ResourceLimits = {
        "cpu_cores": 2,
        "memory_bytes": None,
        "optimal_threads": 2,
        "optimal_workers": 0,
        "max_batch_size": None,
    }
    # 8 cores -> includes 0..4 (bounded)
    limits8: ResourceLimits = {
        "cpu_cores": 8,
        "memory_bytes": None,
        "optimal_threads": 4,
        "optimal_workers": 2,
        "max_batch_size": None,
    }

    w1 = cal._candidate_workers(limits1)
    w2 = cal._candidate_workers(limits2)
    w8 = cal._candidate_workers(limits8)
    assert w1 == [0]
    assert 1 in w2 and 0 in w2
    assert w8[0] == 0 and w8[-1] <= 4


def test_measure_candidate_interop_branch() -> None:
    class _Base(MNISTLike):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
            img = Image.new("L", (28, 28), 0)
            return img, idx % 10

    _cfg: AugmentConfig = {
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

    ds = PreprocessDataset(_Base(), _cfg)
    cand: cal.Candidate = {
        "intra_threads": 1,
        "interop_threads": 1,
        "num_workers": 0,
        "batch_size": 2,
    }
    # Use tiny sample and rely on default _measure_loader implementation
    _ = cal._measure_candidate(ds, cand, samples=1)


def test_measure_candidate_exhausts_backoff_raises() -> None:
    # Always raise to force while-loop to exhaust backoff

    def _safe(
        ds: PreprocessDatasetProtocol, cfg: DataLoaderConfigProtocol
    ) -> BatchIterableProtocol:
        _ = (ds, cfg)  # unused
        raise RuntimeError("no memory")

    # Use safe_loader hook
    _test_hooks.safe_loader = _safe

    class _Base(MNISTLike):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
            return Image.new("L", (28, 28), 0), idx % 10

    _cfg: AugmentConfig = {
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

    ds = PreprocessDataset(_Base(), _cfg)
    cand: cal.Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 1,
    }
    # Should raise after exhausting backoff
    with pytest.raises(RuntimeError, match="no memory"):
        cal._measure_candidate(ds, cand, samples=1)


def test_measure_loader_empty_iterable() -> None:
    def _loader_empty() -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        if False:
            yield (
                torch.zeros((1, 1, 28, 28), dtype=torch.float32),
                torch.zeros((1,), dtype=torch.int64),
            )

    sps, p95 = cal._measure_loader(1, _loader_empty(), 1, batch_size_hint=1)
    assert p95 >= 0.0 and sps >= 0.0
