from __future__ import annotations

from collections.abc import Generator

import torch
from PIL import Image

from handwriting_ai import _test_hooks
from handwriting_ai._hook_protocols_training import MemorySnapshotDict
from handwriting_ai.training.calibration import measure as _measure
from handwriting_ai.training.calibration._types import Candidate
from handwriting_ai.training.dataset import AugmentConfig, PreprocessDataset
from handwriting_ai.training.safety import (
    MemoryGuardConfig,
    reset_memory_guard,
    set_memory_guard_config,
)


class _FakeMNIST:
    def __init__(self, n: int = 64) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        # 28x28 grayscale image, label 0
        img = Image.new("L", (28, 28), color=0)
        return img, 0


def test_measure_candidate_basic_runs() -> None:
    # Eight images at a requested batch of four take the same binary search as
    # 64 at 32 did, at a sixteenth of the ResNet-18 compute: 15 backward passes
    # at batch 32 measured 28.7 s of one thread on the hub (API task 8bbe083b).
    base = _FakeMNIST(8)
    cfg: AugmentConfig = {
        "batch_size": 4,
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
        "morph_kernel_px": 3,
    }
    ds = PreprocessDataset(base, cfg)
    cand: Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 4,
    }
    # A disabled guard with a 92 percent threshold never backs off below the
    # cap and never expands past it, so the search must settle on the
    # requested batch whatever guard an earlier test in this worker left.
    set_memory_guard_config(
        MemoryGuardConfig(enabled=False, threshold_percent=92.0, required_consecutive=3)
    )
    reset_memory_guard()
    res = _measure._measure_candidate(ds, cand, samples=2)
    assert res["batch_size"] == 4
    assert res["samples_per_sec"] > 0.0
    assert res["p95_ms"] > 0.0


def test_measure_training_zero_length_loader() -> None:
    # Call internal helper with an empty iterator to exercise the early-return path

    import torch as _t

    def _empty_loader() -> Generator[tuple[_t.Tensor, _t.Tensor], None, None]:
        if False:
            yield _t.zeros((1, 1, 28, 28)), _t.zeros((1,), dtype=_t.long)

    # Build a minimal model and optimizer for the new signature
    from handwriting_ai.training.optim import (
        build_optimizer_and_scheduler as _build_optim,
    )
    from handwriting_ai.training.optim import (
        default_optim_config,
    )
    from handwriting_ai.training.train_utils import _build_model as _build_train_model

    model = _build_train_model()
    opt, _sch = _build_optim(model, default_optim_config())

    sps_f: float
    p95_f: float
    peak_f: float
    exceeded_b: bool
    sps_f, p95_f, peak_f, exceeded_b = _measure._measure_training(
        ds_len=0,
        loader=_empty_loader(),
        k=2,
        device=_t.device("cpu"),
        batch_size_hint=16,
        model=model,
        opt=opt,
    )
    assert sps_f == 0.0 and p95_f == 0.0 and peak_f == 0.0 and exceeded_b is False


def test_measure_candidate_exceeds_threshold_backoff() -> None:
    # Force threshold to 0 so any usage triggers backoff path in binary search
    set_memory_guard_config(
        MemoryGuardConfig(enabled=True, threshold_percent=0.0, required_consecutive=1)
    )
    reset_memory_guard()
    try:
        base = _FakeMNIST(8)
        cfg: AugmentConfig = {
            "batch_size": 8,
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
            "morph_kernel_px": 3,
        }
        ds = PreprocessDataset(base, cfg)
        cand: Candidate = {
            "intra_threads": 1,
            "interop_threads": None,
            "num_workers": 0,
            "batch_size": 8,
        }
        res = _measure._measure_candidate(ds, cand, samples=1)
        # Expect the algorithm to back off from the initial size due to threshold=0.0
        assert res["batch_size"] < 8
    finally:
        # Restore default guard disabled to avoid cross-test effects
        set_memory_guard_config(
            MemoryGuardConfig(enabled=False, threshold_percent=92.0, required_consecutive=3)
        )
        reset_memory_guard()


def test_measure_loader_break_on_exhaustion() -> None:
    # Loader yields fewer batches than requested n_batches, triggers inner break path

    import torch as _t

    from handwriting_ai.training import calibrate as cal

    def _loader_once() -> Generator[tuple[_t.Tensor, _t.Tensor], None, None]:
        x = _t.zeros((1, 1, 28, 28), dtype=_t.float32)
        y = _t.zeros((1,), dtype=_t.long)
        yield x, y

    sps, p95 = cal._measure_loader(100, _loader_once(), 3, batch_size_hint=2)
    assert sps >= 0.0 and p95 >= 0.0


def _snapshot_at(percent: float) -> MemorySnapshotDict:
    return {
        "main_process": {"pid": 1, "rss_bytes": 0},
        "workers": (),
        "cgroup_usage": {"usage_bytes": 0, "limit_bytes": 0, "percent": percent},
        "cgroup_breakdown": {
            "anon_bytes": 0,
            "file_bytes": 0,
            "kernel_bytes": 0,
            "slab_bytes": 0,
        },
    }


def test_measure_training_keeps_the_highest_memory_reading() -> None:
    """The peak is the highest reading across the measured batches, not the last.

    Two measured batches read 50 then 40 percent, so the second is not a new
    peak and the reported peak must stay at 50. A disabled guard at 92
    percent keeps either reading from counting as exceeded.
    """
    readings = [50.0, 40.0]

    def _next_snapshot() -> MemorySnapshotDict:
        return _snapshot_at(readings.pop(0))

    _test_hooks.get_memory_snapshot = _next_snapshot
    set_memory_guard_config(
        MemoryGuardConfig(enabled=False, threshold_percent=92.0, required_consecutive=3)
    )
    reset_memory_guard()

    def _three_batches() -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        for _ in range(3):
            yield torch.zeros((1, 1, 28, 28)), torch.zeros((1,), dtype=torch.long)

    from handwriting_ai.training.optim import build_optimizer_and_scheduler, default_optim_config
    from handwriting_ai.training.train_utils import _build_model

    model = _build_model()
    opt, _sch = build_optimizer_and_scheduler(model, default_optim_config())
    # One warm-up batch, then k=2 measured batches, each followed by one reading
    sps, p95, peak, exceeded = _measure._measure_training(
        ds_len=3,
        loader=_three_batches(),
        k=2,
        device=torch.device("cpu"),
        batch_size_hint=1,
        model=model,
        opt=opt,
    )
    assert readings == []
    assert peak == 50.0
    assert exceeded is False
    assert sps > 0.0 and p95 > 0.0
