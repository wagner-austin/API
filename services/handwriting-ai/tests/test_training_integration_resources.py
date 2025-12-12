from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image
from torch.utils.data import Dataset

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import ResourceLimitsDict
from handwriting_ai.training import safety as _safety
from handwriting_ai.training.train_config import TrainConfig, default_train_config


class MnistRawWriter(Protocol):
    def __call__(self, root: Path, n: int = 8) -> None: ...


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (28, 28), 0)
        return img, int(idx % 10)


def _cfg(tmp: Path) -> TrainConfig:
    return default_train_config(
        data_root=tmp / "data",
        out_dir=tmp / "out",
        model_id="m",
        epochs=1,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-2,
        seed=123,
        device="cpu",
        optim="adamw",
        scheduler="none",
        step_size=1,
        gamma=0.5,
        min_lr=1e-5,
        patience=0,
        min_delta=5e-4,
        threads=0,
        augment=False,
        aug_rotate=0.0,
        aug_translate=0.0,
    )


def test_train_uses_resource_limits(tmp_path: Path, write_mnist_raw: MnistRawWriter) -> None:
    # Force resource limits to a known state
    def _fake_detect() -> ResourceLimitsDict:
        return {
            "cpu_cores": 4,
            "memory_bytes": 1024 * 1024 * 1024,
            "optimal_threads": 2,
            "optimal_workers": 0,
            "max_batch_size": 4,
        }

    _test_hooks.detect_resource_limits = _fake_detect

    # Spy limit_thread_pools to ensure it is called with our intra threads
    calls = {"n": 0, "limit": 0}

    @contextmanager
    def _fake_limit_thread_pools(*, limits: int) -> Generator[None, None, None]:
        calls["n"] += 1
        calls["limit"] = int(limits)
        yield

    # Also neutralize memory guard aborts at the loops layer for this
    # integration test to avoid coupling to host-level memory pressure while
    # still exercising resource limit wiring and guard configuration.
    _safety.set_memory_guard_config(
        _safety.MemoryGuardConfig(enabled=True, threshold_percent=90.0, required_consecutive=3)
    )

    def _no_guard() -> bool:
        return False

    _test_hooks.on_batch_check = _no_guard
    _test_hooks.limit_thread_pools = _fake_limit_thread_pools

    cfg = _cfg(tmp_path)
    base = _TinyBase()
    # Ensure MNIST raw files exist under cfg["data_root"]
    write_mnist_raw(cfg["data_root"], n=8)
    import handwriting_ai.training.mnist_train as mt

    result = mt.train_with_config(cfg, (base, base))
    assert result["state_dict"]  # has state dict
    # Effective batch capped to 4 per ResourceLimits and recorded in metadata
    assert result["metadata"]["batch_size"] == 4
    # Torch threads configured via calibration (may differ from limits)
    t = torch.get_num_threads()
    assert t >= 1
    assert calls["n"] >= 1 and calls["limit"] == t
