from __future__ import annotations

import pytest

import handwriting_ai.training.runtime as rt
from handwriting_ai.training.resources import ResourceLimits
from handwriting_ai.training.train_config import default_train_config


def test_loader_prefetch_reduced_under_low_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    limits: ResourceLimits = {
        "cpu_cores": 2,
        "memory_bytes": 1 * 1024 * 1024 * 1024,
        "optimal_threads": 2,
        "optimal_workers": 0,
        "max_batch_size": 64,
    }
    monkeypatch.setattr(rt, "detect_resource_limits", lambda: limits, raising=True)
    cfg = default_train_config(threads=0, batch_size=64)
    ec, _ = rt.build_effective_config(cfg)
    assert ec["loader_cfg"]["prefetch_factor"] == 1
    assert ec["loader_cfg"]["num_workers"] == 0
    assert ec["loader_cfg"]["persistent_workers"] is False
