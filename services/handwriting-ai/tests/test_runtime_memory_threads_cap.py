from __future__ import annotations

import pytest

import handwriting_ai.training.runtime as rt
from handwriting_ai.training.resources import ResourceLimits
from handwriting_ai.training.train_config import default_train_config


def test_threads_capped_under_low_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force detect_resource_limits to return low memory with high optimal_threads
    limits: ResourceLimits = {
        "cpu_cores": 4,
        "memory_bytes": 1 * 1024 * 1024 * 1024,
        "optimal_threads": 8,
        "optimal_workers": 0,
        "max_batch_size": None,
    }
    monkeypatch.setattr(rt, "detect_resource_limits", lambda: limits)

    cfg = default_train_config(threads=16, batch_size=64)
    ec, got_limits = rt.build_effective_config(cfg)
    assert got_limits == limits
    # Capped to 2 under <2GB
    assert ec["intra_threads"] == 2


def test_threads_capped_mid_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    limits: ResourceLimits = {
        "cpu_cores": 8,
        "memory_bytes": 3 * 1024 * 1024 * 1024,
        "optimal_threads": 8,
        "optimal_workers": 1,
        "max_batch_size": None,
    }
    monkeypatch.setattr(rt, "detect_resource_limits", lambda: limits)
    cfg = default_train_config(threads=16, batch_size=64)
    ec, _ = rt.build_effective_config(cfg)
    # Capped to <=4 when 2-4GB
    assert ec["intra_threads"] <= 4


def test_threads_not_capped_high_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    limits: ResourceLimits = {
        "cpu_cores": 16,
        "memory_bytes": 8 * 1024 * 1024 * 1024,
        "optimal_threads": 8,
        "optimal_workers": 2,
        "max_batch_size": None,
    }
    monkeypatch.setattr(rt, "detect_resource_limits", lambda: limits)
    cfg = default_train_config(threads=16, batch_size=64)
    ec, _ = rt.build_effective_config(cfg)
    assert ec["intra_threads"] == 16
