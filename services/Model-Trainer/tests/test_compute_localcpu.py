from __future__ import annotations

from model_trainer.core.contracts.compute import LocalCPUProvider


def test_local_cpu_provider_clamps_threads_and_env() -> None:
    p = LocalCPUProvider(threads_count=0)
    assert p.kind() == "local-cpu"
    assert p.threads() >= 1
    env = p.env()
    assert env["OMP_NUM_THREADS"] == str(p.threads())
    assert env["MKL_NUM_THREADS"] == str(p.threads())
