from __future__ import annotations

import pytest

from handwriting_ai.training.calibration import measure as meas


def test_resolve_worker_context_returns_none_for_zero_workers() -> None:
    """No workers means no multiprocessing context at all."""
    assert meas._resolve_worker_context(0) is None


@pytest.mark.parametrize("num_workers", [1, 4])
def test_resolve_worker_context_is_spawn_for_any_worker(num_workers: int) -> None:
    """Spawn on every platform: the candidate stays the workers' parent, so
    torch's ManagerWatchdog sees it exit; a forkserver would outlive a killed
    candidate and strand its workers (board task 49899fd2)."""
    assert meas._resolve_worker_context(num_workers) == "spawn"
