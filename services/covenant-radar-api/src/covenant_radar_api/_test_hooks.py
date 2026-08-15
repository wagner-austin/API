"""Test hooks for worker entry - allows injecting a runner before module load."""

from __future__ import annotations

from typing import Protocol

from platform_workers.rq_harness import WorkerConfig, run_rq_worker


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


# The worker runner, bound to the real implementation. Tests rebind it before
# running worker_entry as __main__; because this is a separate module, the
# rebinding persists across runpy.run_module.
worker_runner: WorkerRunnerProtocol = run_rq_worker
