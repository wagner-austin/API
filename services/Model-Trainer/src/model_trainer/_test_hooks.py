"""Test hooks for worker entry - allows injecting test runner before module load."""

from __future__ import annotations

from typing import Protocol

from platform_workers.rq_harness import WorkerConfig, run_rq_worker


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


# Hook for the worker runner, bound to the real RQ runner. Tests rebind
# it BEFORE running worker_entry as __main__; because this is a separate
# module, the binding persists across runpy.run_module.
worker_runner: WorkerRunnerProtocol = run_rq_worker
