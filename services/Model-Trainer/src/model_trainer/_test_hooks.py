"""Test hooks for worker entry - allows injecting test runner before module load."""

from __future__ import annotations

from typing import Protocol

from platform_workers.rq_harness import WorkerConfig, run_single_job_rq_worker


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


# Hook for the worker runner, bound to the single-job RQ runner: the worker
# process exits after each job and the container supervisor restarts it, so
# every training run gets a fresh CUDA context. A long-lived worker shares
# one context across runs, and one asynchronous CUDA fault (observed twice on
# the WSL2 GPU stack after ~7h of sustained training) poisons that context,
# instantly killing every subsequent job and pinning GPU memory until the
# process dies. Tests rebind this BEFORE running worker_entry as __main__;
# because this is a separate module, the binding persists across
# runpy.run_module.
worker_runner: WorkerRunnerProtocol = run_single_job_rq_worker
