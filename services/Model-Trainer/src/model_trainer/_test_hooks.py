"""Test hooks for worker entry - allows injecting test runner before module load."""

from __future__ import annotations

from typing import Protocol

from platform_workers.rq_harness import WorkerConfig


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


# Module-level injectable runner for testing.
# Tests set this BEFORE running worker_entry as __main__.
# Because this is a separate module, it persists across runpy.run_module.
test_runner: WorkerRunnerProtocol | None = None
