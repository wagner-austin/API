"""Test hooks for worker entry - allows injecting test runner before module load."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from platform_core.config import _optional_env_str
from platform_workers.rq_harness import WorkerConfig, run_rq_worker


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


def _default_get_env(key: str) -> str | None:
    """Production implementation - reads from os.environ."""
    return _optional_env_str(key)


# Hook for the worker runner, bound to the real RQ runner. Tests rebind it
# BEFORE running worker_entry as __main__; because this is a separate module,
# the binding persists across runpy.run_module.
worker_runner: WorkerRunnerProtocol = run_rq_worker

# Hook for environment variable access. Tests can override to provide fake values.
get_env: Callable[[str], str | None] = _default_get_env
