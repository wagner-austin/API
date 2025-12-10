"""Test hooks for worker entry - allows injecting test runner before module load."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from platform_core.config import _optional_env_str
from platform_workers.rq_harness import WorkerConfig


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


def _default_get_env(key: str) -> str | None:
    """Production implementation - reads from os.environ."""
    return _optional_env_str(key)


# Module-level injectable runner for testing.
# Tests set this BEFORE running worker_entry as __main__.
# Because this is a separate module, it persists across runpy.run_module.
test_runner: WorkerRunnerProtocol | None = None

# Hook for environment variable access. Tests can override to provide fake values.
get_env: Callable[[str], str | None] = _default_get_env
