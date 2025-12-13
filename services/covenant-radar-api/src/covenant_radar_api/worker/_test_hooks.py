"""Test hooks for worker components and ML registry injection.

Production code uses real implementations; tests can override these module-level
symbols to inject fakes without conditionals in core logic.
"""

from __future__ import annotations

from typing import Protocol

from covenant_ml.backends.registry import ClassifierRegistry, default_registry
from platform_workers.rq_harness import WorkerConfig


class WorkerRunnerProtocol(Protocol):
    def __call__(self, config: WorkerConfig) -> None: ...


test_runner: WorkerRunnerProtocol | None = None


class RegistryFactory(Protocol):
    def __call__(self) -> ClassifierRegistry: ...


registry_factory: RegistryFactory = default_registry


__all__ = ["RegistryFactory", "WorkerRunnerProtocol", "registry_factory", "test_runner"]
