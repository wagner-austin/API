"""Hooks for container factories - production defaults, tests override.

Production code initializes these to real implementations at module level.
Tests replace them with fakes before exercising the code under test.
No conditionals needed - just call the hook directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from platform_workers.redis import (
    RedisStrProto,
    _RedisBytesClient,
    redis_for_kv,
    redis_raw_for_rq,
)
from platform_workers.rq_harness import RQClientQueue, RQRetryLike, rq_queue, rq_retry

from art_trainer.core.config.settings import Settings

# ============================================================================
# Redis and RQ infrastructure hooks
# ============================================================================


class KVStoreFactoryProto(Protocol):
    """Protocol for redis_for_kv factory."""

    def __call__(self, url: str) -> RedisStrProto:
        """Create Redis client from URL.

        Args:
            url: Redis connection URL.

        Returns:
            Redis client for string operations.
        """
        ...


class RQConnectionFactoryProto(Protocol):
    """Protocol for redis_raw_for_rq factory."""

    def __call__(self, url: str) -> _RedisBytesClient:
        """Create Redis RQ client from URL.

        Args:
            url: Redis connection URL.

        Returns:
            Redis client for RQ operations.
        """
        ...


class RQQueueFactoryProto(Protocol):
    """Protocol for rq_queue factory."""

    def __call__(self, name: str, connection: _RedisBytesClient) -> RQClientQueue:
        """Create RQ queue from name and connection.

        Args:
            name: Queue name.
            connection: Redis connection.

        Returns:
            RQ queue instance.
        """
        ...


class RQRetryFactoryProto(Protocol):
    """Protocol for rq_retry factory."""

    def __call__(self, *, max_retries: int, intervals: list[int]) -> RQRetryLike:
        """Create RQ retry from max_retries and intervals.

        Args:
            max_retries: Maximum number of retries.
            intervals: Retry intervals in seconds.

        Returns:
            RQ retry configuration.
        """
        ...


class LoadSettingsProto(Protocol):
    """Protocol for load_settings factory."""

    def __call__(self) -> Settings:
        """Load settings.

        Returns:
            Application settings.
        """
        ...


# ============================================================================
# Path and file system hooks
# ============================================================================


class LoraOutputDirProto(Protocol):
    """Protocol for lora_output_dir hook."""

    def __call__(self, settings: Settings, job_id: str) -> Path:
        """Get LoRA output directory path.

        Args:
            settings: Application settings.
            job_id: Job identifier.

        Returns:
            Path to output directory.
        """
        ...


class ShutilWhichProto(Protocol):
    """Protocol for shutil_which hook."""

    def __call__(self, cmd: str) -> str | None:
        """Find command on PATH, return path or None.

        Args:
            cmd: Command to find.

        Returns:
            Path to command or None if not found.
        """
        ...


# ============================================================================
# Guard script hooks for testing
# ============================================================================


class FindMonorepoRootProto(Protocol):
    """Protocol for _find_monorepo_root hook."""

    def __call__(self, start: Path) -> Path:
        """Find the monorepo root directory.

        Args:
            start: Starting directory.

        Returns:
            Path to monorepo root.
        """
        ...


class RunForProjectProto(Protocol):
    """Protocol for run_for_project hook."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guards for a project.

        Args:
            monorepo_root: Path to monorepo root.
            project_root: Path to project root.

        Returns:
            Exit code.
        """
        ...


class LoadOrchestratorProto(Protocol):
    """Protocol for _load_orchestrator hook."""

    def __call__(self, monorepo_root: Path) -> RunForProjectProto:
        """Load the guard orchestrator.

        Args:
            monorepo_root: Path to monorepo root.

        Returns:
            Function to run guards for a project.
        """
        ...


# ============================================================================
# Default implementations
# ============================================================================


def _default_rq_queue(name: str, connection: _RedisBytesClient) -> RQClientQueue:
    """Production rq_queue - used as default hook."""
    return rq_queue(name, connection)


def _default_rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
    """Production rq_retry - used as default hook."""
    return rq_retry(max_retries=max_retries, intervals=intervals)


def _default_load_settings() -> Settings:
    """Production load_settings - used as default hook."""
    from art_trainer.core.config.settings import load_settings as _load

    return _load()


def _default_lora_output_dir(settings: Settings, job_id: str) -> Path:
    """Production lora_output_dir - used as default hook."""
    from art_trainer.core.infra.paths import lora_output_dir as _lora_output_dir

    return _lora_output_dir(settings, job_id)


def _default_shutil_which(cmd: str) -> str | None:
    """Production shutil_which - used as default hook."""
    import shutil

    return shutil.which(cmd)


# ============================================================================
# Factory hooks - initialized to production implementations.
# Tests replace these with fakes before calling container code.
# ============================================================================

kv_store_factory: KVStoreFactoryProto = redis_for_kv
rq_connection_factory: RQConnectionFactoryProto = redis_raw_for_rq
rq_queue_factory: RQQueueFactoryProto = _default_rq_queue
rq_retry_factory: RQRetryFactoryProto = _default_rq_retry
load_settings: LoadSettingsProto = _default_load_settings
lora_output_dir: LoraOutputDirProto = _default_lora_output_dir
shutil_which: ShutilWhichProto = _default_shutil_which

# Guard hooks - None means use default behavior (production implementation)
guard_find_monorepo_root: FindMonorepoRootProto | None = None
guard_load_orchestrator: LoadOrchestratorProto | None = None


__all__ = [
    "FindMonorepoRootProto",
    "KVStoreFactoryProto",
    "LoadOrchestratorProto",
    "LoadSettingsProto",
    "LoraOutputDirProto",
    "RQConnectionFactoryProto",
    "RQQueueFactoryProto",
    "RQRetryFactoryProto",
    "RunForProjectProto",
    "ShutilWhichProto",
    "guard_find_monorepo_root",
    "guard_load_orchestrator",
    "kv_store_factory",
    "load_settings",
    "lora_output_dir",
    "rq_connection_factory",
    "rq_queue_factory",
    "rq_retry_factory",
    "shutil_which",
]
