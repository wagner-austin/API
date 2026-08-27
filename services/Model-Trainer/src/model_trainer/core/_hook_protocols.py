"""Hook protocols and shapes for Model-Trainer dependency injection.

Defaults: :mod:`model_trainer.core._hook_defaults`; the rebindable
bindings stay in :mod:`model_trainer.core._test_hooks`.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Protocol

import httpx
import torch
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.environment_record import HostProbe
from platform_core.json_utils import _JSONInputValue as JSONInputValue
from platform_workers.redis import (
    RedisStrProto,
    _RedisBytesClient,
)
from platform_workers.rq_harness import RQClientQueue, RQRetryLike

from model_trainer.core.config.settings import Settings
from model_trainer.core.services.registries import ModelRegistry
from model_trainer.core.types import LMModelProto, TorchStateValue


class EnvGitCommitProto(Protocol):
    """Protocol for env_git_commit hook."""

    def __call__(self) -> str | None:
        """Read the build-stamped GIT_COMMIT variable, None when unset or empty."""
        ...


class EnvImageDigestProto(Protocol):
    """Protocol for env_image_digest hook."""

    def __call__(self) -> str | None:
        """Read the IMAGE_DIGEST variable, None when unset or empty.

        Distinct from the commit: a commit says which code was built, an
        image digest says which environment ran it. Two runs can share a
        commit and differ in torch, which is exactly the difference a
        fingerprint exists to catch.
        """
        ...


class PkgVersionProto(Protocol):
    """Protocol for pkg_version hook."""

    def __call__(self, name: str) -> str:
        """Get package version by name."""
        ...


class HostProbeProto(Protocol):
    """Protocol for the host_probe hook.

    Separate from :class:`PkgVersionProto` because it answers a different
    question: which MACHINE, rather than which libraries.
    """

    def __call__(self) -> HostProbe:
        """Build the probe that reads this machine's identity.

        Returns:
            The probe, whose fields become the fingerprint's host axis.
        """
        ...


class InstalledVersionProto(Protocol):
    """Protocol for the installed_version hook.

    DISTINCT FROM :class:`PkgVersionProto` AND THAT IS DELIBERATE, not a fork.
    ``pkg_version`` answers "what shall I write in a human-readable manifest"
    and returns ``"unknown"`` for a library that is not installed. A
    fingerprint axis cannot accept that: ``"unknown"`` is a non-empty string,
    so it would pass every validator and then compare EQUAL to any other run
    that also could not find the library, reporting two different
    environments as one. This hook propagates instead.
    """

    def __call__(self, distribution: str) -> str:
        """Read one distribution's resolved version.

        Args:
            distribution: The distribution name.

        Returns:
            Its installed version.

        Raises:
            PackageNotFoundError: When the distribution is not installed.
        """
        ...


class ShutilWhichProto(Protocol):
    """Protocol for shutil_which hook."""

    def __call__(self, cmd: str) -> str | None:
        """Find command on PATH, return path or None."""
        ...


class HttpxClientFactoryProto(Protocol):
    """Protocol for httpx.Client factory.

    Tests inject fake transports by returning httpx.Client(transport=MockTransport(...)).
    Production returns httpx.Client(timeout=timeout_seconds).
    """

    def __call__(self, *, timeout_seconds: float = 30.0) -> httpx.Client:
        """Create httpx.Client instance."""
        ...


class KVStoreFactoryProto(Protocol):
    """Protocol for redis_for_kv factory."""

    def __call__(self, url: str) -> RedisStrProto:
        """Create Redis client from URL."""
        ...


class RQConnectionFactoryProto(Protocol):
    """Protocol for redis_raw_for_rq factory."""

    def __call__(self, url: str) -> _RedisBytesClient:
        """Create Redis RQ client from URL."""
        ...


class RQQueueFactoryProto(Protocol):
    """Protocol for rq_queue factory."""

    def __call__(self, name: str, connection: _RedisBytesClient) -> RQClientQueue:
        """Create RQ queue from name and connection."""
        ...


class RQRetryFactoryProto(Protocol):
    """Protocol for rq_retry factory."""

    def __call__(self, *, max_retries: int, intervals: list[int]) -> RQRetryLike:
        """Create RQ retry from max_retries and intervals."""
        ...


class LoadSettingsProto(Protocol):
    """Protocol for load_settings factory."""

    def __call__(self) -> Settings:
        """Load settings."""
        ...


class ArtifactStoreProto(Protocol):
    """Protocol for ArtifactStore."""

    def upload_artifact(
        self,
        dir_path: Path,
        *,
        artifact_name: str,
        request_id: str,
    ) -> FileUploadResponse:
        """Upload a directory as a tarball artifact."""
        ...

    def download_artifact(
        self,
        file_id: str,
        *,
        dest_dir: Path,
        request_id: str,
        expected_root: str,
    ) -> Path:
        """Download and extract a tarball artifact."""
        ...


class ArtifactStoreFactoryProto(Protocol):
    """Protocol for ArtifactStore factory."""

    def __call__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 600.0,
    ) -> ArtifactStoreProto:
        """Create ArtifactStore instance."""
        ...


class ServiceContainerProto(Protocol):
    """Protocol for ServiceContainer."""

    @property
    def settings(self) -> Settings:
        """Get settings."""
        ...

    @property
    def redis(self) -> RedisStrProto:
        """Get Redis client."""
        ...

    @property
    def model_registry(self) -> ModelRegistry:
        """Get model registry."""
        ...


class ServiceContainerFactoryProto(Protocol):
    """Protocol for ServiceContainer.from_settings factory."""

    def __call__(self, settings: Settings) -> ServiceContainerProto:
        """Create ServiceContainer from settings."""
        ...


class RandomFactoryProto(Protocol):
    """Protocol for random.Random factory."""

    def __call__(self, seed: int) -> RandomLikeProto: ...


class RandomLikeProto(Protocol):
    """Protocol for random.Random-like objects."""

    def randint(self, a: int, b: int) -> int: ...


class ShutilRmtreeProto(Protocol):
    """Protocol for shutil.rmtree hook."""

    def __call__(self, path: Path | str) -> None: ...


class OsUtimeProto(Protocol):
    """Protocol for the os.utime hook.

    Used to mark a materialized run directory as recently USED, so the cache
    evicts by recency of use rather than by age of download.
    """

    def __call__(self, path: Path | str) -> None: ...


class OsScandirProto(Protocol):
    """Protocol for os.scandir hook."""

    def __call__(self, path: str) -> ScandirIterator: ...


class ScandirIterator(Protocol):
    """Protocol for os.scandir context manager."""

    def __enter__(self) -> ScandirIteratorContext: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None: ...


class ScandirIteratorContext(Protocol):
    """Protocol for iterating os.scandir entries."""

    def __iter__(self) -> ScandirIteratorContext: ...
    def __next__(self) -> DirEntryProto: ...


class DirEntryProto(Protocol):
    """Protocol for os.DirEntry-like objects."""

    @property
    def path(self) -> str: ...
    def is_file(self) -> bool: ...
    def stat(self) -> StatResultProto: ...


class StatResultProto(Protocol):
    """Protocol for stat result."""

    @property
    def st_size(self) -> int: ...
    @property
    def st_atime(self) -> float: ...
    @property
    def st_mtime(self) -> float: ...


class DiskUsageProto(Protocol):
    """Protocol for shutil.disk_usage result."""

    @property
    def total(self) -> int: ...
    @property
    def used(self) -> int: ...
    @property
    def free(self) -> int: ...


class ShutilDiskUsageProto(Protocol):
    """Protocol for shutil.disk_usage hook."""

    def __call__(self, path: str) -> DiskUsageProto: ...


class PathUnlinkProto(Protocol):
    """Protocol for Path.unlink hook."""

    def __call__(self, path: Path) -> None: ...


class TimeSleepProto(Protocol):
    """Protocol for time.sleep hook."""

    def __call__(self, seconds: float) -> None: ...


class PathIterdirProto(Protocol):
    """Protocol for Path.iterdir hook."""

    def __call__(self, path: Path) -> PathIterator: ...


class PathIterator(Protocol):
    """Protocol for Path iterator."""

    def __iter__(self) -> PathIterator: ...
    def __next__(self) -> Path: ...


class DumpJsonStrProto(Protocol):
    """Protocol for dump_json_str hook."""

    def __call__(self, value: JSONInputValue, *, compact: bool = True) -> str: ...


class PreparedLMModelProto(Protocol):
    """Protocol for PreparedLMModel-like objects returned by load hooks."""

    @property
    def model(self) -> LMModelProto: ...
    @property
    def tokenizer_id(self) -> str: ...
    @property
    def eos_id(self) -> int: ...
    @property
    def pad_id(self) -> int: ...
    @property
    def max_seq_len(self) -> int: ...


class TimeMonotonicProto(Protocol):
    """Protocol for time.monotonic hook."""

    def __call__(self) -> float:
        """Return monotonic time in seconds.

        Returns:
            Current monotonic clock value.
        """
        ...


class TimeWallClockProto(Protocol):
    """Protocol for the time.time hook.

    Distinct from :class:`TimeMonotonicProto` because heartbeats are stamped
    with wall-clock time and compared across processes: the worker writes the
    stamp and the API reads it, and two processes share no monotonic epoch.
    """

    def __call__(self) -> float:
        """Return seconds since the Unix epoch.

        Returns:
            Current wall-clock value.
        """
        ...


class DatetimeUtcnowIsoProto(Protocol):
    """Protocol for getting current UTC time as ISO 8601 string."""

    def __call__(self) -> str:
        """Return current UTC time as ISO 8601 string.

        Returns:
            ISO 8601 formatted timestamp (e.g., '2024-01-15T10:30:00').
        """
        ...


class GpuMaxMemoryAllocatedProto(Protocol):
    """Protocol for torch.cuda.max_memory_allocated hook."""

    def __call__(self) -> int:
        """Return peak GPU memory allocated in bytes.

        Returns:
            Peak memory in bytes, or 0 if CUDA not available.
        """
        ...


class GpuResetPeakMemoryStatsProto(Protocol):
    """Protocol for torch.cuda.reset_peak_memory_stats hook."""

    def __call__(self) -> None:
        """Reset peak memory tracking stats."""
        ...


class CountModelParametersProto(Protocol):
    """Protocol for counting model parameters."""

    def __call__(self, model: LMModelProto) -> int:
        """Count total trainable parameters in model.

        Args:
            model: The language model.

        Returns:
            Total number of trainable parameters.
        """
        ...


class GetDirectorySizeBytesProto(Protocol):
    """Protocol for calculating directory size on disk."""

    def __call__(self, path: Path) -> int:
        """Calculate total size of directory contents in bytes.

        Args:
            path: Directory path.

        Returns:
            Total size in bytes.
        """
        ...


class RandomGetstateProto(Protocol):
    """Protocol for the ``random.getstate`` hook."""

    def __call__(self) -> tuple[TorchStateValue, ...]:
        """Return the current python RNG state tuple."""
        ...


class RandomSetstateProto(Protocol):
    """Protocol for the ``random.setstate`` hook."""

    def __call__(self, state: tuple[TorchStateValue, ...]) -> None:
        """Restore a python RNG state tuple.

        Args:
            state: State tuple previously returned by ``random.getstate``.
        """
        ...


class TorchDeviceProto(Protocol):
    """Protocol for torch.device creation hook."""

    def __call__(self, device_str: str) -> torch.device:
        """Create a torch.device from string.

        Args:
            device_str: Device string ('cpu' or 'cuda').

        Returns:
            The torch device.
        """
        ...
