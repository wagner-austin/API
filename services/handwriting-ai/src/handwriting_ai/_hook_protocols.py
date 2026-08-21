"""Service-infrastructure hook protocols (workers, queues, guards, stores).

System probes: ``_hook_protocols_system``; ML: ``_hook_protocols_ml``;
training pipeline: ``_hook_protocols_training``. Bindings stay in
:mod:`handwriting_ai._test_hooks` (tests rebind its attributes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TextIO

import torch
from platform_core.config import HandwritingAiSettings
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.job_events import JobDomain
from platform_core.json_utils import JSONValue
from platform_core.logging import (
    LogFormat,
    LogLevel,
)
from platform_workers.job_context import JobContext
from platform_workers.redis import (
    RedisStrProto,
    _RedisBytesClient,
)
from platform_workers.rq_harness import RQClientQueue, WorkerConfig

from handwriting_ai.training.calibration._types import (
    CandidateOutcomeDict,
)


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


class GuardRunForProjectProtocol(Protocol):
    """Protocol for run_for_project function from monorepo_guards."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guards for a project."""
        ...


class GuardFindMonorepoRootProtocol(Protocol):
    """Protocol for _find_monorepo_root function."""

    def __call__(self, start: Path) -> Path:
        """Find the monorepo root from a starting path."""
        ...


class KVStoreFactoryProtocol(Protocol):
    """Protocol for key-value store factory (returns RedisStrProto)."""

    def __call__(self, url: str) -> RedisStrProto:
        """Create a string-based KV store client from URL."""
        ...


class QueueConnFactoryProtocol(Protocol):
    """Protocol for queue connection factory (returns _RedisBytesClient)."""

    def __call__(self, url: str) -> _RedisBytesClient:
        """Create a bytes-based connection for queue operations from URL."""
        ...


class QueueFactoryProtocol(Protocol):
    """Protocol for job queue factory (returns RQClientQueue)."""

    def __call__(self, name: str, connection: _RedisBytesClient) -> RQClientQueue:
        """Create a job queue from name and connection."""
        ...


class ArtifactStoreProtocol(Protocol):
    """Protocol for ArtifactStore - allows injecting fakes for testing."""

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


class ArtifactStoreFactoryProtocol(Protocol):
    """Protocol for ArtifactStore factory."""

    def __call__(self, api_url: str, api_key: str) -> ArtifactStoreProtocol:
        """Create an ArtifactStore instance."""
        ...


class LoggerProtocol(Protocol):
    """Minimal protocol for logging.Logger."""

    def info(self, msg: str, *args: str) -> None: ...

    def error(self, msg: str, *args: str) -> None: ...


class RunWorkerProtocol(Protocol):
    """Protocol for run_worker function."""

    def __call__(
        self,
        config: WorkerConfig,
        logger: LoggerProtocol,
        runner: WorkerRunnerProtocol,
    ) -> None: ...


class IsCgroupAvailableProtocol(Protocol):
    """Protocol for is_cgroup_available function."""

    def __call__(self) -> bool: ...


class LoggerInstanceProtocol(Protocol):
    """Protocol for logger instance returned by get_logger.

    Note: setLevel is intentionally omitted as it's not used through this Protocol.
    Code that needs setLevel uses the actual logger directly.
    """

    def info(
        self,
        msg: str,
        *args: float | int | str | Path | BaseException,
        extra: dict[str, str | int | float | bool | None] | None = None,
    ) -> None: ...

    def warning(
        self,
        msg: str,
        *args: float | int | str | Path | BaseException,
        extra: dict[str, str | int | float | bool | None] | None = None,
    ) -> None: ...

    def error(
        self,
        msg: str,
        *args: float | int | str | Path | BaseException,
        extra: dict[str, str | int | float | bool | None] | None = None,
    ) -> None: ...

    def debug(
        self,
        msg: str,
        *args: float | int | str | Path | BaseException,
        extra: dict[str, str | int | float | bool | None] | None = None,
    ) -> None: ...


class GetLoggerProtocol(Protocol):
    """Protocol for get_logger function."""

    def __call__(self, name: str) -> LoggerInstanceProtocol: ...


class PerfCounterProtocol(Protocol):
    """Protocol for time.perf_counter."""

    def __call__(self) -> float: ...


class OsAccessProtocol(Protocol):
    """Protocol for os.access."""

    def __call__(self, path: str, mode: int) -> bool: ...


class LoadSettingsProtocol(Protocol):
    """Protocol for _load_settings function."""

    def __call__(self, *, create_dirs: bool = True) -> HandwritingAiSettings: ...


class JobContextProtocol(Protocol):
    """Protocol for JobContext returned by make_job_context."""

    def publish_started(self) -> None: ...

    def publish_progress(
        self, progress: int, message: str | None = None, *, payload: JSONValue | None = None
    ) -> None: ...

    def publish_completed(self, result_id: str, result_bytes: int) -> None: ...

    def publish_failed(self, error_kind: str, message: str) -> None: ...


class MakeJobContextProtocol(Protocol):
    """Protocol for make_job_context function.

    Returns JobContext or None. None is used in tests to verify
    job completion without a publisher.
    """

    def __call__(
        self,
        *,
        redis: RedisStrProto,
        domain: JobDomain,
        events_channel: str,
        job_id: str,
        user_id: int,
        queue_name: str,
    ) -> JobContext | None: ...


class SetupLoggingProtocol(Protocol):
    """Protocol for setup_logging."""

    def __call__(
        self,
        *,
        level: LogLevel,
        format_mode: LogFormat,
        service_name: str,
        instance_id: str | None,
        extra_fields: list[str] | None,
    ) -> None: ...


class FileOpenProtocol(Protocol):
    """Protocol for file open function for text mode."""

    def __call__(
        self,
        file: str | Path,
        encoding: str = "utf-8",
    ) -> TextIO: ...


class TryReadResultProtocol(Protocol):
    """Protocol for _try_read_result static method."""

    def __call__(
        self, out_path: str, *, exited: bool, exit_code: int | None
    ) -> CandidateOutcomeDict | None: ...


class NowTsProtocol(Protocol):
    """Protocol for _now_ts function."""

    def __call__(self) -> float: ...


class StatResultProtocol(Protocol):
    """Protocol for os.stat_result - minimal interface for file stats."""

    @property
    def st_mtime(self) -> float: ...

    @property
    def st_size(self) -> int: ...


class PathStatProtocol(Protocol):
    """Protocol for Path.stat function."""

    def __call__(self, path: Path, *, follow_symlinks: bool = True) -> StatResultProtocol: ...


class LogSystemInfoProtocol(Protocol):
    """Protocol for log_system_info function."""

    def __call__(self) -> None: ...


class InjectBadStateDictListProtocol(Protocol):
    """Protocol for function that injects bad state dict (list instead of dict)."""

    def __call__(self) -> dict[str, torch.Tensor]: ...


class InjectBadStateDictValuesProtocol(Protocol):
    """Protocol for function that injects bad state dict (int values instead of Tensor)."""

    def __call__(self) -> dict[str, torch.Tensor]: ...


class InjectBadStateDictNonStringKeyProtocol(Protocol):
    """Protocol for function that injects state dict with non-string key."""

    def __call__(self) -> dict[str, torch.Tensor]: ...
