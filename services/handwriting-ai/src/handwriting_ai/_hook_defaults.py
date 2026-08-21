"""Default (production) implementations for the infrastructure hooks."""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from platform_core.config import HandwritingAiSettings
from platform_core.job_events import JobDomain
from platform_core.logging import (
    LogFormat,
    LogLevel,
)
from platform_workers.job_context import JobContext
from platform_workers.redis import (
    RedisStrProto,
)
from platform_workers.rq_harness import WorkerConfig

from handwriting_ai._hook_protocols import (
    ArtifactStoreProtocol,
    LoggerInstanceProtocol,
    LoggerProtocol,
    StatResultProtocol,
    WorkerRunnerProtocol,
)
from handwriting_ai._hook_protocols_training import (
    MultiprocessingChildProtocol,
)


def _default_guard_find_monorepo_root(start: Path) -> Path:
    """Production implementation - finds monorepo root by climbing directories."""
    current = start
    while True:
        if (current / "libs").is_dir():
            return current
        if current.parent == current:
            raise RuntimeError("monorepo root with 'libs' directory not found")
        current = current.parent


def _default_artifact_store_factory(api_url: str, api_key: str) -> ArtifactStoreProtocol:
    """Production implementation - creates real ArtifactStore."""
    from platform_core.data_bank_client import DataBankClient
    from platform_ml import ArtifactStore

    client = DataBankClient(api_url, api_key, timeout_seconds=600.0)
    return ArtifactStore(client)


def _default_run_worker(
    config: WorkerConfig,
    logger: LoggerProtocol,
    runner: WorkerRunnerProtocol,
) -> None:
    """Production implementation - runs actual worker."""
    runner(config)


def _default_is_cgroup_available() -> bool:
    """Production implementation - checks actual cgroup availability."""
    from .monitoring import is_cgroup_available as _ica

    return _ica()


def _default_get_logger(name: str) -> LoggerInstanceProtocol:
    """Production implementation - calls real get_logger."""
    from platform_core.logging import get_logger as _gl

    return _gl(name)


def _default_perf_counter() -> float:
    """Production implementation."""
    import time as _time

    return _time.perf_counter()


def _default_os_access(path: str, mode: int) -> bool:
    """Production implementation."""
    import os as _os

    return _os.access(path, mode)


def _default_randint(a: int, b: int) -> int:
    """Production implementation."""
    import random as _random

    return _random.randint(a, b)


def _default_uniform(a: float, b: float) -> float:
    """Production implementation."""
    import random as _random

    return _random.uniform(a, b)


def _default_mp_active_children() -> list[MultiprocessingChildProtocol]:
    """Production implementation."""
    import multiprocessing as _mp

    return list(_mp.active_children())


def _default_load_settings(*, create_dirs: bool = True) -> HandwritingAiSettings:
    """Production implementation - loads actual settings."""
    from handwriting_ai.config import load_settings as _ls

    return _ls(create_dirs=create_dirs)


def _default_make_job_context(
    *,
    redis: RedisStrProto,
    domain: JobDomain,
    events_channel: str,
    job_id: str,
    user_id: int,
    queue_name: str,
) -> JobContext:
    """Production implementation - creates real JobContext."""
    from platform_workers.job_context import make_job_context as _mjc

    return _mjc(
        redis=redis,
        domain=domain,
        events_channel=events_channel,
        job_id=job_id,
        user_id=user_id,
        queue_name=queue_name,
    )


def _default_runner_setup_logging(
    *,
    level: LogLevel,
    format_mode: LogFormat,
    service_name: str,
    instance_id: str | None,
    extra_fields: list[str] | None,
) -> None:
    """Production implementation."""
    from platform_core.logging import setup_logging as _sl

    _sl(
        level=level,
        format_mode=format_mode,
        service_name=service_name,
        instance_id=instance_id,
        extra_fields=extra_fields,
    )


def _default_file_open(
    file: str | Path,
    encoding: str = "utf-8",
) -> TextIO:
    """Production implementation - opens file in text mode using builtin open.

    Note: Caller is responsible for closing the returned file handle.
    """
    return Path(file).open(encoding=encoding)


def _default_now_ts() -> float:
    """Production implementation - returns current timestamp."""
    import time as _time

    return _time.time()


def _default_path_stat(path: Path, *, follow_symlinks: bool = True) -> StatResultProtocol:
    """Production implementation - calls real Path.stat."""
    return path.stat(follow_symlinks=follow_symlinks)


def _default_log_system_info() -> None:
    """Production implementation - logs system info."""
    from handwriting_ai.monitoring import log_system_info as _lsi

    _lsi()
