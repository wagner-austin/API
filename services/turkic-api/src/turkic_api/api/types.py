from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Protocol

from platform_workers.rq_harness import RQJobLike, RQRetryLike

from turkic_api.core.models import UnknownJson

__all__ = ["JsonDict", "LoggerProtocol", "QueueProtocol", "RQJobLike", "RQRetryLike", "UnknownJson"]

# Public JSON type for API boundaries - non-recursive, one-level deep
JsonDict = dict[str, str | int | float | bool | None | list[str | int | float | bool | None]]


class _EnqCallable(Protocol):
    def __call__(
        self,
        *args: UnknownJson,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike: ...


class LoggerProtocol(Protocol):
    """Protocol for a minimal structured logger interface."""

    def debug(
        self,
        msg: str,
        *args: UnknownJson,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, UnknownJson] | None = None,
    ) -> None: ...

    def info(
        self,
        msg: str,
        *args: UnknownJson,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, UnknownJson] | None = None,
    ) -> None: ...

    def warning(
        self,
        msg: str,
        *args: UnknownJson,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, UnknownJson] | None = None,
    ) -> None: ...

    def error(
        self,
        msg: str,
        *args: UnknownJson,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, UnknownJson] | None = None,
    ) -> None: ...


class QueueProtocol(Protocol):
    """Minimal interface for a background job queue."""

    def enqueue(
        self,
        func: str | _EnqCallable,
        *args: UnknownJson,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike: ...
