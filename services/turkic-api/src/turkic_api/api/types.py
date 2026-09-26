from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Protocol

from platform_core.json_utils import JSONValue
from platform_workers.rq_harness import RQJobLike, RQRetryLike

__all__ = ["JSONValue", "LoggerProtocol", "RQJobLike", "RQRetryLike"]


class LoggerProtocol(Protocol):
    """Protocol for a minimal structured logger interface."""

    def debug(
        self,
        msg: str,
        *args: JSONValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, JSONValue] | None = None,
    ) -> None: ...

    def info(
        self,
        msg: str,
        *args: JSONValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, JSONValue] | None = None,
    ) -> None: ...

    def warning(
        self,
        msg: str,
        *args: JSONValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, JSONValue] | None = None,
    ) -> None: ...

    def error(
        self,
        msg: str,
        *args: JSONValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, JSONValue] | None = None,
    ) -> None: ...
