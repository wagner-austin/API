from __future__ import annotations

import types
from collections.abc import Generator
from contextlib import contextmanager
from typing import Protocol


class _ThreadpoolLimitsCallable(Protocol):
    """Protocol for threadpoolctl.threadpool_limits callable."""

    def __call__(self, *, limits: int) -> _ThreadpoolContextManager: ...


class _ThreadpoolContextManager(Protocol):
    """Protocol for the context manager returned by threadpool_limits."""

    def __enter__(self) -> None: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...


def _load_threadpool_limits() -> _ThreadpoolLimitsCallable:
    """Load threadpool_limits from threadpoolctl with strict typing."""
    mod = __import__("threadpoolctl")
    fn: _ThreadpoolLimitsCallable = mod.threadpool_limits
    return fn


@contextmanager
def limit_thread_pools(*, limits: int) -> Generator[None, None, None]:
    """Type-safe wrapper for threadpoolctl.threadpool_limits.

    Limits OpenBLAS, MKL, and other numerical library thread pools.

    Args:
        limits: Maximum number of threads per pool

    Yields:
        None - context manager limits thread pools for its duration
    """
    fn = _load_threadpool_limits()
    with fn(limits=limits):
        yield


__all__ = ["limit_thread_pools"]
