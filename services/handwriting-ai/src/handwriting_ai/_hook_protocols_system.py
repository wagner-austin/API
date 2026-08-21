"""System-probe hook protocols (psutil, cgroup files, threads, imports)."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType
from typing import Protocol


class VirtualMemoryResultProtocol(Protocol):
    """Protocol for psutil.virtual_memory() result (a NamedTuple with total and used)."""

    @property
    def total(self) -> int: ...

    @property
    def used(self) -> int: ...


class MemoryInfoProtocol(Protocol):
    """Protocol for psutil.Process.memory_info() result.

    Note: rss is int | str to allow tests to inject bad data for
    testing runtime isinstance defensive checks.
    """

    @property
    def rss(self) -> int | str: ...


class PsutilProcessProtocol(Protocol):
    """Protocol for psutil.Process - what monitoring code needs."""

    @property
    def pid(self) -> int: ...

    def memory_info(self) -> MemoryInfoProtocol: ...

    def children(self, recursive: bool = False) -> Sequence[PsutilProcessProtocol]: ...


class ProcessFactoryProtocol(Protocol):
    """Protocol for psutil.Process factory."""

    def __call__(self, pid: int | None = None) -> PsutilProcessProtocol: ...


class VirtualMemoryFactoryProtocol(Protocol):
    """Protocol for psutil.virtual_memory factory."""

    def __call__(self) -> VirtualMemoryResultProtocol: ...


class CpuCountFactoryProtocol(Protocol):
    """Protocol for psutil.cpu_count factory."""

    def __call__(self, *, logical: bool = True) -> int | None: ...


class GetPidProtocol(Protocol):
    """Protocol for os.getpid."""

    def __call__(self) -> int: ...


class ReadTextFileProtocol(Protocol):
    """Protocol for reading text files."""

    def __call__(self, path: Path) -> str:
        """Read text content from a file path."""
        ...


class OsCpuCountProtocol(Protocol):
    """Protocol for os.cpu_count."""

    def __call__(self) -> int | None:
        """Return number of CPUs."""
        ...


class TorchSetInteropThreadsProtocol(Protocol):
    """Protocol for torch.set_num_interop_threads."""

    def __call__(self, nthreads: int) -> None:
        """Set number of interop threads."""
        ...


class TorchHasSetNumInteropThreadsProtocol(Protocol):
    """Protocol for hasattr(torch, 'set_num_interop_threads')."""

    def __call__(self) -> bool: ...


class TorchHasGetNumInteropThreadsProtocol(Protocol):
    """Protocol for hasattr(torch, 'get_num_interop_threads')."""

    def __call__(self) -> bool: ...


class TorchGetNumInteropThreadsProtocol(Protocol):
    """Protocol for torch.get_num_interop_threads."""

    def __call__(self) -> int: ...


class ThreadProtocol(Protocol):
    """Protocol for threading.Thread."""

    def start(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...


class EventProtocol(Protocol):
    """Protocol for threading.Event."""

    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...

    def is_set(self) -> bool: ...


class ThreadTargetProtocol(Protocol):
    """Protocol for thread target callable - callable with no args."""

    def __call__(self) -> None: ...


class ThreadFactoryProtocol(Protocol):
    """Protocol for threading.Thread constructor."""

    def __call__(
        self,
        *,
        target: ThreadTargetProtocol,
        daemon: bool = True,
        name: str | None = None,
    ) -> ThreadProtocol: ...


class EventFactoryProtocol(Protocol):
    """Protocol for threading.Event constructor."""

    def __call__(self) -> EventProtocol: ...


class ImportModuleProtocol(Protocol):
    """Protocol for importlib.import_module."""

    def __call__(self, name: str, package: str | None = None) -> ModuleType: ...


class LimitThreadPoolsProtocol(Protocol):
    """Protocol for limit_thread_pools function."""

    def __call__(self, *, limits: int) -> AbstractContextManager[None]: ...
