"""Default (production) implementations for the system-probe hooks."""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import psutil

from handwriting_ai._hook_protocols_system import (
    EventProtocol,
    PsutilProcessProtocol,
    ThreadProtocol,
    ThreadTargetProtocol,
    VirtualMemoryResultProtocol,
)


def _default_psutil_process(pid: int | None = None) -> PsutilProcessProtocol:
    """Production implementation - returns real psutil.Process."""
    return psutil.Process(pid)


def _default_psutil_virtual_memory() -> VirtualMemoryResultProtocol:
    """Production implementation - returns real psutil.virtual_memory()."""
    return psutil.virtual_memory()


def _default_psutil_cpu_count(*, logical: bool = True) -> int | None:
    """Production implementation - returns real psutil.cpu_count()."""
    import psutil as _psutil

    return _psutil.cpu_count(logical=logical)


def _default_os_getpid() -> int:
    """Production implementation - returns real os.getpid()."""
    import os as _os

    return _os.getpid()


def _default_read_text_file(path: Path) -> str:
    """Production implementation - reads text file."""
    return path.read_text(encoding="utf-8").strip()


def _default_os_cpu_count() -> int | None:
    """Production implementation - returns real os.cpu_count()."""
    import os as _os

    return _os.cpu_count()


def _default_torch_has_set_num_interop_threads() -> bool:
    """Production implementation - checks if torch has set_num_interop_threads."""
    import torch as _torch

    return hasattr(_torch, "set_num_interop_threads")


def _default_torch_has_get_num_interop_threads() -> bool:
    """Production implementation - checks if torch has get_num_interop_threads."""
    import torch as _torch

    return hasattr(_torch, "get_num_interop_threads")


def _default_torch_get_num_interop_threads() -> int:
    """Production implementation - gets real interop threads count."""
    import torch as _torch

    return _torch.get_num_interop_threads()


def _default_thread_factory(
    *,
    target: ThreadTargetProtocol,
    daemon: bool = True,
    name: str | None = None,
) -> ThreadProtocol:
    """Production implementation - creates real thread."""
    return threading.Thread(target=target, daemon=daemon, name=name)


def _default_event_factory() -> EventProtocol:
    """Production implementation - creates real event."""
    return threading.Event()


def _default_import_module(name: str, package: str | None = None) -> ModuleType:
    """Production implementation - imports actual module."""
    import importlib

    return importlib.import_module(name, package)


@contextmanager
def _default_limit_thread_pools(*, limits: int) -> Generator[None, None, None]:
    """Production implementation - limits thread pools."""
    from handwriting_ai.training.threadpool import limit_thread_pools as _ltp

    with _ltp(limits=limits):
        yield
