"""Minimal Protocols for the ASGI / FastAPI boundary.

Split out of ``errors.py`` when that module crossed the 600-line ceiling, and
kept together because they share one role: they are the narrowest possible
description of the framework objects this library touches, so nothing here
depends on FastAPI or Starlette being importable.

They are private to ``platform_core``. ``fastapi.py`` previously carried its
own second copy of ``_URLProto``; there is now one definition, which is the
point of a boundary module rather than a per-file protocol.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class _RequestProto(Protocol):
    """Minimal protocol for FastAPI Request."""

    @property
    def url(self) -> _URLProto: ...

    @property
    def method(self) -> str: ...


@runtime_checkable
class _URLProto(Protocol):
    """Minimal protocol for Request.url."""

    @property
    def path(self) -> str: ...


@runtime_checkable
class _JSONResponseProto(Protocol):
    """Minimal protocol for FastAPI JSONResponse."""

    def __init__(self, content: dict[str, str], status_code: int) -> None: ...

    @property
    def body(self) -> bytes | memoryview[int]: ...

    @property
    def status_code(self) -> int: ...


@runtime_checkable
class _FastAPIAppProto(Protocol):
    """Minimal protocol for FastAPI application adapter.

    Services should create an adapter that wraps FastAPI and converts response types.
    See qr-api/src/qr_api/app.py for reference implementation.
    """

    def add_exception_handler(
        self,
        exc_class_or_status_code: int | type[Exception],
        handler: Callable[[_RequestProto, Exception], Awaitable[_JSONResponseProto]],
    ) -> None: ...


__all__ = [
    "_FastAPIAppProto",
    "_JSONResponseProto",
    "_RequestProto",
    "_URLProto",
]
