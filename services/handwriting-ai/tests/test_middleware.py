from __future__ import annotations

import asyncio

import pytest
from platform_core.errors import AppError
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from handwriting_ai.middleware import ExceptionNormalizeMiddleware


async def _raise_group(_: Request) -> Response:
    raise BaseExceptionGroup("group", [RuntimeError("boom")])


def test_exception_middleware_converts_exception_group() -> None:
    async def _noop_app(scope: Scope, receive: Receive, send: Send) -> None:
        return None

    headers: list[tuple[bytes, bytes]] = []
    scope: Scope = {"type": "http", "method": "GET", "path": "/", "headers": headers}

    middleware = ExceptionNormalizeMiddleware(_noop_app)
    request = Request(scope)

    async def _run_dispatch() -> None:
        await middleware.dispatch(request, _raise_group)

    with pytest.raises(AppError):
        asyncio.run(_run_dispatch())
