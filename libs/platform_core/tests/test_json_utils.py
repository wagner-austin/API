from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from platform_core.errors import (
    AppError,
    ErrorCode,
    _JSONResponseProto,
    _RequestProto,
    install_exception_handlers,
)
from platform_core.json_utils import (
    InvalidJsonError,
    JSONValue,
    dump_json_str,
    load_json_bytes,
    load_json_str,
    register_json_error_handler,
)


def _install_handlers_with_adapter(app: FastAPI) -> None:
    """Adapter to wrap FastAPI for install_exception_handlers."""

    class _Adapter:
        def __init__(self, inner: FastAPI) -> None:
            self._inner = inner

        def add_exception_handler(
            self,
            exc_class_or_status_code: int | type[Exception],
            handler: Callable[[_RequestProto, Exception], Awaitable[_JSONResponseProto]],
        ) -> None:
            async def _wrapped(request: Request, exc: Exception) -> Response:
                result = await handler(request, exc)
                return Response(
                    content=result.body if isinstance(result.body, bytes) else bytes(result.body),
                    status_code=result.status_code,
                    media_type="application/json",
                )

            self._inner.add_exception_handler(exc_class_or_status_code, _wrapped)

    install_exception_handlers(_Adapter(app))


def test_dump_json_str_valid() -> None:
    data: JSONValue = {"a": 1, "b": ["x", 2]}
    result = dump_json_str(data)
    assert type(result) is str
    parsed: JSONValue = load_json_str(result)
    assert parsed == data


def test_dump_json_str_non_compact() -> None:
    data: JSONValue = {"a": 1}
    result = dump_json_str(data, compact=False)
    assert type(result) is str
    parsed: JSONValue = load_json_str(result)
    assert parsed == data


def test_dump_json_str_with_indent() -> None:
    data: JSONValue = {"a": 1, "b": 2}
    result = dump_json_str(data, indent=2)
    assert type(result) is str
    assert "\n" in result  # Indented JSON has newlines
    assert "  " in result  # 2-space indentation
    parsed: JSONValue = load_json_str(result)
    assert parsed == data


def test_load_json_str_valid() -> None:
    val: JSONValue = load_json_str('{"a": 1, "b": ["x", 2]}')
    assert type(val) is dict
    assert val["a"] == 1


def test_load_json_str_invalid() -> None:
    with pytest.raises(InvalidJsonError):
        load_json_str("{")


def test_load_json_str_rejects_non_json_value(monkeypatch: pytest.MonkeyPatch) -> None:
    import json as json_module

    import platform_core.json_utils as ju

    def _fake_loads(_s: str) -> tuple[int, int]:
        return (1, 2)

    monkeypatch.setattr(json_module, "loads", _fake_loads, raising=True)
    with pytest.raises(ju.InvalidJsonError):
        ju.load_json_str("[]")


def test_load_json_bytes_valid() -> None:
    val: JSONValue = load_json_bytes(b'["x", 1]')
    assert type(val) is list
    assert val[0] == "x"


def test_register_json_error_handler_maps_invalid_json() -> None:
    app = FastAPI()
    _install_handlers_with_adapter(app)
    register_json_error_handler(app)

    async def _echo(request: Request) -> dict[str, str]:
        payload = load_json_bytes(await request.body())
        assert type(payload) is dict
        result: dict[str, str] = {k: str(v) for k, v in payload.items()}
        return result

    app.add_api_route("/echo", _echo, methods=["POST"])

    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/echo", content=b"{")
            assert resp.status_code == 400
            body: JSONValue = load_json_bytes(resp.content)
            assert type(body) is dict
            assert body["code"] == "INVALID_INPUT"
            assert body["message"] == "Invalid JSON body"

    asyncio.run(_run())


def test_register_json_error_handler_raises_http_exception() -> None:
    app = FastAPI()
    _install_handlers_with_adapter(app)
    register_json_error_handler(app)

    async def _boom(_: Request) -> dict[str, str]:
        raise InvalidJsonError("bad")

    app.add_api_route("/boom", _boom, methods=["GET"])

    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/boom")
            assert resp.status_code == 400
            body: JSONValue = load_json_bytes(resp.content)
            assert type(body) is dict
            assert body["code"] == "INVALID_INPUT"
            assert body["message"] == "Invalid JSON body"

    asyncio.run(_run())


def test_register_json_error_handler_reraises_unhandled() -> None:
    app = FastAPI()
    handler = register_json_error_handler(app)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.1"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "headers": [],
    }
    request = Request(scope)
    with pytest.raises(ValueError):
        handler(request, ValueError("boom"))

    with pytest.raises(AppError) as app_err:
        handler(request, InvalidJsonError("boom"))
    assert app_err.value.code is ErrorCode.INVALID_INPUT
    assert app_err.value.http_status == 400
