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
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_bytes,
    load_json_str,
    narrow_json_to_bool,
    narrow_json_to_dict,
    narrow_json_to_float,
    narrow_json_to_int,
    narrow_json_to_list,
    narrow_json_to_str,
    optional_float,
    optional_int,
    optional_str,
    register_json_error_handler,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
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


class TestNarrowJsonToDict:
    """Tests for narrow_json_to_dict."""

    def test_narrows_dict(self) -> None:
        value: JSONValue = {"a": 1, "b": "hello"}
        result = narrow_json_to_dict(value)
        assert result == {"a": 1, "b": "hello"}
        assert type(result) is dict

    def test_raises_for_list(self) -> None:
        value: JSONValue = [1, 2, 3]
        with pytest.raises(JSONTypeError, match="Expected JSON object, got list"):
            narrow_json_to_dict(value)

    def test_raises_for_str(self) -> None:
        value: JSONValue = "hello"
        with pytest.raises(JSONTypeError, match="Expected JSON object, got str"):
            narrow_json_to_dict(value)

    def test_raises_for_int(self) -> None:
        value: JSONValue = 42
        with pytest.raises(JSONTypeError, match="Expected JSON object, got int"):
            narrow_json_to_dict(value)

    def test_raises_for_none(self) -> None:
        value: JSONValue = None
        with pytest.raises(JSONTypeError, match="Expected JSON object, got NoneType"):
            narrow_json_to_dict(value)


class TestNarrowJsonToList:
    """Tests for narrow_json_to_list."""

    def test_narrows_list(self) -> None:
        value: JSONValue = [1, "two", 3.0]
        result = narrow_json_to_list(value)
        assert result == [1, "two", 3.0]
        assert type(result) is list

    def test_raises_for_dict(self) -> None:
        value: JSONValue = {"a": 1}
        with pytest.raises(JSONTypeError, match="Expected JSON array, got dict"):
            narrow_json_to_list(value)

    def test_raises_for_str(self) -> None:
        value: JSONValue = "hello"
        with pytest.raises(JSONTypeError, match="Expected JSON array, got str"):
            narrow_json_to_list(value)

    def test_raises_for_int(self) -> None:
        value: JSONValue = 42
        with pytest.raises(JSONTypeError, match="Expected JSON array, got int"):
            narrow_json_to_list(value)

    def test_raises_for_none(self) -> None:
        value: JSONValue = None
        with pytest.raises(JSONTypeError, match="Expected JSON array, got NoneType"):
            narrow_json_to_list(value)


class TestNarrowJsonToStr:
    """Tests for narrow_json_to_str."""

    def test_narrows_str(self) -> None:
        value: JSONValue = "hello world"
        result = narrow_json_to_str(value)
        assert result == "hello world"
        assert type(result) is str

    def test_raises_for_dict(self) -> None:
        value: JSONValue = {"a": 1}
        with pytest.raises(JSONTypeError, match="Expected JSON string, got dict"):
            narrow_json_to_str(value)

    def test_raises_for_list(self) -> None:
        value: JSONValue = [1, 2]
        with pytest.raises(JSONTypeError, match="Expected JSON string, got list"):
            narrow_json_to_str(value)

    def test_raises_for_int(self) -> None:
        value: JSONValue = 42
        with pytest.raises(JSONTypeError, match="Expected JSON string, got int"):
            narrow_json_to_str(value)

    def test_raises_for_none(self) -> None:
        value: JSONValue = None
        with pytest.raises(JSONTypeError, match="Expected JSON string, got NoneType"):
            narrow_json_to_str(value)


class TestNarrowJsonToInt:
    """Tests for narrow_json_to_int."""

    def test_narrows_int(self) -> None:
        value: JSONValue = 42
        result = narrow_json_to_int(value)
        assert result == 42
        assert type(result) is int

    def test_raises_for_bool(self) -> None:
        value: JSONValue = True
        with pytest.raises(JSONTypeError, match="Expected JSON integer, got bool"):
            narrow_json_to_int(value)

    def test_raises_for_float(self) -> None:
        value: JSONValue = 3.14
        with pytest.raises(JSONTypeError, match="Expected JSON integer, got float"):
            narrow_json_to_int(value)

    def test_raises_for_str(self) -> None:
        value: JSONValue = "42"
        with pytest.raises(JSONTypeError, match="Expected JSON integer, got str"):
            narrow_json_to_int(value)

    def test_raises_for_none(self) -> None:
        value: JSONValue = None
        with pytest.raises(JSONTypeError, match="Expected JSON integer, got NoneType"):
            narrow_json_to_int(value)


class TestNarrowJsonToFloat:
    """Tests for narrow_json_to_float."""

    def test_narrows_float(self) -> None:
        value: JSONValue = 3.14
        result = narrow_json_to_float(value)
        assert result == 3.14
        assert type(result) is float

    def test_converts_int_to_float(self) -> None:
        value: JSONValue = 42
        result = narrow_json_to_float(value)
        assert result == 42.0
        assert type(result) is float

    def test_raises_for_bool(self) -> None:
        value: JSONValue = True
        with pytest.raises(JSONTypeError, match="Expected JSON number, got bool"):
            narrow_json_to_float(value)

    def test_raises_for_str(self) -> None:
        value: JSONValue = "3.14"
        with pytest.raises(JSONTypeError, match="Expected JSON number, got str"):
            narrow_json_to_float(value)

    def test_raises_for_dict(self) -> None:
        value: JSONValue = {"a": 1}
        with pytest.raises(JSONTypeError, match="Expected JSON number, got dict"):
            narrow_json_to_float(value)

    def test_raises_for_none(self) -> None:
        value: JSONValue = None
        with pytest.raises(JSONTypeError, match="Expected JSON number, got NoneType"):
            narrow_json_to_float(value)


class TestNarrowJsonToBool:
    """Tests for narrow_json_to_bool."""

    def test_narrows_true(self) -> None:
        value: JSONValue = True
        result = narrow_json_to_bool(value)
        assert result is True
        assert type(result) is bool

    def test_narrows_false(self) -> None:
        value: JSONValue = False
        result = narrow_json_to_bool(value)
        assert result is False
        assert type(result) is bool

    def test_raises_for_int(self) -> None:
        value: JSONValue = 1
        with pytest.raises(JSONTypeError, match="Expected JSON boolean, got int"):
            narrow_json_to_bool(value)

    def test_raises_for_str(self) -> None:
        value: JSONValue = "true"
        with pytest.raises(JSONTypeError, match="Expected JSON boolean, got str"):
            narrow_json_to_bool(value)

    def test_raises_for_none(self) -> None:
        value: JSONValue = None
        with pytest.raises(JSONTypeError, match="Expected JSON boolean, got NoneType"):
            narrow_json_to_bool(value)


class TestRequireStr:
    """Tests for require_str."""

    def test_extracts_str(self) -> None:
        obj: JSONObject = {"name": "Alice"}
        result = require_str(obj, "name")
        assert result == "Alice"

    def test_raises_for_missing(self) -> None:
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field 'name'"):
            require_str(obj, "name")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"name": 123}
        with pytest.raises(JSONTypeError, match="Field 'name' must be a string, got int"):
            require_str(obj, "name")


class TestRequireInt:
    """Tests for require_int."""

    def test_extracts_int(self) -> None:
        obj: JSONObject = {"count": 42}
        result = require_int(obj, "count")
        assert result == 42

    def test_raises_for_missing(self) -> None:
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field 'count'"):
            require_int(obj, "count")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"count": "42"}
        with pytest.raises(JSONTypeError, match="Field 'count' must be an integer, got str"):
            require_int(obj, "count")

    def test_raises_for_bool(self) -> None:
        obj: JSONObject = {"count": True}
        with pytest.raises(JSONTypeError, match="Field 'count' must be an integer, got bool"):
            require_int(obj, "count")


class TestRequireFloat:
    """Tests for require_float."""

    def test_extracts_float(self) -> None:
        obj: JSONObject = {"rate": 3.14}
        result = require_float(obj, "rate")
        assert result == 3.14

    def test_converts_int_to_float(self) -> None:
        obj: JSONObject = {"rate": 42}
        result = require_float(obj, "rate")
        assert result == 42.0
        assert type(result) is float

    def test_raises_for_missing(self) -> None:
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field 'rate'"):
            require_float(obj, "rate")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"rate": "3.14"}
        with pytest.raises(JSONTypeError, match="Field 'rate' must be a number, got str"):
            require_float(obj, "rate")

    def test_raises_for_bool(self) -> None:
        obj: JSONObject = {"rate": True}
        with pytest.raises(JSONTypeError, match="Field 'rate' must be a number, got bool"):
            require_float(obj, "rate")


class TestRequireBool:
    """Tests for require_bool."""

    def test_extracts_true(self) -> None:
        obj: JSONObject = {"enabled": True}
        result = require_bool(obj, "enabled")
        assert result is True

    def test_extracts_false(self) -> None:
        obj: JSONObject = {"enabled": False}
        result = require_bool(obj, "enabled")
        assert result is False

    def test_raises_for_missing(self) -> None:
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field 'enabled'"):
            require_bool(obj, "enabled")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"enabled": 1}
        with pytest.raises(JSONTypeError, match="Field 'enabled' must be a boolean, got int"):
            require_bool(obj, "enabled")


class TestRequireList:
    """Tests for require_list."""

    def test_extracts_list(self) -> None:
        obj: JSONObject = {"items": [1, 2, 3]}
        result = require_list(obj, "items")
        assert result == [1, 2, 3]

    def test_raises_for_missing(self) -> None:
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field 'items'"):
            require_list(obj, "items")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"items": "not a list"}
        with pytest.raises(JSONTypeError, match="Field 'items' must be an array, got str"):
            require_list(obj, "items")


class TestRequireDict:
    """Tests for require_dict."""

    def test_extracts_dict(self) -> None:
        obj: JSONObject = {"config": {"key": "value"}}
        result = require_dict(obj, "config")
        assert result == {"key": "value"}

    def test_raises_for_missing(self) -> None:
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field 'config'"):
            require_dict(obj, "config")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"config": [1, 2, 3]}
        with pytest.raises(JSONTypeError, match="Field 'config' must be an object, got list"):
            require_dict(obj, "config")


class TestOptionalStr:
    """Tests for optional_str."""

    def test_returns_none_when_key_missing(self) -> None:
        obj: JSONObject = {}
        result = optional_str(obj, "name")
        assert result is None

    def test_returns_none_when_value_is_none(self) -> None:
        obj: JSONObject = {"name": None}
        result = optional_str(obj, "name")
        assert result is None

    def test_extracts_str(self) -> None:
        obj: JSONObject = {"name": "Alice"}
        result = optional_str(obj, "name")
        assert result == "Alice"
        assert type(result) is str

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"name": 123}
        with pytest.raises(JSONTypeError, match="Field 'name' must be a string, got int"):
            optional_str(obj, "name")


class TestOptionalInt:
    """Tests for optional_int."""

    def test_returns_none_when_key_missing(self) -> None:
        obj: JSONObject = {}
        result = optional_int(obj, "count")
        assert result is None

    def test_returns_none_when_value_is_none(self) -> None:
        obj: JSONObject = {"count": None}
        result = optional_int(obj, "count")
        assert result is None

    def test_extracts_int(self) -> None:
        obj: JSONObject = {"count": 42}
        result = optional_int(obj, "count")
        assert result == 42
        assert type(result) is int

    def test_raises_for_bool(self) -> None:
        obj: JSONObject = {"count": True}
        with pytest.raises(JSONTypeError, match="Field 'count' must be an integer, got bool"):
            optional_int(obj, "count")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"count": "42"}
        with pytest.raises(JSONTypeError, match="Field 'count' must be an integer, got str"):
            optional_int(obj, "count")


class TestOptionalFloat:
    """Tests for optional_float."""

    def test_returns_none_when_key_missing(self) -> None:
        obj: JSONObject = {}
        result = optional_float(obj, "rate")
        assert result is None

    def test_returns_none_when_value_is_none(self) -> None:
        obj: JSONObject = {"rate": None}
        result = optional_float(obj, "rate")
        assert result is None

    def test_extracts_float(self) -> None:
        obj: JSONObject = {"rate": 3.14}
        result = optional_float(obj, "rate")
        assert result == 3.14
        assert type(result) is float

    def test_converts_int_to_float(self) -> None:
        obj: JSONObject = {"rate": 42}
        result = optional_float(obj, "rate")
        assert result == 42.0
        assert type(result) is float

    def test_raises_for_bool(self) -> None:
        obj: JSONObject = {"rate": True}
        with pytest.raises(JSONTypeError, match="Field 'rate' must be a number, got bool"):
            optional_float(obj, "rate")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"rate": "3.14"}
        with pytest.raises(JSONTypeError, match="Field 'rate' must be a number, got str"):
            optional_float(obj, "rate")
