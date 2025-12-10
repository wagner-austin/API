from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import InvalidJsonError, load_json_str, register_json_error_handler
from starlette.requests import Request

from qr_api.app import create_app
from qr_api.settings import load_default_options_from_env
from qr_api.validators import _decode_qr_payload

_DEFAULTS = load_default_options_from_env()


def test_qr_decode_generates_valid_options() -> None:
    opts = _decode_qr_payload({"url": "example.com"}, _DEFAULTS)
    assert opts["url"] == "https://example.com"
    assert opts["ecc"] == "M"
    assert opts["box_size"] == 10


def test_qr_decode_validates_fields() -> None:
    with pytest.raises(AppError):
        _decode_qr_payload({}, _DEFAULTS)
    with pytest.raises(AppError):
        _decode_qr_payload({"url": "example.com", "ecc": "Z"}, _DEFAULTS)
    with pytest.raises(AppError):
        _decode_qr_payload({"url": "example.com", "box_size": 1}, _DEFAULTS)
    with pytest.raises(AppError):
        _decode_qr_payload({"url": "example.com", "border": 0}, _DEFAULTS)
    with pytest.raises(AppError):
        _decode_qr_payload({"url": "example.com", "fill_color": "red"}, _DEFAULTS)


def test_create_app_registers_qr_route() -> None:
    from fastapi.testclient import TestClient

    app = create_app(_DEFAULTS)
    # Verify the route exists by making a request
    with TestClient(app) as client:
        # OPTIONS request to check route exists without triggering validation
        response = client.options("/v1/qr")
        # 405 Method Not Allowed means route exists but OPTIONS not supported
        # 200 means route exists and OPTIONS is supported
        assert response.status_code in (200, 405)


def test_decode_qr_request_returns_valid_png() -> None:
    """Test the full FastAPI route integration for a valid PNG response."""
    from fastapi.testclient import TestClient

    app = create_app(_DEFAULTS)

    with TestClient(app) as client:
        json_payload: dict[str, str] = {"url": "https://example.com"}
        response = client.post("/v1/qr", json=json_payload)

        assert response.status_code == 200
        content_type_value: str | None = response.headers.get("content-type")
        content_type: str = str(content_type_value) if content_type_value is not None else ""
        assert "image/png" in content_type
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_qr_route_integration() -> None:
    """Test the full FastAPI route integration."""
    from fastapi.testclient import TestClient

    app = create_app(_DEFAULTS)

    with TestClient(app) as client:
        json_data: dict[str, str] = {"url": "https://example.com"}
        response = client.post("/v1/qr", json=json_data)

        assert response.status_code == 200
        # Access headers via dict interface to avoid Any from .get()
        content_type_value = response.headers["content-type"]
        content_type: str = str(content_type_value) if content_type_value is not None else ""
        assert "image/png" in content_type
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_qr_route_rejects_invalid_json() -> None:
    from fastapi.testclient import TestClient

    app = create_app(_DEFAULTS)

    with TestClient(app) as client:
        response = client.post(
            "/v1/qr",
            content=b"{",
            headers={"content-type": "application/json"},
        )  # malformed JSON

        assert response.status_code == 400
        content_type_value = response.headers["content-type"]
        content_type: str = str(content_type_value) if content_type_value is not None else ""
        assert "application/json" in content_type
        parsed_body_raw = load_json_str(response.text)
        if type(parsed_body_raw) is not dict:
            pytest.fail("expected dict response body")
        parsed_body = parsed_body_raw
        assert parsed_body["code"] == "INVALID_INPUT"
        assert parsed_body["message"] == "Invalid JSON body"
        assert parsed_body["request_id"] == ""  # empty in test context without middleware


def test_json_error_handler_reraises_non_jsondecodeerror() -> None:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "query_string": b"",
        "scheme": "http",
        "server": ("localhost", 80),
        "client": ("0.0.0.0", 0),
        "http_version": "1.1",
    }
    request = Request(scope)

    def _handler(_: Request, exc: Exception) -> None:
        handler = register_json_error_handler(create_app())
        handler(_, exc)

    with pytest.raises(ValueError):
        _handler(request, ValueError("boom"))

    handler = register_json_error_handler(create_app())
    with pytest.raises(AppError) as ex:
        handler(request, InvalidJsonError("boom"))
    assert ex.value.code is ErrorCode.INVALID_INPUT
    assert ex.value.http_status == 400
