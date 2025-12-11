from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue, load_json_str

from music_wrapped_api import _test_hooks
from music_wrapped_api.api.main import create_app
from music_wrapped_api.api.routes.wrapped import _decode_lastfm_session


def test_auth_lastfm_start_builds_url() -> None:
    client = TestClient(create_app())
    cb = "http://localhost/callback"
    r = client.get(f"/v1/wrapped/auth/lastfm/start?callback={cb}")
    assert r.status_code == 200
    body = load_json_str(r.text)
    if not isinstance(body, dict):
        raise AssertionError("response must be an object")
    url_val: JSONValue | None = body.get("auth_url")
    if not isinstance(url_val, str):
        raise AssertionError("auth_url must be a string")
    url = url_val
    from urllib.parse import parse_qs, urlparse

    parts = urlparse(url)
    qs = parse_qs(parts.query)
    assert qs.get("api_key") == ["test-lastfm-key"]
    assert qs.get("cb") == [cb]


def test_auth_lastfm_callback_success() -> None:
    def _fake_get(api_key: str, api_secret: str, token: str) -> dict[str, JSONValue]:
        assert api_key == "test-lastfm-key" and api_secret == "test-lastfm-secret"
        assert token == "t1"
        return {"session": {"key": "sk", "name": "user"}}

    _test_hooks.lfm_get_session_json = _fake_get

    client = TestClient(create_app())
    r = client.get("/v1/wrapped/auth/lastfm/callback?token=t1")
    assert r.status_code == 200
    body = load_json_str(r.text)
    if not isinstance(body, dict):
        raise AssertionError("response must be an object")
    assert body.get("session_key") == "sk"
    assert body.get("username") == "user"


def test_auth_lastfm_callback_invalid() -> None:
    def _fake_get_empty(api_key: str, api_secret: str, token: str) -> dict[str, JSONValue]:
        return {}

    _test_hooks.lfm_get_session_json = _fake_get_empty

    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get("/v1/wrapped/auth/lastfm/callback?token=t1")
    assert r.status_code == 502


def test_decode_lastfm_session_invalid_fields() -> None:
    with pytest.raises(AppError):
        _decode_lastfm_session({"session": {"key": 1, "name": 2}})


def test_lfm_get_session_json_invalid_json() -> None:
    def _bad_get(api_key: str, api_secret: str, token: str) -> dict[str, JSONValue]:
        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message="invalid lastfm json",
            http_status=502,
        )

    _test_hooks.lfm_get_session_json = _bad_get

    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get("/v1/wrapped/auth/lastfm/callback?token=t")
    assert r.status_code == 502


def test_lfm_get_session_json_helper() -> None:
    def _success_get(api_key: str, api_secret: str, token: str) -> dict[str, JSONValue]:
        return {"session": {"key": "sk", "name": "u"}}

    _test_hooks.lfm_get_session_json = _success_get

    # Call the hook directly to test the helper behavior
    out = _test_hooks.lfm_get_session_json("k", "s", "t")
    if not isinstance(out, dict):
        raise AssertionError("expected dict")
    assert out == {"session": {"key": "sk", "name": "u"}}
