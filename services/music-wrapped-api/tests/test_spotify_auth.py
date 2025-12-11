from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis

from music_wrapped_api import _test_hooks
from music_wrapped_api.api.main import create_app


def test_auth_spotify_start() -> None:
    fr = FakeRedis()
    _test_hooks.redis_factory = lambda url: fr

    client = TestClient(create_app())
    cb = "http://localhost/cb"
    r = client.get(f"/v1/wrapped/auth/spotify/start?callback={cb}")
    assert r.status_code == 200
    body = load_json_str(r.text)
    if not isinstance(body, dict):
        raise AssertionError("response must be an object")
    assert "auth_url" in body and "state" in body
    fr.assert_only_called({"sadd", "hset", "expire"})


def test_auth_spotify_callback_success() -> None:
    fr = FakeRedis()
    st = "test-state-12345"
    fr.hset(f"spotify:state:{st}", {"ok": "1"})
    _test_hooks.redis_factory = lambda url: fr

    def _fake_exchange(
        code: str, redirect_uri: str, client_id: str, client_secret: str
    ) -> dict[str, JSONValue]:
        return {
            "access_token": "at",
            "refresh_token": "rt",
            "expires_in": 3600,
        }

    _test_hooks.spotify_exchange_code = _fake_exchange

    client = TestClient(create_app())
    r = client.get(
        f"/v1/wrapped/auth/spotify/callback?code=abc&state={st}&callback=http://localhost/cb"
    )
    assert r.status_code == 200
    body = load_json_str(r.text)
    if not isinstance(body, dict):
        raise AssertionError("response must be an object")
    tid = body.get("token_id")
    assert isinstance(tid, str) and len(tid) == 32
    stored = fr.hgetall(f"spotify:session:{tid}")
    assert stored.get("access_token") == "at"
    fr.assert_only_called({"sadd", "hset", "expire", "hgetall"})


def test_auth_spotify_callback_invalid_state() -> None:
    fr = FakeRedis()
    _test_hooks.redis_factory = lambda url: fr

    client = TestClient(create_app())
    r = client.get(
        "/v1/wrapped/auth/spotify/callback?code=abc&state=bad&callback=http://localhost/cb"
    )
    assert r.status_code == 400
    fr.assert_only_called({"sadd", "hgetall"})


def test_auth_spotify_callback_invalid_json() -> None:
    fr = FakeRedis()
    st = "test-state-12345"
    fr.hset(f"spotify:state:{st}", {"ok": "1"})
    _test_hooks.redis_factory = lambda url: fr

    def _fake_exchange_bad(
        code: str, redirect_uri: str, client_id: str, client_secret: str
    ) -> dict[str, JSONValue]:
        # Return non-dict from the hook implementation (simulating array response)
        from platform_core.errors import AppError, ErrorCode

        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message="invalid spotify json",
            http_status=502,
        )

    _test_hooks.spotify_exchange_code = _fake_exchange_bad

    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get(
        f"/v1/wrapped/auth/spotify/callback?code=ax&state={st}&callback=http://localhost/cb"
    )
    assert r.status_code == 502
    fr.assert_only_called({"sadd", "hset", "expire", "hgetall"})


def test_auth_spotify_callback_invalid_fields() -> None:
    fr = FakeRedis()
    st = "test-state-12345"
    fr.hset(f"spotify:state:{st}", {"ok": "1"})
    _test_hooks.redis_factory = lambda url: fr

    def _fake_exchange_bad_fields(
        code: str, redirect_uri: str, client_id: str, client_secret: str
    ) -> dict[str, JSONValue]:
        # Return dict with wrong types
        return {
            "access_token": "at",
            "refresh_token": 1,  # Should be string
            "expires_in": "x",  # Should be int
        }

    _test_hooks.spotify_exchange_code = _fake_exchange_bad_fields

    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get(
        f"/v1/wrapped/auth/spotify/callback?code=ay&state={st}&callback=http://localhost/cb"
    )
    assert r.status_code == 502
    fr.assert_only_called({"sadd", "hset", "expire", "hgetall"})
