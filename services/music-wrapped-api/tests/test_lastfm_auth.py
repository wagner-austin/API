from __future__ import annotations

from types import ModuleType, TracebackType

import pytest
from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from pytest import MonkeyPatch

from music_wrapped_api.app import create_app


def test_auth_lastfm_start_builds_url(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LASTFM_API_KEY", "k")
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
    assert qs.get("api_key") == ["k"]
    assert qs.get("cb") == [cb]


def test_auth_lastfm_callback_success(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LASTFM_API_KEY", "k")
    monkeypatch.setenv("LASTFM_API_SECRET", "s")

    import music_wrapped_api.routes.wrapped as routes

    def _fake_get(api_key: str, api_secret: str, token: str) -> dict[str, JSONValue]:
        assert api_key == "k" and api_secret == "s" and token == "t1"
        return {"session": {"key": "sk", "name": "user"}}

    monkeypatch.setattr(routes, "_lfm_get_session_json", _fake_get)

    client = TestClient(create_app())
    r = client.get("/v1/wrapped/auth/lastfm/callback?token=t1")
    assert r.status_code == 200
    body = load_json_str(r.text)
    if not isinstance(body, dict):
        raise AssertionError("response must be an object")
    assert body.get("session_key") == "sk"
    assert body.get("username") == "user"


def test_auth_lastfm_callback_invalid(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LASTFM_API_KEY", "k")
    monkeypatch.setenv("LASTFM_API_SECRET", "s")

    import music_wrapped_api.routes.wrapped as routes

    def _fake_get(api_key: str, api_secret: str, token: str) -> dict[str, JSONValue]:
        return {}

    monkeypatch.setattr(routes, "_lfm_get_session_json", _fake_get)

    client = TestClient(create_app())
    r = client.get("/v1/wrapped/auth/lastfm/callback?token=t1")
    assert r.status_code == 502


def test_decode_lastfm_session_invalid_fields() -> None:
    from platform_core.errors import AppError

    import music_wrapped_api.routes.wrapped as routes

    with pytest.raises(AppError):
        routes._decode_lastfm_session({"session": {"key": 1, "name": 2}})


def test_lfm_get_session_json_invalid_json(monkeypatch: MonkeyPatch) -> None:
    import music_wrapped_api.routes.wrapped as routes

    class _FakeResp:
        def __init__(self, data: str) -> None:
            self._data = data.encode("utf-8")

        def read(self) -> bytes:
            return self._data

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

    # Provide a real module stub for urllib.request
    fake_mod = ModuleType("urllib.request")

    def urlopen(url: str, timeout: float) -> _FakeResp:
        return _FakeResp("[]")

    object.__setattr__(fake_mod, "urlopen", urlopen)

    import sys

    sys.modules["urllib.request"] = fake_mod
    from platform_core.errors import AppError

    with pytest.raises(AppError):
        routes._lfm_get_session_json("k", "s", "t")


def test_lfm_get_session_json_helper(monkeypatch: MonkeyPatch) -> None:
    import music_wrapped_api.routes.wrapped as routes

    class _FakeResp:
        def __init__(self, data: str) -> None:
            self._data = data.encode("utf-8")

        def read(self) -> bytes:
            return self._data

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

    fake_mod2 = ModuleType("urllib.request")

    def urlopen2(url: str, timeout: float) -> _FakeResp:
        assert url.startswith("https://ws.audioscrobbler.com/2.0/")
        body = '{"session": {"key": "sk", "name": "u"}}'
        return _FakeResp(body)

    object.__setattr__(fake_mod2, "urlopen", urlopen2)

    import sys

    sys.modules["urllib.request"] = fake_mod2
    out = routes._lfm_get_session_json("k", "s", "t")
    if not isinstance(out, dict):
        raise AssertionError("expected dict")
