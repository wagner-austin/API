from __future__ import annotations

from collections.abc import Callable
from types import ModuleType, TracebackType
from typing import Protocol

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str
from platform_workers.testing import FakeRedis
from pytest import MonkeyPatch

from music_wrapped_api.app import create_app


class _Resp:
    def __init__(self, data: str) -> None:
        self._d = data.encode("utf-8")

    def read(self) -> bytes:
        return self._d

    def __enter__(self) -> _Resp:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class _ReqProto(Protocol):
    def add_header(self, name: str, value: str) -> None: ...


def _fake_urllib_with_body(body: str) -> ModuleType:
    mod = ModuleType("urllib.request")

    def urlopen(req: _ReqProto, timeout: float) -> _Resp:
        return _Resp(body)

    object.__setattr__(mod, "urlopen", urlopen)
    return mod


def _import_override_with(mod: ModuleType) -> Callable[[str, list[str]], ModuleType]:
    def _fake_import(name: str, fromlist: list[str]) -> ModuleType:
        if name == "urllib.request":
            return mod
        return __import__(name, fromlist=fromlist)

    return _fake_import


def test_auth_spotify_start(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")

    import music_wrapped_api.routes.wrapped as routes

    fr = FakeRedis()

    def _rf(url: str) -> FakeRedis:
        return fr

    monkeypatch.setattr(routes, "redis_for_kv", _rf)
    client = TestClient(create_app())
    cb = "http://localhost/cb"
    r = client.get(f"/v1/wrapped/auth/spotify/start?callback={cb}")
    assert r.status_code == 200
    body = load_json_str(r.text)
    if not isinstance(body, dict):
        raise AssertionError("response must be an object")
    assert "auth_url" in body and "state" in body


def test_auth_spotify_callback_success(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "csec")

    import music_wrapped_api.routes.wrapped as routes

    fr = FakeRedis()
    st = "state123"
    fr.hset(f"spotify:state:{st}", {"ok": "1"})

    def _rf(url: str) -> FakeRedis:
        return fr

    fake_mod = _fake_urllib_with_body(
        '{"access_token":"at","refresh_token":"rt","expires_in":3600}'
    )
    _fake_import = _import_override_with(fake_mod)

    monkeypatch.setattr(routes, "redis_for_kv", _rf)
    monkeypatch.setattr(routes, "__import__", _fake_import)
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


def test_auth_spotify_callback_invalid_state(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "csec")
    import music_wrapped_api.routes.wrapped as routes

    def _rf(url: str) -> FakeRedis:
        return FakeRedis()

    monkeypatch.setattr(routes, "redis_for_kv", _rf)
    client = TestClient(create_app())
    r = client.get(
        "/v1/wrapped/auth/spotify/callback?code=abc&state=bad&callback=http://localhost/cb"
    )
    assert r.status_code == 400


def test_auth_spotify_callback_invalid_json(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "csec")

    import music_wrapped_api.routes.wrapped as routes

    fr = FakeRedis()
    st = "statex"
    fr.hset(f"spotify:state:{st}", {"ok": "1"})

    def _rf(url: str) -> FakeRedis:
        return fr

    fake_mod2 = _fake_urllib_with_body("[]")
    _fake_import = _import_override_with(fake_mod2)

    monkeypatch.setattr(routes, "redis_for_kv", _rf)
    monkeypatch.setattr(routes, "__import__", _fake_import)
    client = TestClient(create_app())
    r = client.get(
        f"/v1/wrapped/auth/spotify/callback?code=ax&state={st}&callback=http://localhost/cb"
    )
    assert r.status_code == 502


def test_auth_spotify_callback_invalid_fields(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "csec")

    import music_wrapped_api.routes.wrapped as routes

    fr = FakeRedis()
    st = "statey"
    fr.hset(f"spotify:state:{st}", {"ok": "1"})

    def _rf(url: str) -> FakeRedis:
        return fr

    fake_mod3 = _fake_urllib_with_body('{"access_token":"at","refresh_token":1,"expires_in":"x"}')
    _fake_import = _import_override_with(fake_mod3)

    monkeypatch.setattr(routes, "redis_for_kv", _rf)
    monkeypatch.setattr(routes, "__import__", _fake_import)
    client = TestClient(create_app())
    r = client.get(
        f"/v1/wrapped/auth/spotify/callback?code=ay&state={st}&callback=http://localhost/cb"
    )
    assert r.status_code == 502
