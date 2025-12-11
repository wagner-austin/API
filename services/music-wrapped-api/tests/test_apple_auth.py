from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis

from music_wrapped_api import _test_hooks
from music_wrapped_api.api.main import create_app


def test_auth_apple_store_success() -> None:
    fr = FakeRedis()
    _test_hooks.redis_factory = lambda url: fr

    client = TestClient(create_app())
    payload = {"music_user_token": "apple-user-token"}
    r = client.post("/v1/wrapped/auth/apple/store", json=payload)
    assert r.status_code == 200
    body = load_json_str(r.text)
    if not isinstance(body, dict):
        raise AssertionError("response must be an object")
    tok_val: JSONValue | None = body.get("token_id")
    if not isinstance(tok_val, str):
        raise AssertionError("token_id must be a string")
    tok = tok_val
    assert len(tok) == 32
    stored = fr.hgetall(f"apple:session:{tok}")
    assert stored.get("music_user_token") == "apple-user-token"
    fr.assert_only_called({"sadd", "hset", "expire", "hgetall"})


def test_auth_apple_store_invalid() -> None:
    fr = FakeRedis()
    _test_hooks.redis_factory = lambda url: fr

    client = TestClient(create_app())
    bad: dict[str, JSONValue] = {"music_user_token": 1}
    r = client.post("/v1/wrapped/auth/apple/store", json=bad)
    assert r.status_code == 400
    fr.assert_only_called({"sadd"})
