from __future__ import annotations

import pytest
from platform_core.testing import FakeEnv, make_fake_env

from data_bank_api.config import load_settings


def _set_required(env: FakeEnv, upload: str) -> None:
    env.set("API_UPLOAD_KEYS", upload)
    env.set("REDIS_URL", "redis://ignored")
    env.delete("API_READ_KEYS")
    env.delete("API_DELETE_KEYS")


def test_missing_upload_keys_raises() -> None:
    env = make_fake_env()
    env.delete("API_UPLOAD_KEYS")
    with pytest.raises(RuntimeError):
        load_settings()


def test_upload_keys_required_and_trimmed() -> None:
    env = make_fake_env()
    _set_required(env, " a , a , b ,  c ")
    s = load_settings()
    assert s["api_upload_keys"] == frozenset({"a", "b", "c"})
    assert s["api_read_keys"] == s["api_upload_keys"]
    assert s["api_delete_keys"] == s["api_upload_keys"]


def test_read_delete_override() -> None:
    env = make_fake_env()
    _set_required(env, "u1,u2")
    env.set("API_READ_KEYS", "r1, r2")
    env.set("API_DELETE_KEYS", "d1")
    s = load_settings()
    assert s["api_upload_keys"] == frozenset({"u1", "u2"})
    assert s["api_read_keys"] == frozenset({"r1", "r2"})
    assert s["api_delete_keys"] == frozenset({"d1"})
