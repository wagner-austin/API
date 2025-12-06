from __future__ import annotations

import pytest
from _pytest.monkeypatch import MonkeyPatch

from data_bank_api.config import settings_from_env


def _set_required(monkeypatch: MonkeyPatch, upload: str) -> None:
    monkeypatch.setenv("API_UPLOAD_KEYS", upload)
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    monkeypatch.delenv("API_READ_KEYS", raising=False)
    monkeypatch.delenv("API_DELETE_KEYS", raising=False)


def test_missing_upload_keys_raises(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("API_UPLOAD_KEYS", raising=False)
    with pytest.raises(RuntimeError):
        settings_from_env()


def test_upload_keys_required_and_trimmed(monkeypatch: MonkeyPatch) -> None:
    _set_required(monkeypatch, " a , a , b ,  c ")
    s = settings_from_env()
    assert s["api_upload_keys"] == frozenset({"a", "b", "c"})
    assert s["api_read_keys"] == s["api_upload_keys"]
    assert s["api_delete_keys"] == s["api_upload_keys"]


def test_read_delete_override(monkeypatch: MonkeyPatch) -> None:
    _set_required(monkeypatch, "u1,u2")
    monkeypatch.setenv("API_READ_KEYS", "r1, r2")
    monkeypatch.setenv("API_DELETE_KEYS", "d1")
    s = settings_from_env()
    assert s["api_upload_keys"] == frozenset({"u1", "u2"})
    assert s["api_read_keys"] == frozenset({"r1", "r2"})
    assert s["api_delete_keys"] == frozenset({"d1"})
