from __future__ import annotations

import pytest
from _pytest.monkeypatch import MonkeyPatch

from turkic_api.api.config import settings_from_env


def test_missing_api_key_raises(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("TURKIC_DATA_BANK_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        settings_from_env()


def test_defaults_and_overrides(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("TURKIC_DATA_BANK_API_KEY", "secret")
    monkeypatch.delenv("TURKIC_DATA_BANK_API_URL", raising=False)
    monkeypatch.delenv("TURKIC_REDIS_URL", raising=False)
    cfg = settings_from_env()
    assert cfg["redis_url"] == "redis://redis:6379/0"
    assert cfg["data_dir"] == "/data"
    assert cfg["environment"] == "local"
    assert cfg["data_bank_api_url"] == ""
    assert cfg["data_bank_api_key"] == "secret"

    monkeypatch.setenv("TURKIC_REDIS_URL", "redis://override")
    cfg2 = settings_from_env()
    assert cfg2["redis_url"] == "redis://override"
