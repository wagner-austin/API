from __future__ import annotations

import sys
from types import ModuleType
from typing import Protocol

import pytest

from transcript_api.settings import (
    build_clients_from_env,
    build_config_from_env,
)


class _FakeOpenAIClient(Protocol):
    pass


def test_build_config_from_env_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIPT_MAX_VIDEO_SECONDS", "120")
    monkeypatch.setenv("TRANSCRIPT_ENABLE_CHUNKING", "1")
    monkeypatch.setenv("TRANSCRIPT_STT_RTF", "0.6")

    cfg = build_config_from_env()
    assert cfg["TRANSCRIPT_MAX_VIDEO_SECONDS"] == 120
    assert cfg["TRANSCRIPT_ENABLE_CHUNKING"] is True
    assert cfg["TRANSCRIPT_STT_RTF"] == 0.6


def test_build_clients_from_env_requires_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_AI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        _ = build_clients_from_env()

    # Provide fake openai module before building (strict signature)
    class _Fake:
        pass

    def _factory(*, api_key: str, timeout: float, max_retries: int) -> _FakeOpenAIClient:
        return _Fake()

    class _OpenAIFactory(Protocol):
        def __call__(
            self, *, api_key: str, timeout: float, max_retries: int
        ) -> _FakeOpenAIClient: ...

    class _OpenAIModule(ModuleType):
        OpenAI: _OpenAIFactory

    mod = _OpenAIModule("openai")
    mod.OpenAI = _factory
    monkeypatch.setitem(sys.modules, "openai", mod)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    clients = build_clients_from_env()
    # We only assert the shapes; adapters are validated elsewhere
    assert "youtube" in clients and "stt" in clients and "probe" in clients
