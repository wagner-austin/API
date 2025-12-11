"""Tests for transcript_api.settings module."""

from __future__ import annotations

import pytest
from platform_core.config import _test_hooks as platform_hooks
from platform_core.testing import make_fake_env

from transcript_api import _test_hooks
from transcript_api.settings import (
    build_clients_from_env,
    build_config_from_env,
)
from transcript_api.types import (
    OpenAIClientProto,
    SupportsToDictRecursive,
    _AudioProto,
    _BinaryFileProto,
    _TranscriptionsProto,
)


class _FakeTranscriptionResult:
    """Fake transcription result implementing SupportsToDictRecursive."""

    def to_dict_recursive(
        self,
    ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]:
        return {"text": "", "segments": []}


class _FakeTranscriptions:
    """Fake transcriptions implementing _TranscriptionsProto."""

    def create(
        self,
        *,
        model: str,
        file: _BinaryFileProto,
        response_format: str,
        timeout: float | None,
    ) -> SupportsToDictRecursive:
        return _FakeTranscriptionResult()


class _FakeAudio:
    @property
    def transcriptions(self) -> _TranscriptionsProto:
        return _FakeTranscriptions()


class _FakeOpenAIClient:
    @property
    def audio(self) -> _AudioProto:
        return _FakeAudio()


def test_build_config_from_env_defaults_and_overrides() -> None:
    """Test build_config_from_env reads from environment."""
    platform_hooks.get_env = make_fake_env(
        {
            "TRANSCRIPT_MAX_VIDEO_SECONDS": "120",
            "TRANSCRIPT_ENABLE_CHUNKING": "1",
            "TRANSCRIPT_STT_RTF": "0.6",
        }
    )

    cfg = build_config_from_env()
    assert cfg["TRANSCRIPT_MAX_VIDEO_SECONDS"] == 120
    assert cfg["TRANSCRIPT_ENABLE_CHUNKING"] is True
    assert cfg["TRANSCRIPT_STT_RTF"] == 0.6


def test_build_clients_from_env_requires_openai_key() -> None:
    """Test build_clients_from_env raises when OPENAI_API_KEY missing."""
    platform_hooks.get_env = make_fake_env({})

    with pytest.raises(RuntimeError):
        _ = build_clients_from_env()


def test_build_clients_from_env_with_key() -> None:
    """Test build_clients_from_env builds clients when key provided."""
    platform_hooks.get_env = make_fake_env({"OPENAI_API_KEY": "test-key"})

    def _fake_openai_factory(
        *, api_key: str, timeout: float, max_retries: int
    ) -> OpenAIClientProto:
        return _FakeOpenAIClient()

    _test_hooks.openai_client_factory = _fake_openai_factory

    clients = build_clients_from_env()
    # We only assert the shapes; adapters are validated elsewhere
    assert "youtube" in clients and "stt" in clients and "probe" in clients
