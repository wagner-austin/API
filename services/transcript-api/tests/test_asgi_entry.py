"""Tests for transcript_api.asgi module."""

from __future__ import annotations

import logging
import sys

from platform_core.config import _test_hooks as platform_hooks
from platform_core.testing import make_fake_env

from transcript_api import _test_hooks
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


def test_asgi_app_builds() -> None:
    """Test that asgi app can be built with test dependencies."""
    platform_hooks.get_env = make_fake_env({"OPENAI_API_KEY": "test-key"})

    def _fake_openai_factory(
        *, api_key: str, timeout: float, max_retries: int
    ) -> OpenAIClientProto:
        return _FakeOpenAIClient()

    _test_hooks.openai_client_factory = _fake_openai_factory

    # Remove cached module to force reimport
    sys.modules.pop("transcript_api.asgi", None)

    from transcript_api.asgi import app as asgi_app

    # Verify app has title attribute by accessing it directly
    assert asgi_app.title == "transcript-api"


logger = logging.getLogger(__name__)
