from __future__ import annotations

import logging
import sys
from typing import BinaryIO

import pytest

from transcript_api.types import VerboseResponseTD


def test_asgi_app_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provide minimal env so startup can build clients
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    sys.modules.pop("transcript_api.asgi", None)
    # Provide a fake openai module so startup can construct clients without network deps
    import types as _types

    class _OpenAIModule(_types.ModuleType):
        OpenAI: type

    mod = _OpenAIModule("openai")

    class _Transcriptions:
        def create(
            self,
            *,
            model: str,
            file: BinaryIO,
            response_format: str,
            timeout: float | None,
        ) -> VerboseResponseTD:
            return {"text": "", "segments": []}

    class _Audio:
        def __init__(self) -> None:
            self.transcriptions = _Transcriptions()

    class OpenAI:  # matches from openai import OpenAI usage
        def __init__(self, *, api_key: str, timeout: float, max_retries: int) -> None:
            self.audio = _Audio()

    mod.OpenAI = OpenAI
    sys.modules["openai"] = mod
    from transcript_api.asgi import app as asgi_app

    # Verify app has title attribute by accessing it directly
    assert asgi_app.title == "transcript-api"


logger = logging.getLogger(__name__)
