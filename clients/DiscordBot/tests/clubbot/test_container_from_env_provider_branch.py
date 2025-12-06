from __future__ import annotations

import pytest
from tests.support.settings import build_settings

from clubbot.container import ServiceContainer
from clubbot.services.qr.client import QRService


def test_container_from_env_transcript_provider_stt(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = build_settings(transcript_provider="stt")
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    assert cont.transcript_service is None
