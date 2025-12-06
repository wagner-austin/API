from __future__ import annotations

import base64

import pytest
from tests.support.settings import build_settings

from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator
from clubbot.services.qr.client import QRService


def test_preflight_token_check_raises_on_bot_prefix() -> None:
    cfg = build_settings(discord_token="Bot abc")
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    with pytest.raises(RuntimeError):
        orch._preflight_token_check()


def test_token_matches_app_id_roundtrip() -> None:
    cfg = build_settings(discord_token="t")
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    app_id = "1234"
    first = base64.b64encode(app_id.encode()).decode().split("=")[0]
    token = first + ".x.y"
    assert orch._token_matches_app_id(token, app_id) is True
