from __future__ import annotations

import logging

import pytest
from tests.support.settings import build_settings

from clubbot.config import DiscordbotSettings as Config
from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator
from clubbot.services.qr.client import QRService


def test_preflight_no_application_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISCORD_APPLICATION_ID", raising=False)
    cfg: Config = build_settings(discord_token="abc.def.ghi")
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    # Should no-op when env is absent
    orch._preflight_token_check()


def test_preflight_app_id_matches_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # '999' base64url is 'OTk5OQ'
    token = "OTk5OQ.x.y"
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "999")
    cfg: Config = build_settings(discord_token=token)
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    orch._preflight_token_check()  # should not raise


def test_run_invokes_preflight_match_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure the preflight check runs inside run() with a matching token/app id
    token = "OTk5OQ.x.y"  # base64url("999").
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "999")
    cfg: Config = build_settings(discord_token=token)
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)

    class _Bot:
        def run(self, tok: str) -> None:
            assert tok == token

    monkeypatch.setattr(orch, "build_bot", lambda: _Bot(), raising=True)
    orch.run()  # exercises preflight branch with equal app id


logger = logging.getLogger(__name__)
