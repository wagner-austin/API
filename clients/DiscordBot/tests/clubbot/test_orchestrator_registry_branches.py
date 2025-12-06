from __future__ import annotations

import logging

import pytest
from discord.ext import commands
from tests.support.settings import build_settings

from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator
from clubbot.services.qr.client import QRService


def test_orchestrator_registry_skip_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only include digits in registry to force 'continue' branch for trainer
    reg_mod = __import__("clubbot.services.registry", fromlist=["SERVICE_REGISTRY"])

    def decode_event(s: str) -> None:
        return None

    monkeypatch.setattr(
        reg_mod,
        "SERVICE_REGISTRY",
        {"digits": {"id": "digits", "channel": "digits:events", "decode_event": decode_event}},
        raising=True,
    )

    cfg = build_settings()
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    bot = orch.build_bot()

    original_get_cog = bot.get_cog

    def fake_get_cog(name: str) -> commands.Cog | None:
        return original_get_cog(name)

    object.__setattr__(bot, "get_cog", fake_get_cog)
    orch.start_background_subscribers()


logger = logging.getLogger(__name__)
