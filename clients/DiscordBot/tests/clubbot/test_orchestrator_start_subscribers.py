from __future__ import annotations

import logging

from discord.ext import commands
from tests.support.settings import build_settings

from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator
from clubbot.services.qr.client import QRService


class _Cog(commands.Cog):
    """Test cog that tracks ensure_subscriber_started calls."""

    def __init__(self) -> None:
        self.n = 0

    def ensure_subscriber_started(self) -> None:
        self.n += 1


def test_orchestrator_starts_background_subscribers() -> None:
    cfg = build_settings()
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    bot = orch.build_bot()

    dg = _Cog()
    tr = _Cog()

    original_get_cog = bot.get_cog

    def fake_get_cog(name: str) -> commands.Cog | None:
        if name == "DigitsCog":
            return dg
        if name == "TrainerCog":
            return tr
        return original_get_cog(name)

    object.__setattr__(bot, "get_cog", fake_get_cog)
    orch.start_background_subscribers()
    assert dg.n == 1
    assert tr.n == 1


logger = logging.getLogger(__name__)
