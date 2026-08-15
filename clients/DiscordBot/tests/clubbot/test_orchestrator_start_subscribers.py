from __future__ import annotations

import logging

import pytest
from discord.ext import commands
from tests.support.settings import build_settings

from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator
from clubbot.services.qr.client import QRService


class _CountingSubscriberCog(commands.Cog):
    """Cog that satisfies SubscriberCog and counts the starts it receives."""

    def __init__(self) -> None:
        self.n = 0

    def ensure_subscriber_started(self) -> None:
        self.n += 1


class DigitsCog(_CountingSubscriberCog):
    """Registered under the name start_background_subscribers looks up."""


class TrainerCog(_CountingSubscriberCog):
    """Registered under the name start_background_subscribers looks up."""


@pytest.mark.asyncio
async def test_orchestrator_starts_background_subscribers() -> None:
    """Cogs added to the real bot are found by name and started once each."""
    cfg = build_settings()
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    bot = orch.build_bot()

    digits = DigitsCog()
    trainer = TrainerCog()
    await bot.add_cog(digits)
    await bot.add_cog(trainer)

    orch.start_background_subscribers()

    assert digits.n == 1
    assert trainer.n == 1


logger = logging.getLogger(__name__)
