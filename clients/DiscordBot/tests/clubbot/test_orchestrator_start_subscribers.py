from __future__ import annotations

import logging

from discord.ext import commands
from tests.support.settings import build_settings

from clubbot import _test_hooks
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
    _ = orch.build_bot()

    dg = _Cog()
    tr = _Cog()

    def fake_get_cog(name: str) -> _test_hooks.CogLike | None:
        if name == "DigitsCog":
            result: _test_hooks.CogLike = dg
            return result
        if name == "TrainerCog":
            result2: _test_hooks.CogLike = tr
            return result2
        return None

    original = _test_hooks.orchestrator_get_cog_override
    _test_hooks.orchestrator_get_cog_override = fake_get_cog
    try:
        orch.start_background_subscribers()
        assert dg.n == 1
        assert tr.n == 1
    finally:
        _test_hooks.orchestrator_get_cog_override = original


logger = logging.getLogger(__name__)
