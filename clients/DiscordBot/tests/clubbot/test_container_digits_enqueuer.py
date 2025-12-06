from __future__ import annotations

import logging

import discord
import pytest
from discord.ext import commands
from monorepo_guards._types import UnknownJson
from tests.support.settings import build_settings

from clubbot.container import ServiceContainer
from clubbot.services.digits.app import DigitService
from clubbot.services.qr.client import QRService


@pytest.mark.asyncio
async def test_container_wires_digits_enqueuer_when_redis_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Build typed settings explicitly (avoid ambient env)
    cfg = build_settings(handwriting_api_url="http://localhost:1234", redis_url="redis://fake")

    created: dict[str, UnknownJson] = {}

    class _FakeRQDigitsEnqueuer:
        def __init__(self, *, redis_url: str, **_: UnknownJson) -> None:
            created["redis_url"] = redis_url

    def _builder(redis_url: str) -> _FakeRQDigitsEnqueuer:
        return _FakeRQDigitsEnqueuer(redis_url=redis_url)

    monkeypatch.setattr("clubbot.container._build_digits_enqueuer", _builder, raising=True)

    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg), digits_service=DigitService(cfg))
    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    await cont.wire_bot_async(bot)
    assert created.get("redis_url") == "redis://fake"
    assert "DigitsCog" in bot.cogs


@pytest.mark.asyncio
async def test_container_handles_missing_rq_digits_enqueuer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Build typed settings explicitly (avoid ambient env)
    cfg = build_settings(handwriting_api_url="http://localhost:1234", redis_url="redis://fake")

    # Provide a placeholder class that raises to trigger the exception path
    class _MissingRQDigitsEnqueuer:
        def __init__(self, *a: UnknownJson, **k: UnknownJson) -> None:
            raise ImportError("missing")

    monkeypatch.setattr(
        "clubbot.services.jobs.digits_enqueuer.RQDigitsEnqueuer",
        _MissingRQDigitsEnqueuer,
        raising=True,
    )

    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg), digits_service=DigitService(cfg))
    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    # Should not raise even if enqueuer cannot be constructed
    await cont.wire_bot_async(bot)
    assert "DigitsCog" in bot.cogs


logger = logging.getLogger(__name__)
