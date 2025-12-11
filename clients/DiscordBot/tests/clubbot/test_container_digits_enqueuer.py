from __future__ import annotations

import logging

import discord
import pytest
from discord.ext import commands
from monorepo_guards._types import UnknownJson
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.container import ServiceContainer
from clubbot.services.digits.app import DigitService
from clubbot.services.qr.client import QRService


@pytest.mark.asyncio
async def test_container_wires_digits_enqueuer_when_redis_present() -> None:
    # Build typed settings explicitly (avoid ambient env)
    cfg = build_settings(handwriting_api_url="http://localhost:1234", redis_url="redis://fake")

    created: dict[str, UnknownJson] = {}

    class _FakeRQDigitsEnqueuer:
        def __init__(self, *, redis_url: str, **_: UnknownJson) -> None:
            created["redis_url"] = redis_url

        def enqueue_train(
            self,
            *,
            request_id: str,
            user_id: int,
            model_id: str,
            epochs: int,
            batch_size: int,
            lr: float,
            seed: int,
            augment: bool,
            notes: str | None = None,
        ) -> str:
            _ = (request_id, user_id, model_id, epochs, batch_size, lr, seed, augment, notes)
            return "fake-job-id"

    def _builder(redis_url: str) -> _test_hooks.DigitsEnqueuerLike | None:
        return _FakeRQDigitsEnqueuer(redis_url=redis_url)

    _test_hooks.build_digits_enqueuer = _builder

    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg), digits_service=DigitService(cfg))
    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    await cont.wire_bot_async(bot)
    assert created.get("redis_url") == "redis://fake"
    assert "DigitsCog" in bot.cogs


@pytest.mark.asyncio
async def test_container_handles_missing_rq_digits_enqueuer() -> None:
    # Build typed settings explicitly (avoid ambient env)
    cfg = build_settings(handwriting_api_url="http://localhost:1234", redis_url="redis://fake")

    # Return None to simulate enqueuer not available
    def _builder_returns_none(redis_url: str) -> None:
        _ = redis_url
        return

    _test_hooks.build_digits_enqueuer = _builder_returns_none

    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg), digits_service=DigitService(cfg))
    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    # Should not raise even if enqueuer cannot be constructed (returns None)
    await cont.wire_bot_async(bot)
    assert "DigitsCog" in bot.cogs


logger = logging.getLogger(__name__)
