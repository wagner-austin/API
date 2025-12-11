from __future__ import annotations

import logging

import discord
import pytest
from discord.ext import commands
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.config import DiscordbotSettings
from clubbot.container import ServiceContainer
from clubbot.services.jobs.digits_enqueuer import DigitsEnqueuer


class _FakeEnqueuer:
    """Fake enqueuer that implements DigitsEnqueuer Protocol."""

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
        return "fake-job-id"


@pytest.mark.asyncio
async def test_container_digits_wiring() -> None:
    # Configure settings to enable digits
    cfg = build_settings(
        transcript_provider="api",
        transcript_api_url="http://localhost:8000",
        handwriting_api_url="http://localhost:1234",
        redis_url="redis://fake",
    )

    def _test_load_settings() -> DiscordbotSettings:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    fake_enqueuer: DigitsEnqueuer = _FakeEnqueuer()

    def _fake_build_enqueuer(redis_url: str) -> _test_hooks.DigitsEnqueuerLike | None:
        _ = redis_url
        result: _test_hooks.DigitsEnqueuerLike = fake_enqueuer
        return result

    _test_hooks.build_digits_enqueuer = _fake_build_enqueuer

    cont = ServiceContainer.from_env()
    if cont.digits_service is None:
        raise AssertionError("expected digits_service")

    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)

    await cont.wire_bot_async(bot)
    assert "DigitsCog" in bot.cogs


logger = logging.getLogger(__name__)
