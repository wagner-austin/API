from __future__ import annotations

import logging
from pathlib import Path

import discord
import pytest
from discord.ext import commands

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
async def test_container_digits_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Configure env to enable digits
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    monkeypatch.setenv("TRANSCRIPT_PROVIDER", "api")
    monkeypatch.setenv("TRANSCRIPT_API_URL", "http://localhost:8000")
    monkeypatch.setenv("HANDWRITING_API_URL", "http://localhost:1234")
    # Keep other optional vars unset
    cont = ServiceContainer.from_env()
    if cont.digits_service is None:
        raise AssertionError("expected digits_service")

    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    # Force enqueuer path by setting REDIS_URL and patching builder
    monkeypatch.setenv("REDIS_URL", "redis://fake")
    cont_mod = __import__("clubbot.container", fromlist=["_build_digits_enqueuer"])

    fake_enqueuer: DigitsEnqueuer = _FakeEnqueuer()

    def fake_build_enqueuer(url: str) -> DigitsEnqueuer | None:
        return fake_enqueuer

    monkeypatch.setattr(cont_mod, "_build_digits_enqueuer", fake_build_enqueuer, raising=True)

    await cont.wire_bot_async(bot)
    assert "DigitsCog" in bot.cogs


logger = logging.getLogger(__name__)
