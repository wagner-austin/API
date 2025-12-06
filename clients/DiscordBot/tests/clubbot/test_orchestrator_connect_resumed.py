from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine

import pytest

from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator

EventCoro = Callable[[], Coroutine[None, None, None]]


@pytest.mark.asyncio
async def test_on_connect_and_resumed_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Minimal container and bot
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    monkeypatch.setenv("TRANSCRIPT_PROVIDER", "api")
    monkeypatch.setenv("TRANSCRIPT_API_URL", "http://localhost:8000")
    cont: ServiceContainer = ServiceContainer.from_env()
    orch = BotOrchestrator(cont)
    bot = orch.build_bot()
    captured: dict[str, list[EventCoro]] = {"connect": [], "resumed": []}

    original_add_listener = bot.add_listener

    def _add_listener(coro: EventCoro, name: str | None = None) -> None:
        name_attr: str = coro.__name__ if hasattr(coro, "__name__") else ""
        coro_name: str = name if name is not None else name_attr
        event = coro_name.removeprefix("on_")
        captured.setdefault(event, []).append(coro)
        # Call through to real add_listener to keep behavior consistent
        original_add_listener(coro, name=coro_name)

    monkeypatch.setattr(bot, "add_listener", _add_listener)

    # Register listeners and call captured connect/resumed handlers
    orch.register_listeners()
    for fn in captured.get("connect", []):
        await fn()
    for fn in captured.get("resumed", []):
        await fn()


logger = logging.getLogger(__name__)
