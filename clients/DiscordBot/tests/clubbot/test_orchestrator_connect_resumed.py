from __future__ import annotations

import logging

import pytest
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.config import DiscordbotSettings
from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator


@pytest.mark.asyncio
async def test_on_connect_and_resumed_logs(caplog: pytest.LogCaptureFixture) -> None:
    # Configure settings via hook
    cfg = build_settings(
        transcript_provider="api",
        transcript_api_url="http://localhost:8000",
    )

    def _test_load_settings() -> DiscordbotSettings:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    cont: ServiceContainer = ServiceContainer.from_env()
    orch = BotOrchestrator(cont)
    bot = orch.build_bot()

    # Register listeners (this adds on_connect, on_resumed to the bot)
    orch.register_listeners()

    # Get the listeners from the bot's extra_events
    connect_listeners = bot.extra_events.get("on_connect", [])
    resumed_listeners = bot.extra_events.get("on_resumed", [])

    # Call each listener and verify logging
    with caplog.at_level(logging.INFO):
        for listener in connect_listeners:
            await listener()
        for listener in resumed_listeners:
            await listener()

    # Verify logs were emitted
    messages = [r.message for r in caplog.records]
    assert any("Gateway connected" in m for m in messages)
    assert any("Gateway resumed" in m for m in messages)


logger = logging.getLogger(__name__)
