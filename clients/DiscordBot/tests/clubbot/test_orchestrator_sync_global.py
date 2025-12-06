from __future__ import annotations

import logging

import pytest

from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator


@pytest.mark.asyncio
async def test_sync_global_respects_disabled_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    monkeypatch.setenv("TRANSCRIPT_PROVIDER", "api")
    monkeypatch.setenv("TRANSCRIPT_API_URL", "http://localhost:8000")
    monkeypatch.setenv("COMMANDS_SYNC_GLOBAL", "false")
    orchestrator = BotOrchestrator(ServiceContainer.from_env())
    orchestrator.build_bot()
    ok = await orchestrator._sync_global()
    assert ok is False


logger = logging.getLogger(__name__)
