from __future__ import annotations

import logging

import pytest
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.config import DiscordbotSettings
from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator


@pytest.mark.asyncio
async def test_sync_global_respects_disabled_flag() -> None:
    cfg = build_settings(
        transcript_provider="api",
        transcript_api_url="http://localhost:8000",
        commands_sync_global=False,
    )

    def _test_load_settings() -> DiscordbotSettings:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    orchestrator = BotOrchestrator(ServiceContainer.from_env())
    orchestrator.build_bot()
    ok = await orchestrator._sync_global()
    assert ok is False


logger = logging.getLogger(__name__)
