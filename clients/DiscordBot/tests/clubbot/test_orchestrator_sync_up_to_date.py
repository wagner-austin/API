from __future__ import annotations

import logging

import pytest
from tests.support.settings import build_settings

from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator
from clubbot.services.qr.client import QRService


@pytest.mark.asyncio
async def test_sync_commands_logs_up_to_date(caplog: pytest.LogCaptureFixture) -> None:
    """With global sync disabled, _sync_global reports no work and says so.

    This is the one path where sync_commands completes without touching the
    commands endpoint, so the log line is the only observable result.
    """
    cfg = build_settings(commands_sync_global=False)
    container = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(container)
    orch.build_bot()

    with caplog.at_level(logging.INFO, logger="clubbot.orchestrator"):
        await orch.sync_commands()

    assert "Command sync is up-to-date; no changes applied" in caplog.messages
