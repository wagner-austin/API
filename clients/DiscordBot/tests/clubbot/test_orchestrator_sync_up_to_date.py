from __future__ import annotations

import pytest
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator
from clubbot.services.qr.client import QRService


@pytest.mark.asyncio
async def test_sync_commands_logs_up_to_date() -> None:
    cfg = build_settings(commands_sync_global=True)
    container = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(container)
    orch.build_bot()

    async def _no_changes() -> bool:
        return False

    original = _test_hooks.orchestrator_sync_global_override
    _test_hooks.orchestrator_sync_global_override = _no_changes
    try:
        await orch.sync_commands()
    finally:
        _test_hooks.orchestrator_sync_global_override = original
