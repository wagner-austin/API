from __future__ import annotations

import logging

from tests.support.settings import build_settings

from clubbot.config import DiscordbotSettings
from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator
from clubbot.services.qr.client import QRService


def test_preflight_token_check_valid() -> None:
    """Test that preflight check accepts a valid token."""
    cfg: DiscordbotSettings = build_settings(discord_token="abc.def.ghi")
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    # Should not raise with valid token format
    orch._preflight_token_check()


def test_preflight_token_check_rejects_bot_prefix() -> None:
    """Test that preflight check rejects token with 'Bot ' prefix."""
    import pytest

    cfg: DiscordbotSettings = build_settings(discord_token="Bot abc.def.ghi")
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    with pytest.raises(RuntimeError, match="without the 'Bot ' prefix"):
        orch._preflight_token_check()


logger = logging.getLogger(__name__)
