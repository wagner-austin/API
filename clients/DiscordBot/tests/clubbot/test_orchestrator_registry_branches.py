from __future__ import annotations

import logging

from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator
from clubbot.services.qr.client import QRService


def test_orchestrator_registry_skip_branch() -> None:
    # Only include digits in registry to force 'continue' branch for trainer
    def decode_event(s: str) -> None:
        return None

    custom_registry: dict[str, _test_hooks.ServiceDef] = {
        "digits": {"id": "digits", "channel": "digits:events", "decode_event": decode_event}
    }

    def _custom_registry() -> dict[str, _test_hooks.ServiceDef]:
        return custom_registry

    original_registry = _test_hooks.get_service_registry
    _test_hooks.get_service_registry = _custom_registry
    try:
        cfg = build_settings()
        cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
        orch = BotOrchestrator(cont)
        bot = orch.build_bot()

        original_get_cog = bot.get_cog

        def fake_get_cog(name: str) -> _test_hooks.CogLike | None:
            cog = original_get_cog(name)
            if cog is not None and hasattr(cog, "ensure_subscriber_started"):
                result: _test_hooks.CogLike = cog
                return result
            return None

        original_cog_override = _test_hooks.orchestrator_get_cog_override
        _test_hooks.orchestrator_get_cog_override = fake_get_cog
        try:
            orch.start_background_subscribers()
        finally:
            _test_hooks.orchestrator_get_cog_override = original_cog_override
    finally:
        _test_hooks.get_service_registry = original_registry


logger = logging.getLogger(__name__)
