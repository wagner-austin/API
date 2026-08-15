from __future__ import annotations

import logging

from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator
from clubbot.services.qr.client import QRService


def test_orchestrator_registry_skip_branch() -> None:
    """A service absent from the registry is skipped before its cog is sought.

    Only digits is registered, so trainer takes the 'continue' branch; digits
    itself has no cog on the bot, which is the not-a-SubscriberCog branch.
    """

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
        assert bot.get_cog("DigitsCog") is None

        orch.start_background_subscribers()
    finally:
        _test_hooks.get_service_registry = original_registry


logger = logging.getLogger(__name__)
