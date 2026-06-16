"""Factory for creating probe instances with injected services.

The factory creates CDPService and CommandService, wires them together,
and injects them into ProbeBase. This is the canonical way to create
probes with properly wired DI.
"""

from __future__ import annotations

from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.browser.cdp_utils import send_websocket_bytes

ProbeT = type[ProbeBase]


def create_probe_services() -> tuple[CDPService, CommandService]:
    """Create the service pair for a probe session.

    Returns:
        Tuple of (CDPService, CommandService) ready for injection.
    """
    cdp_service = CDPService()
    commands = CommandService(send_ws_bytes=send_websocket_bytes)
    return cdp_service, commands


def create_probe(
    probe_class: type[ProbeBase],
    target_url: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
) -> ProbeBase:
    """Create a probe with factory-wired services.

    Args:
        probe_class: Probe subclass to instantiate.
        target_url: Game URL.
        headless: Whether to run headless.
        prefer_account: Whether to prefer account login.

    Returns:
        Probe instance with injected CDPService and CommandService.
    """
    cdp_service, commands = create_probe_services()
    return probe_class(
        target_url,
        headless=headless,
        prefer_account=prefer_account,
        cdp_service=cdp_service,
        command_service=commands,
    )


__all__ = [
    "create_probe",
    "create_probe_services",
]
