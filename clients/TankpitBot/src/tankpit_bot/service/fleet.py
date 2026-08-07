"""AI-operated bot fleet: the ``tankpit-fleet`` entry point.

Built 2026-08-06 for the operating AI rather than the SPA: it runs in a
terminal the operator owns, so an orchestration harness dying can never
kill a live tank, and the AI drives it with plain HTTP.

This module is the composition root and holds nothing else -- the
registry is :mod:`tankpit_bot.service.fleet_manager` and the routes are
:mod:`tankpit_bot.service.fleet_routes`.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.fleet_manager import (
    FleetManager,
    resolve_fleet_port,
)
from tankpit_bot.service.fleet_routes import make_fleet_app

log = get_logger(__name__)


def main() -> None:
    """Run the ``tankpit-fleet`` manager until interrupted."""
    service_hooks.load_dotenv()
    port = resolve_fleet_port()
    manager = FleetManager()
    app = make_fleet_app(manager)
    log.info("tankpit-fleet listening on 127.0.0.1:%d", port)
    service_hooks.run_web_app(app, host="127.0.0.1", port=port)


__all__ = [
    "log",
    "main",
]
