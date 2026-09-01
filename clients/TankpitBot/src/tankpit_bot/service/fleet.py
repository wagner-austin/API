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

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.fleet_manager import (
    FleetManager,
    resolve_fleet_port,
)
from tankpit_bot.service.fleet_routes import make_fleet_app

log = get_logger(__name__)


def main() -> None:
    """Run the ``tankpit-fleet`` manager until interrupted."""
    core_hooks.load_dotenv()
    port = resolve_fleet_port()
    manager = FleetManager()
    app = make_fleet_app(manager)
    log.info("tankpit-fleet listening on 127.0.0.1:%d", port)
    try:
        service_hooks.run_web_app(app, host="127.0.0.1", port=port)
    except KeyboardInterrupt:
        # Ctrl+C lands wherever the loop happens to be -- usually mid
        # stats poll, which used to print the interrupt's traceback
        # twice (once from the unwinding loop, once from the orphaned
        # request task's "exception was never retrieved" warning).
        # The bots are separate processes with their own stop-file
        # teardown, so a manager interrupt has nothing to clean up.
        log.info("fleet manager stopped (Ctrl+C)")


__all__ = [
    "log",
    "main",
]
