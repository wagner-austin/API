"""The fleet manager's Ctrl+C exit is a log line, not a traceback.

Live 2026-09-01 (operator paste): Ctrl+C landed mid stats poll and the
interrupt's traceback printed twice — once from the unwinding loop,
once from the orphaned request task's "exception was never retrieved"
warning. ``main`` now absorbs the interrupt: the bots are separate
processes with their own stop-file teardown, so a manager interrupt
has nothing to clean up and nothing worth a stack trace.
"""

from __future__ import annotations

from aiohttp import web

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.fleet import main
from tests.conftest import FakeEnv


def test_ctrl_c_ends_main_without_a_traceback() -> None:
    """A KeyboardInterrupt out of the web runner returns, never raises."""
    served: list[tuple[str, int]] = []

    def fake_dotenv() -> None:
        return None

    def fake_run(app: web.Application, *, host: str, port: int) -> None:
        _ = app
        served.append((host, port))
        raise KeyboardInterrupt

    original_dotenv = core_hooks.load_dotenv
    original_run = service_hooks.run_web_app
    original_get_env = top_hooks.get_env
    try:
        core_hooks.load_dotenv = fake_dotenv
        service_hooks.run_web_app = fake_run
        top_hooks.get_env = FakeEnv({"TANKPIT_FLEET_PORT": "27312"})
        main()
    finally:
        core_hooks.load_dotenv = original_dotenv
        service_hooks.run_web_app = original_run
        top_hooks.get_env = original_get_env

    assert served == [("127.0.0.1", 27312)]
