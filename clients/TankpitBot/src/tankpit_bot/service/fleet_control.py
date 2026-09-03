"""Drain a fleet manager: the ``tankpit-fleet-down`` transition tool.

Before this existed the only way to stop a fleet was Ctrl+C in the
window it happened to be running in, which killed the manager and left
its bots playing on with nothing supervising them. ``down`` replaces
that with a request the manager can honour properly.

The fleet LIFECYCLE is the container pair — ``make up`` / ``make down``
in the Makefile (operator order 2026-09-02: one system, not two). What
remains here is the drain those targets direct operators to when a
pre-container HOST-mode manager still holds port 27300: a container
manager cannot adopt host bot processes, so the host fleet must land
through its own manager before the first containerized ``make up``.
The host LAUNCHER (``tankpit-fleet-up`` / ``fleet_control.up``) was
deleted 2026-09-03 under the same ruling — old release folders keep
their own copy for fleets started from them.

``down`` is a CLIENT. It owns no state and holds nothing open, so
interrupting it changes nothing about the fleet: it asks the manager
to drain and then watches, and if the operator walks away mid-drain
the manager carries on draining and still exits only once its last
bot has landed. The waiting is the manager's job, not this process's.

``down`` waits without a deadline on purpose. A bot's teardown ends by
quitting to the lobby, and hurrying that along to satisfy a timeout is
how a tank gets left sitting in a live game losing its rank.
"""

from __future__ import annotations

from http.client import HTTPConnection, HTTPException

from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.fleet import FLEET_HOST_DEFAULT
from tankpit_bot.service.fleet_config import resolve_fleet_port
from tankpit_bot.service.fleet_wire import FleetSnapshotDict, decode_fleet_snapshot

log = get_logger(__name__)

#: How long a single request to the manager may take. The manager
#: answers in milliseconds when it is up; nothing on the port fails
#: fast with a refused connection.
REQUEST_TIMEOUT_S = 2.0

#: Cadence of the "is it up yet" / "is it gone yet" checks.
POLL_SECONDS = 0.5


def _request(port: int, method: str, path: str) -> str | None:
    """Send one request to the manager and return its body.

    Args:
        port: Fleet manager port.
        method: HTTP method.
        path: Request path.

    Returns:
        The response body, or ``None`` when nothing is listening --
        the manager is not running.

    Raises:
        RuntimeError: If something answers on the port but not with
            success. A stranger on the fleet's port is a
            misconfiguration to surface, never something to retry
            around.
    """
    conn = HTTPConnection(FLEET_HOST_DEFAULT, port, timeout=REQUEST_TIMEOUT_S)
    try:
        conn.request(method, path)
        response = conn.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        if response.status >= 300:
            raise RuntimeError(
                f"fleet manager on {FLEET_HOST_DEFAULT}:{port} answered "
                f"{method} {path} with {response.status}: {body.strip()}"
            )
        return body
    except (OSError, HTTPException):
        # Nothing listening. Distinguishing "refused" from "no route"
        # would not change what either command does next.
        return None
    finally:
        conn.close()


def fleet_snapshot(port: int) -> FleetSnapshotDict | None:
    """Ask the manager what it is running.

    Args:
        port: Fleet manager port.

    Returns:
        The decoded snapshot, or ``None`` when no manager is
        listening.

    Raises:
        RuntimeError: If the port answers with an error status.
        InvalidJsonError: If the body is not valid JSON.
        JSONTypeError: If the body is not a fleet snapshot -- which
            means something that is not this manager holds the port.
    """
    body = _request(port, "GET", "/bots")
    if body is None:
        return None
    return decode_fleet_snapshot(narrow_json_to_dict(load_json_str(body)))


def _live_names(snapshot: FleetSnapshotDict) -> list[str]:
    """List the instances still running in a snapshot.

    Args:
        snapshot: A decoded snapshot.

    Returns:
        Instance names, in the order the manager reported them.
    """
    return [bot["instance"] for bot in snapshot["bots"] if bot["alive"]]


def down() -> int:
    """Drain every bot, then wait for the manager to exit.

    Returns:
        ``0`` once no manager is listening -- including when none was
        listening to begin with.
    """
    core_hooks.load_dotenv()
    port = resolve_fleet_port()
    snapshot = fleet_snapshot(port)
    if snapshot is None:
        log.info("No fleet manager listening on %s:%d; nothing to stop.", FLEET_HOST_DEFAULT, port)
        return 0

    live = _live_names(snapshot)
    log.info(
        "Draining %d bot(s): %s. Each one tears down and quits to the lobby; "
        "the manager exits after the last has landed.",
        len(live),
        ", ".join(live) or "none",
    )
    _request(port, "POST", "/shutdown")
    return _await_exit(port)


def _await_exit(port: int) -> int:
    """Watch a draining manager until its port goes quiet.

    No deadline: see the module docstring. Interrupting this loop is
    safe -- the manager owns the drain, not this process.

    Args:
        port: Fleet manager port.

    Returns:
        ``0`` once nothing is listening.
    """
    announced: list[str] = []
    while True:
        service_hooks.sleep_seconds(POLL_SECONDS)
        snapshot = fleet_snapshot(port)
        if snapshot is None:
            log.info("Fleet manager has exited; every bot landed.")
            return 0
        live = _live_names(snapshot)
        if live != announced:
            log.info("Still draining %d bot(s): %s", len(live), ", ".join(live) or "none")
            announced = live


__all__ = [
    "POLL_SECONDS",
    "REQUEST_TIMEOUT_S",
    "down",
    "fleet_snapshot",
]
