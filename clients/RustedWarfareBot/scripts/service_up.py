"""Start the match service's door and its workers, detached from any terminal.

Replaces ``scripts/make/door.ps1`` and ``scripts/make/fleet-up.ps1``, the
recipes behind ``make door`` and ``make fleet-up`` ([[harness-match-service]]).
The whole recovery after a Postgres restart is the two of them: the queue rows
survive, and the first worker's startup reap requeues what a dead owner held.

DETACHED means the child leads its own session, reads the null device and
writes its own log (:func:`~rw_bot.harness._test_hooks.spawn_match`), so it
outlives the ``make`` that started it -- a door tied to a terminal's life
takes the queue's only submission surface down with that terminal.

THE PASSWORD IS READ AT LAUNCH and never written to disk. It is taken from the
``platform-postgres`` container's own environment and exists only in the
child's command line, exactly as the PowerShell recipes held it.

Run as ``python -m scripts.service_up door`` or
``python -m scripts.service_up fleet-up``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from rw_bot import RwBotError
from rw_bot.harness import _test_hooks
from rw_bot.harness.play_match import EXIT_OK, TOOL_WALL_SECONDS
from scripts.match_service import SERVICE_PORT_DEFAULT

_PASSWORD_UNREADABLE = "RW-SERVICE-001"
_DOOR_ALREADY_UP = "RW-SERVICE-002"
_DOOR_NOT_LISTENING = "RW-SERVICE-003"
_BAD_COMMAND = "RW-SERVICE-004"

DOOR_COMMAND = "door"
FLEET_COMMAND = "fleet-up"

#: The command that reads the database password out of its container.
PASSWORD_COMMAND = ("docker", "exec", "platform-postgres", "printenv", "POSTGRES_PASSWORD")

#: Where the door writes, relative to the repository root.
DOOR_LOG = "runs/door.log"

#: How long the door may take to listen, and how often it is asked.
DOOR_READY_SECONDS = 30.0
DOOR_POLL_SECONDS = 0.5

#: The worker fleet: how many, which clones they share, and their names.
WORKERS = 8
CLONE_POOL = "0,1,2,3,4,5,6,7"
WORKER_PREFIX = "creepw-"

#: How long to let freshly started workers reach the process table.
FLEET_SETTLE_SECONDS = 10.0

#: Every process's whole command line, one per line, no header.
PROCESS_TABLE_COMMAND = ("ps", "-eo", "args=")

#: A worker's name on its command line, after the module and the DSN.
_WORKER_NAME = re.compile(rf"match_worker .* ({re.escape(WORKER_PREFIX)}\d+) ")


class ServiceLaunchError(RwBotError):
    """The door or a worker could not be started as asked.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of what went wrong.
    """


def database_dsn() -> str:
    """Return the covenant database's DSN with the password read at launch.

    Returns:
        The DSN.

    Raises:
        ServiceLaunchError: ``RW-SERVICE-001`` when the container will not
            say, or says nothing: a worker given no password fails at its
            first claim rather than here, where the cause is still nameable.
    """
    status, output = _test_hooks.run_capture(PASSWORD_COMMAND, TOOL_WALL_SECONDS)
    password = "".join(output).strip()
    if status != 0 or not password:
        raise ServiceLaunchError(
            _PASSWORD_UNREADABLE,
            f"{' '.join(PASSWORD_COMMAND)} exited {status} with "
            f"{'no password' if not password else 'a password'}: {' '.join(output)[:200]!r}",
        )
    return (
        f"host=127.0.0.1 port=55432 user=covenant password={password} "
        "dbname=covenant connect_timeout=10"
    )


def start_door() -> int:
    """Start the door detached, and wait until it listens.

    Returns:
        :data:`~rw_bot.harness.play_match.EXIT_OK` once the door listens.

    Raises:
        ServiceLaunchError: ``RW-SERVICE-002`` when something already listens
            on the door's port, ``RW-SERVICE-003`` when the door exited or did
            not listen within :data:`DOOR_READY_SECONDS`, and as
            :func:`database_dsn`.
        OSError: When the door cannot be started.
    """
    if _test_hooks.probe_port(SERVICE_PORT_DEFAULT, DOOR_POLL_SECONDS) is None:
        raise ServiceLaunchError(
            _DOOR_ALREADY_UP,
            f"something already listens on {SERVICE_PORT_DEFAULT}; stop the door first",
        )
    door = _test_hooks.spawn_match(
        (_test_hooks.read_executable(), "-u", "-m", "scripts.match_service", database_dsn()),
        Path(DOOR_LOG),
    )
    deadline = _test_hooks.monotonic() + DOOR_READY_SECONDS
    while _test_hooks.probe_port(SERVICE_PORT_DEFAULT, DOOR_POLL_SECONDS) is not None:
        status = door.poll()
        if status is not None:
            why = f"it exited {status}"
        elif _test_hooks.monotonic() >= deadline:
            why = f"it was still not up after {DOOR_READY_SECONDS}s"
        else:
            _test_hooks.sleep(DOOR_POLL_SECONDS)
            continue
        raise ServiceLaunchError(
            _DOOR_NOT_LISTENING,
            f"the door (pid {door.pid}) is not listening on {SERVICE_PORT_DEFAULT}: {why}; "
            f"see {DOOR_LOG}",
        )
    _test_hooks.write_line(
        f"door up: pid {door.pid}, http://127.0.0.1:{SERVICE_PORT_DEFAULT}/, log {DOOR_LOG}"
    )
    return EXIT_OK


def running_workers() -> tuple[str, ...]:
    """Name the match workers running now, by the name on their command line.

    Returns:
        Each name once, sorted.

    Raises:
        OSError: When the process table cannot be read.
    """
    _, output = _test_hooks.run_capture(PROCESS_TABLE_COMMAND, TOOL_WALL_SECONDS)
    names = {found.group(1) for line in output if (found := _WORKER_NAME.search(f"{line} "))}
    return tuple(sorted(names))


def start_fleet() -> int:
    """Start every worker of the fleet that is not already running.

    Safe on a half-alive fleet: a running worker is left alone, so its
    in-flight match is not doubled.

    Returns:
        :data:`~rw_bot.harness.play_match.EXIT_OK`.

    Raises:
        ServiceLaunchError: As :func:`database_dsn`.
        OSError: When a worker cannot be started.
    """
    dsn = database_dsn()
    alive = running_workers()
    started = 0
    for number in range(1, WORKERS + 1):
        name = f"{WORKER_PREFIX}{number}"
        if name in alive:
            _test_hooks.write_line(f"{name} already running; leaving it alone")
            continue
        worker = (_test_hooks.read_executable(), "-u", "-m", "scripts.match_worker")
        _test_hooks.spawn_match((*worker, dsn, name, CLONE_POOL), Path(f"runs/{name}.log"))
        started += 1
    _test_hooks.sleep(FLEET_SETTLE_SECONDS)
    _test_hooks.write_line(
        f"fleet up: started {started}, running now: {', '.join(running_workers())}"
    )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``make door`` and ``make fleet-up``.

    Args:
        argv: ``door`` or ``fleet-up``. ``None`` reads the process arguments.

    Returns:
        What :func:`start_door` or :func:`start_fleet` returned.

    Raises:
        ServiceLaunchError: ``RW-SERVICE-004`` for anything but one of the two
            commands, and as the command run.
        OSError: As the command run.
    """
    tokens = list(argv) if argv is not None else _test_hooks.read_argv()
    if tokens == [DOOR_COMMAND]:
        return start_door()
    if tokens == [FLEET_COMMAND]:
        return start_fleet()
    raise ServiceLaunchError(
        _BAD_COMMAND, f"expected {DOOR_COMMAND!r} or {FLEET_COMMAND!r} alone, got {tokens!r}"
    )


if __name__ == "__main__":
    raise SystemExit(main(None))


__all__ = [
    "CLONE_POOL",
    "DOOR_COMMAND",
    "DOOR_LOG",
    "DOOR_POLL_SECONDS",
    "DOOR_READY_SECONDS",
    "FLEET_COMMAND",
    "FLEET_SETTLE_SECONDS",
    "PASSWORD_COMMAND",
    "PROCESS_TABLE_COMMAND",
    "WORKERS",
    "WORKER_PREFIX",
    "ServiceLaunchError",
    "database_dsn",
    "main",
    "running_workers",
    "start_door",
    "start_fleet",
]
