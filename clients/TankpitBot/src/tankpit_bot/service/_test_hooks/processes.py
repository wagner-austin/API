"""Process seams: spawning bots, and finding the ones already running.

Everything the fleet does to OS processes. Spawning a bot is half of
it; the other half is adoption -- re-attaching to a child this manager
did not fork, which :class:`subprocess.Popen` cannot express because it
only ever describes a process it started
([[fleet-lifecycle]]).
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Protocol

import psutil

from tankpit_bot.runtime_artifacts import FLEET_LOG_PATH, bot_run_dir


class SpawnedProcessProtocol(Protocol):
    """The child-process surface the fleet manager consumes."""

    @property
    def pid(self) -> int:
        """The child's process id."""
        ...

    def poll(self) -> int | None:
        """Return the exit code, or None while the child runs."""
        ...


class ProcessIdentityProtocol(Protocol):
    """Reads the identity of a running process.

    A bare pid is not an identity: the OS recycles pids, so a manager
    that restarts minutes later and finds "pid 4312 exists" has learned
    nothing about whether that pid is still the tank it spawned. The
    creation time pins it -- the same (pid, create_time) pair psutil
    itself uses -- and it is compared exactly, never within a window.
    """

    def __call__(self, pid: int) -> float | None:
        """Return a process's creation time.

        Args:
            pid: Process id to identify.

        Returns:
            The creation time in epoch seconds, or ``None`` when no
            such process is running. A child that died between being
            spawned and being asked is the ordinary way this is
            ``None``, and it means there will never be anything to
            adopt.
        """
        ...


class OpenAdoptedProcessProtocol(Protocol):
    """Re-attaches to a bot process this manager did not fork.

    The fleet's children outlive their manager on purpose, so a
    restarted manager has to find them again.
    :class:`subprocess.Popen` cannot: it only ever describes a process
    it started.
    """

    def __call__(self, pid: int, created_at: float) -> SpawnedProcessProtocol | None:
        """Return a handle on a live, identity-matched process.

        Args:
            pid: Process id recorded when the bot was spawned.
            created_at: Creation time recorded alongside it.

        Returns:
            A handle satisfying :class:`SpawnedProcessProtocol`, or
            ``None`` when nothing is running under that pid or the
            process running under it was created at a different time
            -- a recycled pid belonging to some unrelated program.
        """
        ...


class SleepSecondsProtocol(Protocol):
    """Blocks the calling thread for a while."""

    def __call__(self, seconds: float) -> None:
        """Sleep.

        Args:
            seconds: How long to block for.
        """
        ...


class SpawnFleetManagerProtocol(Protocol):
    """Launches a fleet manager that outlives the launching process."""

    def __call__(self) -> SpawnedProcessProtocol:
        """Start a fleet manager detached from this process.

        The manager must survive the shell that ran ``make up``: it
        supervises live tanks, and a supervisor that dies with the
        terminal that started it is the orphan problem all over again.

        Returns:
            A handle on the launched manager, so the caller can name
            its pid.
        """
        ...


class SpawnBotProcessProtocol(Protocol):
    """Spawns one bot child process for the fleet manager."""

    def __call__(self, env_overrides: dict[str, str]) -> SpawnedProcessProtocol:
        """Start a bot child with the given environment overrides.

        Args:
            env_overrides: Variables set in the child's environment.

        Returns:
            The spawned process handle.
        """
        ...


#: Bootstrap the fleet child runs: apply ``KEY=VALUE`` argv pairs to
#: its OWN environment, then hand off to the bot entry point. The
#: manager never reads the parent environment — the child inherits it
#: whole (``env=None``) and the per-instance overrides ride in as
#: arguments, applied on the far side of the process boundary where
#: the ``get_env`` seam does not exist yet.
_CHILD_BOOTSTRAP = (
    "import os, sys\n"
    "for pair in sys.argv[1:]:\n"
    "    key, _, value = pair.partition('=')\n"
    "    os.environ[key] = value\n"
    "del sys.argv[1:]\n"
    "from tankpit_bot.bot.entry import main\n"
    "main()\n"
)
# The argv wipe matters: the entry point parses sys.argv, and the
# KEY=VALUE pairs are bootstrap freight, not bot arguments — the
# first live fleet spawn died on "unrecognized arguments" without it.


def _real_spawn_bot_process(env_overrides: dict[str, str]) -> subprocess.Popen[bytes]:
    """Spawn one ``tankpit-bot`` child with instance environment.

    The child runs the existing bot entry point in its own process;
    the per-instance isolation (artifact namespace, stop sentinel,
    account selection) all lands through the environment.

    The child's stdout and stderr go to its OWN
    ``runs/bot/<instance>/console.log``, never to the manager's
    terminal. Inheriting the console (the behavior until 2026-08-28)
    put every tick line and viewport dump of every bot into the
    ``make fleet`` window — N interleaved streams with no instance
    prefix — duplicating what the bot already writes to
    ``latest.log``, and contradicting this service's own rule that
    the manager owns lifecycle while telemetry stays on disk.

    Redirecting to a FILE rather than discarding matters: the
    interpreter prints an uncaught exception's traceback to stderr as
    the process dies, AFTER the bot's file logging is gone. The
    2026-08-28 bad-password run is the case in point — its
    ``latest.log`` ends at "Login errors: Invalid username or
    password." and the ``GameNotJoinedError`` traceback existed only
    on the console. ``DEVNULL`` would have destroyed the one artifact
    that explained the exit.

    Args:
        env_overrides: Variables set in the child's environment
            (``TANKPIT_BOT_INSTANCE`` and friends), layered over the
            inherited parent environment by the child's bootstrap.

    Returns:
        The spawned process handle.
    """
    pairs = [f"{key}={value}" for key, value in env_overrides.items()]
    console = bot_run_dir(env_overrides.get("TANKPIT_BOT_INSTANCE", "")) / "console.log"
    console.parent.mkdir(parents=True, exist_ok=True)
    # Append, not truncate: a restart must not erase the traceback
    # that explains why the previous run of this instance died.
    with console.open("a", encoding="utf-8") as stream:
        return subprocess.Popen(
            [sys.executable, "-c", _CHILD_BOOTSTRAP, *pairs],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


class _AdoptedProcess:
    """A live bot process re-attached to by pid, not by parentage.

    Satisfies :class:`SpawnedProcessProtocol` so the fleet registry
    holds adopted and freshly-spawned bots in the same shape and never
    branches on which it has.

    Holding the :class:`psutil.Process` matters beyond convenience: it
    observes the exit itself, so ``poll`` still answers with a real
    exit code after the tank dies, which a pid looked up fresh at that
    point could not do.
    """

    def __init__(self, process: psutil.Process) -> None:
        """Bind to an already-identified process.

        Args:
            process: A psutil handle whose identity the caller has
                already matched against the recorded creation time.
        """
        self._process = process

    @property
    def pid(self) -> int:
        """The adopted process's id.

        Returns:
            The process id.
        """
        return self._process.pid

    def poll(self) -> int | None:
        """Return the exit code, or None while the bot runs.

        ``is_running`` is checked first so the exit code is only ever
        asked for once there is one -- psutil raises
        :class:`psutil.TimeoutExpired` from ``wait`` on a live
        process, and this class has no business using that as control
        flow. ``is_running`` also re-checks identity, so a recycled
        pid reads as "gone", not as a running bot.

        Returns:
            The exit code once the process has ended, else ``None``.
        """
        if self._process.is_running():
            return None
        return self._process.wait(timeout=0)


def _real_process_identity(pid: int) -> float | None:
    """Read a running process's creation time.

    Args:
        pid: Process id to identify.

    Returns:
        Creation time in epoch seconds, or ``None`` when no process is
        running under that pid.
    """
    try:
        return psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        # The OS boundary, not a softened failure: a child that exited
        # before the manager could record it has no identity to record
        # and nothing to adopt later.
        return None


def _real_open_adopted_process(pid: int, created_at: float) -> SpawnedProcessProtocol | None:
    """Re-attach to a bot process by pid, verifying its identity.

    Args:
        pid: Process id recorded when the bot was spawned.
        created_at: Creation time recorded alongside it.

    Returns:
        A handle on the live process, or ``None`` when the pid is free
        or now belongs to something else.
    """
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Nothing under that pid: the bot finished while no manager
        # was watching.
        return None
    if process.create_time() != created_at:
        return None
    return _AdoptedProcess(process)


#: Bootstrap the detached fleet manager runs. ``sys.executable -c``
#: keeps it in this virtualenv and inherits the working directory, so
#: the manager reads and writes the same ``runs/`` tree the command
#: that launched it was standing in.
_FLEET_BOOTSTRAP = "from tankpit_bot.service.fleet import main\nmain()\n"


def _real_sleep_seconds(seconds: float) -> None:
    """Real implementation using :func:`time.sleep`.

    Args:
        seconds: How long to block for.
    """
    time.sleep(seconds)


def _real_spawn_fleet_manager() -> subprocess.Popen[bytes]:
    """Launch the fleet manager detached, logging to a file.

    Detached rather than a child of the shell that ran ``make up``:
    the manager supervises live tanks, and one that died with the
    terminal that started it would be the orphan problem again, one
    level up.

    Its console goes to :data:`FLEET_LOG_PATH` for the same reason a
    bot's does -- a failed boot prints its traceback to stderr as the
    interpreter dies, after any file logging is gone, and discarding
    that stream would destroy the one artifact explaining why the
    fleet never came up.

    Returns:
        The launched manager's process handle.
    """
    FLEET_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Append, not truncate: a relaunch must not erase the record of
    # why the previous manager stopped.
    with FLEET_LOG_PATH.open("a", encoding="utf-8") as stream:
        return subprocess.Popen(
            [sys.executable, "-c", _FLEET_BOOTSTRAP],
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS,
            close_fds=True,
        )


#: Fleet-manager spawn seam. Tests inject a fake that records env and
#: returns a controllable process double; production spawns the real
#: ``tankpit-bot`` child.
spawn_bot_process: SpawnBotProcessProtocol = _real_spawn_bot_process

#: Sleep seam for the lifecycle CLI's poll loops. Tests inject a fake
#: that records the waits and returns instantly, so a drain is
#: exercised without one.
sleep_seconds: SleepSecondsProtocol = _real_sleep_seconds

#: Detached-manager launch seam behind ``make up``. Tests inject a fake
#: that records the launch instead of opening a console.
spawn_fleet_manager: SpawnFleetManagerProtocol = _real_spawn_fleet_manager

#: Process-identity seam. Production reads the creation time psutil
#: reports; tests inject a fake so no real process is needed.
process_identity: ProcessIdentityProtocol = _real_process_identity

#: Adoption seam — how a restarted manager finds the bots still
#: playing. Production re-attaches through psutil; tests hand back a
#: controllable double, or ``None`` to model a bot that has finished.
open_adopted_process: OpenAdoptedProcessProtocol = _real_open_adopted_process
