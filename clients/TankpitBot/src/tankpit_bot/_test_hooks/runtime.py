"""Misc runtime hooks: clock, argv, watchdog, signals, forced exit.

Small, otherwise-orphaned process-level hooks live here so they do
not require their own modules — e.g. ``get_argv`` (tests substitute a
fixed list) and ``get_current_time_ms`` (scenario clocks).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Protocol

import psutil
from platform_core.environment_record import (
    HostProbe,
    VersionReader,
    installed_version,
    stdlib_host_probe,
)


def _real_get_current_time_ms() -> int:
    """Real implementation -- wall-clock milliseconds since the Unix epoch.

    Returns:
        Current Unix timestamp in milliseconds.
    """
    return int(time.time() * 1000)


#: Hookable clock. Production code that needs "now in ms" -- the
#: dispatcher's wire-timestamp stamping, the AI scan cooldown, the
#: bot's tick cadence -- imports this name and calls it. Tests that
#: drive multi-tick scenarios with controlled time replace this
#: attribute via save-and-restore so the scenario's clock is the
#: only ms-source in the system.
get_current_time_ms: Callable[[], int] = _real_get_current_time_ms


def _real_get_argv() -> list[str]:
    """Real implementation - returns sys.argv.

    Returns:
        The command line arguments.
    """
    return sys.argv


get_argv: Callable[[], list[str]] = _real_get_argv


class StartWatchdogProtocol(Protocol):
    """Protocol for arming a one-shot daemon watchdog timer."""

    def __call__(self, seconds: float, on_fire: Callable[[], None]) -> None:
        """Arm a daemon timer that calls ``on_fire`` after ``seconds``.

        Args:
            seconds: Delay before firing.
            on_fire: Zero-argument callback to invoke on expiry.
        """
        ...


def _real_start_watchdog(seconds: float, on_fire: Callable[[], None]) -> None:
    """Real implementation - daemon ``threading.Timer``.

    Daemon so a normally exiting process is never kept alive by an
    armed watchdog; the timer simply dies with the process.

    Args:
        seconds: Delay before firing.
        on_fire: Zero-argument callback to invoke on expiry.
    """
    timer = threading.Timer(seconds, on_fire)
    timer.daemon = True
    timer.start()


start_watchdog: StartWatchdogProtocol = _real_start_watchdog


class InstallSignalHandlersProtocol(Protocol):
    """Protocol for registering SIGINT / SIGTERM handlers on the bot CLI.

    Production binds ``signal.signal`` for both signals; tests inject
    a fake that records the registered callback and exercise it
    synchronously without touching process-wide signal state.
    """

    def __call__(self, on_interrupt: Callable[[], None]) -> None:
        """Register ``on_interrupt`` for SIGINT and SIGTERM.

        Args:
            on_interrupt: Zero-argument callback invoked when either
                signal fires.
        """
        ...


def _real_install_signal_handlers(on_interrupt: Callable[[], None]) -> None:
    """Real implementation -- bind SIGINT and SIGTERM to ``on_interrupt``.

    The signal-handler signature ``(signum, frame)`` is adapted to the
    zero-argument ``on_interrupt`` callback by ignoring both arguments.
    SIGINT is what Ctrl+C raises on every platform; SIGTERM is what
    process supervisors (systemd, Docker, ``kill PID``) send for a
    graceful shutdown request. Both go through the same handler so
    callers do not have to discriminate.

    Args:
        on_interrupt: Zero-argument callback to invoke when either
            signal fires.
    """
    import signal as _signal
    from types import FrameType

    def _handle(signum: int, frame: FrameType | None) -> None:
        _ = (signum, frame)
        on_interrupt()

    _signal.signal(_signal.SIGINT, _handle)
    _signal.signal(_signal.SIGTERM, _handle)


install_signal_handlers: InstallSignalHandlersProtocol = _real_install_signal_handlers


# The real implementation IS os._exit -- bound directly, no wrapper.
# It must bypass interpreter teardown: the watchdog fires precisely
# because normal teardown is hung.
force_exit: Callable[[int], None] = os._exit


class KillableProcessProtocol(Protocol):
    """The slice of ``psutil.Process`` the browser-engine kill reads.

    Matches the real API signatures: ``pid`` is an attribute,
    ``name()`` reports the executable name, ``kill()`` terminates
    unconditionally. Both callables raise ``psutil.NoSuchProcess``
    when the process exited between enumeration and the call — the
    OS boundary the kill loop must accept (same law as
    ``service/_test_hooks/processes.py``).
    """

    @property
    def pid(self) -> int:
        """Process id."""
        ...

    def name(self) -> str:
        """Return the process's executable name.

        Raises:
            psutil.NoSuchProcess: The process exited already.
        """
        ...

    def kill(self) -> None:
        """Terminate the process unconditionally.

        Raises:
            psutil.NoSuchProcess: The process exited already.
        """
        ...


_BROWSER_ENGINE_NAMES = frozenset(
    {
        "chrome.exe",
        "chromium.exe",
        "headless_shell.exe",
        "chrome",
        "chromium",
        "headless_shell",
    }
)
"""Executable names Playwright's bundled Chromium runs under. The
node driver (``node.exe``) is deliberately absent: the teardown
remedy must SPARE the driver so it can observe the engine's death
and resolve the pending ``browser.close()`` on the main thread."""


def _kill_browser_engines(candidates: list[KillableProcessProtocol]) -> list[int]:
    """Kill every browser-engine process among the candidates.

    Args:
        candidates: Processes to consider — in production, this
            process's full descendant tree.

    Returns:
        PIDs actually killed. A candidate that exited between
        enumeration and the name read or the kill is skipped — either
        fate (killed by us, exited on its own) removes the process,
        which is all the caller needs.
    """
    killed: list[int] = []
    for candidate in candidates:
        try:
            name = candidate.name().lower()
        except psutil.NoSuchProcess:
            continue
        if name not in _BROWSER_ENGINE_NAMES:
            continue
        try:
            candidate.kill()
        except psutil.NoSuchProcess:
            continue
        killed.append(candidate.pid)
    return killed


def _real_kill_browser_processes() -> list[int]:
    """Real implementation — kill Chromium descendants of this process.

    The second rung of the teardown ladder (``browser/lifecycle.py``):
    when ``browser.close()`` stalls past its grace window, the engine
    is removed directly and the spared driver resolves the close.

    Returns:
        PIDs actually killed.
    """
    return _kill_browser_engines(list(psutil.Process().children(recursive=True)))


#: Browser-engine kill seam. Production arms it inside the teardown
#: watchdog ladder; tests inject a recording fake (the conftest
#: default raises, so an unexpected remedy firing fails the test).
kill_browser_processes: Callable[[], list[int]] = _real_kill_browser_processes


def _git_head_ref(cwd: str) -> str:
    """Return the HEAD commit of the repository at ``cwd``, or ``""``.

    Args:
        cwd: Directory to ask git in — the process's own directory for
            the build stamp, but a real parameter: where to ask IS the
            question.

    Returns:
        The full HEAD sha, or ``""`` when ``cwd`` is not inside a git
        repository — which is a fact about the environment (a release
        tree is a ``git archive`` and HAS no repository), not an
        error to soften.
    """
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _real_resolve_build_ref() -> str:
    """Real implementation — name the build this process runs from.

    Three environment CLASSES answer one question, each by the only
    means it has ([[flag-triage-20260902]] board task 7e766d65; the
    same philosophy as hpc3's image ``git_commit`` baking):

    * A STAMPED environment — the fleet container, whose image bakes
      ``TANKPIT_BUILD_REF`` (the release tag) at build time — answers
      from that variable.
    * A CHECKOUT answers from ``git rev-parse HEAD``.
    * An unstamped, repository-less environment answers ``""`` — the
      artifact records that nothing identified the build, rather than
      inventing one.

    Returns:
        The build reference, or ``""`` in an unstamped environment.
    """
    from tankpit_bot._test_hooks import env as env_hooks

    stamped = env_hooks.get_env("TANKPIT_BUILD_REF")
    if stamped is not None and stamped != "":
        return stamped
    return _git_head_ref(".")


#: Build-reference seam for the session_build stamp. Tests inject a
#: fixed string so artifact assertions stay deterministic and no git
#: subprocess runs per test.
resolve_build_ref: Callable[[], str] = _real_resolve_build_ref


def _real_get_host_probe() -> HostProbe:
    """Real implementation -- the stdlib probe over this machine.

    ``os.cpu_count`` is passed rather than read inside the probe because a
    machine that does not report a count must be refusable, and the arm that
    refuses it has to be reachable without owning such a machine.

    Returns:
        The probe, reading this process's real platform, architecture and
        logical processor count.
    """
    return stdlib_host_probe(os.cpu_count)


#: Hookable host probe, for the machine axis of a run fingerprint
#: (:mod:`tankpit_bot.diagnostics.feature_provenance`). Tests replace this
#: attribute by save-and-restore with
#: :class:`platform_core.testing.FakeHostProbe` so a record can be asserted
#: without depending on the machine the suite happens to run on.
get_host_probe: Callable[[], HostProbe] = _real_get_host_probe


#: Hookable distribution-version reader, for the package axis of a run
#: fingerprint. Bound to the real reader, which RAISES on a missing
#: distribution rather than softening it to a placeholder: a fingerprint that
#: recorded "unknown" would compare equal between two genuinely different
#: environments, which is the one failure a comparability axis must not have.
read_distribution_version: VersionReader = installed_version


__all__ = [
    "InstallSignalHandlersProtocol",
    "KillableProcessProtocol",
    "StartWatchdogProtocol",
    "_git_head_ref",
    "_kill_browser_engines",
    "_real_get_current_time_ms",
    "_real_get_host_probe",
    "_real_kill_browser_processes",
    "_real_resolve_build_ref",
    "force_exit",
    "get_argv",
    "get_current_time_ms",
    "get_host_probe",
    "install_signal_handlers",
    "kill_browser_processes",
    "read_distribution_version",
    "resolve_build_ref",
    "start_watchdog",
]
