"""Bring-up and teardown shared by every capture helper process.

A streamed session owns three helpers: the Xvfb server Chromium draws
on, the PulseAudio server Chromium plays into, and the ffmpeg that
records both. The two servers come ready the same way, by binding a
Unix socket their clients dial, and all three end the same way, a
polite SIGTERM that escalates to SIGKILL. Those two procedures live
here once, so the display (:mod:`tankpit_bot.stream.capture`) and the
sound (:mod:`tankpit_bot.stream.audio`) cannot drift apart on a
deadline or on what a helper's death is reported as.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks as root_hooks
from tankpit_bot.stream import _test_hooks

log = get_logger(__name__)

SOCKET_READY_TIMEOUT_SECONDS = 10.0
"""How long to wait for a server's socket before declaring it dead.

Xvfb and PulseAudio each bind their socket within tens of milliseconds
on an idle machine; ten seconds covers a container under heavy fleet
spawn load with a wide margin, and past it the honest reading is that
the server is not coming up.
"""

SOCKET_POLL_INTERVAL_SECONDS = 0.05
"""Cadence of the readiness poll. Short, because readiness gates the
browser launch and every tick of waiting here is startup latency."""

PROCESS_END_TIMEOUT_SECONDS = 5.0
"""How long :func:`end_process` waits after SIGTERM before escalating.

ffmpeg flushes and finalises the open segment on SIGTERM in well under
a second, and the two servers exit at once; a helper that has not
exited after five is stuck, and the session teardown behind this call
must not hang on it.
"""


class CaptureError(Exception):
    """A capture helper failed to start, come ready, or already ran."""


def await_socket(
    process: _test_hooks.CaptureProcessProtocol, name: str, socket: Path, log_path: Path
) -> None:
    """Block until a helper server's socket exists.

    Args:
        process: The server process, polled so an early death is
            reported as what it is rather than as a timeout.
        name: The program's name, for the error.
        socket: The socket path the server binds when it is ready.
        log_path: Where the server's console went, named in errors so
            the reader is one ``cat`` from the real reason.

    Raises:
        CaptureError: The server exited, or the deadline passed.
    """
    deadline = _test_hooks.monotonic_seconds() + SOCKET_READY_TIMEOUT_SECONDS
    while True:
        code = process.poll()
        if code is not None:
            raise CaptureError(f"{name} exited {code} before {socket} appeared; see {log_path}")
        if root_hooks.path_exists(socket):
            return
        if _test_hooks.monotonic_seconds() >= deadline:
            raise CaptureError(
                f"{socket} not ready after {SOCKET_READY_TIMEOUT_SECONDS}s;"
                f" {name} pid {process.pid} still running, see {log_path}"
            )
        _test_hooks.sleep_seconds(SOCKET_POLL_INTERVAL_SECONDS)


def end_process(process: _test_hooks.CaptureProcessProtocol, name: str) -> None:
    """Terminate one helper, escalating to kill if it lingers.

    The ``TimeoutExpired`` arm is a typed translation, not a swallow:
    that one exception means "still running", which is exactly the
    state the escalation exists for, and every other failure
    propagates.

    Args:
        process: The helper to end.
        name: Human name for the log line.
    """
    if process.poll() is not None:
        log.info("Capture: %s pid %d already exited %d", name, process.pid, process.poll())
        return
    process.terminate()
    try:
        code = process.wait(PROCESS_END_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        log.warning(
            "Capture: %s pid %d ignored SIGTERM for %.0fs; killing",
            name,
            process.pid,
            PROCESS_END_TIMEOUT_SECONDS,
        )
        process.kill()
        code = process.wait(PROCESS_END_TIMEOUT_SECONDS)
    log.info("Capture: %s pid %d ended %d", name, process.pid, code)


__all__ = [
    "PROCESS_END_TIMEOUT_SECONDS",
    "SOCKET_POLL_INTERVAL_SECONDS",
    "SOCKET_READY_TIMEOUT_SECONDS",
    "CaptureError",
    "await_socket",
    "end_process",
]
